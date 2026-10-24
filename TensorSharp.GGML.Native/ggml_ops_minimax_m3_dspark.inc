// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// DSpark speculative drafter for the MiniMax-M3 executor (see
// ggml_ops_minimax_m3_dspark.h). Included once, by ggml_ops_minimax_m3_exec.cpp,
// after its namespace: it uses the executor's model, slot and builder types.
//
// Math (llama.cpp src/models/dflash.cpp + common/speculative.cpp draft-dspark,
// unslothai minimax-m31 @ ef8f94cc; vLLM's DSpark speculator agrees):
//
//   context, for every position t the TRUNK processes (inside its graph):
//     feat_t = [x_{k1}(t) | ... | x_{kn}(t)]       x_k = the residual entering
//                                                  target layer k (k = n_layer:
//                                                  the final pre-norm stream)
//     g_t = rms_norm(fc . feat_t) * enc.output_norm
//     per draft layer: K_t = rope(rms_norm_head(Wk g_t) * k_norm), V_t = Wv g_t
//     (no attn_norm, no Q, no FFN for context rows)
//   draft at anchor position n (the last sampled token, not yet in the trunk):
//     rows [anchor @ n, mask @ n+1 .. n+B-1]; every row predicts (sample_from_anchor)
//     per layer: x += Wo . attn(rope(rms_head(Wq rms(x)*attn_norm)*q_norm), ...)
//                over [context t with q - t < window] U [the whole block],
//                NON-causal, scale 1/sqrt(head_dim); x += down(swiglu_oai(gate, up))
//     h_i = rms_norm(x_i) * output_norm; base_i = target output . h_i
//     Markov chain, greedy and in-graph: prev_0 = anchor,
//       e_i = markov_w1[prev_i], logits_i = base_i + markov_w2^T e_i,
//       conf_i = sigmoid(conf_proj . [h_i ; e_i] + b), prev_{i+1} = argmax logits_i
//
// Rollback needs nothing: every ring row is masked by the position it holds,
// and a row is marked with its new position before the graph that writes it
// runs, so rows a rejected verify, a rewind or a draft block left behind are
// never context, and nothing ever reads a row as a position it does not hold.

namespace tsg_mm3
{

static const char * const DSPARK_PREFIX = "dspark.";

static std::string dspark_name(const std::string & name)
{
    return std::string(DSPARK_PREFIX) + name;
}

static bool dspark_refuse(const std::string & path, const std::string & why)
{
    const size_t slash = path.find_last_of("/\\");
    const std::string file = slash == std::string::npos ? path : path.substr(slash + 1);
    tsg::report_load_refusal("[minimax-m3] --draft-model %s: %s\n", file.c_str(), why.c_str());
    return false;
}

static int64_t dspark_next_pow2(int64_t v)
{
    int64_t p = 1;
    while (p < v) p <<= 1;
    return p;
}

static size_t dspark_ring_bytes(const dspark_state & ds)
{
    return (size_t) 2 * (size_t) ds.n_layer * (size_t) ds.rows * (size_t) ds.n_head_kv * (size_t) ds.head_dim * sizeof(uint16_t);
}

static bool dspark_scan(model & m, const char * path, std::map<std::string, tensor_source> & sources, size_t * placement_bytes)
{
    *placement_bytes = 0;
    if (!path || !*path) return true;
    dspark_state & ds = m.ds;
    const hparams & hp = m.hp;
    ds.path = path;

    ggml_context * meta = nullptr;
    gguf_init_params ip = { true, &meta };
    gguf_context * g = gguf_init_from_file(path, ip);
    if (!g) return dspark_refuse(ds.path, "cannot be read as a GGUF file");
    struct closer
    {
        gguf_context * g; ggml_context * c;
        ~closer() { gguf_free(g); if (c) ggml_free(c); }
    } close_it { g, meta };

    auto str_key = [&](const char * key) -> std::string
    {
        const int64_t id = gguf_find_key(g, key);
        return id >= 0 && gguf_get_kv_type(g, id) == GGUF_TYPE_STRING ? std::string(gguf_get_val_str(g, id)) : std::string();
    };
    const std::string arch = str_key("general.architecture");
    if (arch != "dflash")
        return dspark_refuse(ds.path, "general.architecture is '" + (arch.empty() ? std::string("missing") : arch) +
                             "'; minimax-m3 drafts with a DSpark drafter (arch dflash with a Markov head)");

    // --- hyper-parameters ---------------------------------------------------
    bool ok = true;
    ok = ok && kv_u32(g, "dflash.block_count", &ds.n_layer);
    ok = ok && kv_u32(g, "dflash.embedding_length", &ds.n_embd);
    ok = ok && kv_u32(g, "dflash.attention.head_count", &ds.n_head);
    ok = ok && kv_u32(g, "dflash.attention.head_count_kv", &ds.n_head_kv);
    ok = ok && kv_u32(g, "dflash.attention.key_length", &ds.head_dim);
    ok = ok && kv_u32(g, "dflash.feed_forward_length", &ds.n_ff);
    ok = ok && kv_u32(g, "dflash.block_size", &ds.block);
    if (!ok) return dspark_refuse(ds.path, "dflash.* metadata (block_count, embedding_length, head counts, key_length, "
                                  "feed_forward_length, block_size) is incomplete");
    int32_t value_len = ds.head_dim;
    kv_u32(g, "dflash.attention.value_length", &value_len);
    ds.n_rot = ds.head_dim;
    kv_u32(g, "dflash.rope.dimension_count", &ds.n_rot);
    kv_f32(g, "dflash.rope.freq_base", &ds.rope_base);
    kv_f32(g, "dflash.attention.layer_norm_rms_epsilon", &ds.rms_eps);
    if (!kv_u32(g, "dflash.attention.sliding_window", &ds.window) || ds.window <= 0)
        return dspark_refuse(ds.path, "has no dflash.attention.sliding_window; the executor keeps a sliding-window "
                             "context per sequence and cannot attend a drafter over the whole sequence");
    {
        const int64_t id = gguf_find_key(g, "dflash.attention.sliding_window_pattern");
        if (id >= 0 && gguf_get_kv_type(g, id) == GGUF_TYPE_ARRAY && gguf_get_arr_type(g, id) == GGUF_TYPE_BOOL)
        {
            const int8_t * swa = (const int8_t *) gguf_get_arr_data(g, id);
            for (size_t i = 0; i < gguf_get_arr_n(g, id); i++)
                if (!swa[i])
                    return dspark_refuse(ds.path, "draft layer " + std::to_string(i) + " attends the whole sequence "
                                         "(sliding_window_pattern false); only sliding-window drafters are supported");
        }
    }
    bool from_anchor = true;
    kv_bool(g, "dflash.sample_from_anchor", &from_anchor);
    if (!from_anchor)
        return dspark_refuse(ds.path, "uses the bonus-anchor block layout (dflash.sample_from_anchor = false), where the "
                             "anchor row is not a prediction; MiniMax-M3.1's DSpark predicts from every row and that is "
                             "the only layout implemented");
    kv_bool(g, "dflash.attention.causal", &ds.causal);
    {
        const std::string act = str_key("dflash.hidden_activation");
        if (!act.empty() && act != "swigluoai")
            return dspark_refuse(ds.path, "hidden_activation is '" + act + "'; the executor runs swigluoai drafters only");
    }
    if (!kv_u32(g, "tokenizer.ggml.mask_token_id", &ds.mask_token))
        return dspark_refuse(ds.path, "has no tokenizer.ggml.mask_token_id (the token filling the draft block)");
    {
        const int64_t id = gguf_find_key(g, "dflash.target_layers");
        if (id < 0 || gguf_get_kv_type(g, id) != GGUF_TYPE_ARRAY ||
            (gguf_get_arr_type(g, id) != GGUF_TYPE_INT32 && gguf_get_arr_type(g, id) != GGUF_TYPE_UINT32))
            return dspark_refuse(ds.path, "has no integer dflash.target_layers array");
        const int32_t * tl = (const int32_t *) gguf_get_arr_data(g, id);
        ds.target_layers.assign(tl, tl + gguf_get_arr_n(g, id));
    }

    // --- against the target -------------------------------------------------
    auto list = [](const std::vector<int32_t> & v)
    {
        std::string s;
        for (int32_t x : v) s += (s.empty() ? "" : ",") + std::to_string(x);
        return "[" + s + "]";
    };
    if (ds.n_embd != hp.n_embd)
        return dspark_refuse(ds.path, "embedding_length " + std::to_string(ds.n_embd) + " does not match the target's " +
                             std::to_string(hp.n_embd) + ": it was trained for another model");
    if (ds.target_layers.empty())
        return dspark_refuse(ds.path, "dflash.target_layers is empty");
    for (size_t i = 0; i < ds.target_layers.size(); i++)
    {
        const int32_t k = ds.target_layers[i];
        if (k < 1 || k > hp.n_layer)
            return dspark_refuse(ds.path, "dflash.target_layers " + list(ds.target_layers) + " names layer input " +
                                 std::to_string(k) + "; this " + std::to_string(hp.n_layer) + "-layer target exports the "
                                 "outputs of layers 0.." + std::to_string(hp.n_layer - 1) + " (inputs 1.." +
                                 std::to_string(hp.n_layer) + "): it was trained for another model");
        for (size_t j = 0; j < i; j++)
            if (ds.target_layers[j] == k)
                return dspark_refuse(ds.path, "dflash.target_layers " + list(ds.target_layers) + " repeats layer " + std::to_string(k));
    }
    if (ds.n_layer < 1 || ds.n_head < 1 || ds.n_head_kv < 1 || ds.n_head % ds.n_head_kv != 0 || ds.head_dim < 1 ||
        value_len != ds.head_dim || ds.n_rot < 2 || ds.n_rot % 2 != 0 || ds.n_rot > ds.head_dim || ds.n_ff < 1 ||
        !(ds.rms_eps > 0.0f) || !(ds.rope_base > 0.0f))
        return dspark_refuse(ds.path, "invalid attention, RoPE or FFN geometry");
    if (ds.mask_token < 0 || ds.mask_token >= hp.n_vocab)
        return dspark_refuse(ds.path, "mask token " + std::to_string(ds.mask_token) + " is outside the target's " +
                             std::to_string(hp.n_vocab) + "-token vocabulary");
    if (ds.block < 1 || ds.block + 1 > m.n_ubatch)
        return dspark_refuse(ds.path, "block_size " + std::to_string(ds.block) + " needs a verification of " +
                             std::to_string(ds.block + 1) + " rows, more than the " + std::to_string(m.n_ubatch) + "-row ubatch");

    // --- tensors --------------------------------------------------------------
    const int64_t E = ds.n_embd, Q = (int64_t) ds.n_head * ds.head_dim, KV = (int64_t) ds.n_head_kv * ds.head_dim;
    std::map<std::string, std::vector<int64_t>> want = {
        { "fc.weight", { (int64_t) ds.target_layers.size() * E, E } },
        { "enc.output_norm.weight", { E } },
        { "output_norm.weight", { E } },
    };
    for (int il = 0; il < ds.n_layer; il++)
    {
        const std::string p = "blk." + std::to_string(il) + ".";
        want[p + "attn_norm.weight"] = { E };
        want[p + "attn_q.weight"] = { E, Q };
        want[p + "attn_k.weight"] = { E, KV };
        want[p + "attn_v.weight"] = { E, KV };
        want[p + "attn_output.weight"] = { Q, E };
        want[p + "attn_q_norm.weight"] = { ds.head_dim };
        want[p + "attn_k_norm.weight"] = { ds.head_dim };
        want[p + "ffn_norm.weight"] = { E };
        want[p + "ffn_gate.weight"] = { E, ds.n_ff };
        want[p + "ffn_up.weight"] = { E, ds.n_ff };
        want[p + "ffn_down.weight"] = { ds.n_ff, E };
    }
    ggml_tensor * w1 = ggml_get_tensor(meta, "markov_w1.weight");
    ggml_tensor * w2 = ggml_get_tensor(meta, "markov_w2.weight");
    if (!w1 || !w2)
        return dspark_refuse(ds.path, "has no Markov head (markov_w1/markov_w2): a plain DFlash drafter, which minimax-m3 "
                             "does not run (its DSpark drafter carries one)");
    ds.rank = (int32_t) w1->ne[0];
    want["markov_w1.weight"] = { ds.rank, hp.n_vocab };
    want["markov_w2.weight"] = { ds.rank, hp.n_vocab };
    const bool has_conf = ggml_get_tensor(meta, "conf_proj.weight") != nullptr;
    if (has_conf)
    {
        want["conf_proj.weight"] = { E + ds.rank, 1 };
        if (ggml_get_tensor(meta, "conf_proj.bias")) want["conf_proj.bias"] = { 1 };
    }
    // F32 where the graph multiplies activations element-wise; matrices any type the backend multiplies.
    auto must_f32 = [](const std::string & n)
    {
        return n.find("norm") != std::string::npos || n.rfind("conf_proj", 0) == 0;
    };
    for (int64_t ti = 0; ti < gguf_get_n_tensors(g); ti++)
    {
        const std::string name = gguf_get_tensor_name(g, ti);
        auto it = want.find(name);
        if (it == want.end())
            return dspark_refuse(ds.path, "carries tensor " + name + ", which the executor's DSpark graph does not implement "
                                 "(its own embedding or head, a reduced draft vocabulary, convolutions or attention sinks)");
        ggml_tensor * mt = ggml_get_tensor(meta, name.c_str());
        bool shape_ok = mt != nullptr;
        for (int d = 0; d < 4 && shape_ok; d++)
            shape_ok = mt->ne[d] == (d < (int) it->second.size() ? it->second[(size_t) d] : 1);
        if (!shape_ok)
        {
            std::string expect;
            for (int64_t v : it->second) expect += (expect.empty() ? "" : ",") + std::to_string(v);
            return dspark_refuse(ds.path, "tensor " + name + " is not [" + expect + "]: it was trained for another model");
        }
        if (must_f32(name) && mt->type != GGML_TYPE_F32)
            return dspark_refuse(ds.path, "tensor " + name + " must be F32, not " + ggml_type_name(mt->type));
    }
    for (const auto & w : want)
        if (!ggml_get_tensor(meta, w.first.c_str()))
            return dspark_refuse(ds.path, "tensor " + w.first + " is missing");

    // --- the drafter's file becomes one more shard -------------------------------
    const int si = (int) m.shards.paths.size();
    const size_t data_off = gguf_get_data_offset(g);
    const size_t fsz = file_size_of(path);
    for (int64_t ti = 0; ti < gguf_get_n_tensors(g); ti++)
    {
        const char * name = gguf_get_tensor_name(g, ti);
        ggml_tensor * mt = ggml_get_tensor(meta, name);
        tensor_source src;
        src.shard = si;
        src.offset = data_off + gguf_get_tensor_offset(g, ti);
        src.size = ggml_nbytes(mt);
        src.type = mt->type;
        for (int d = 0; d < 4; d++) src.ne[d] = mt->ne[d];
        if (fsz != (size_t) -1 && src.offset + src.size > fsz)
            return dspark_refuse(ds.path, std::string("is incomplete (tensor ") + name + " ends past EOF); re-download it");
        sources[dspark_name(name)] = src;
        ds.weight_bytes += src.size;
    }
    m.shards.paths.push_back(path);

    // Ring rows: the window, plus the largest graph's rows (no two rows of one
    // graph share a ring row) and a verify's tail, rounded to a power of two
    // so the trunk graph can take a position modulo it exactly in F32.
    ds.rows = dspark_next_pow2((int64_t) ds.window + std::max<int64_t>(m.n_ubatch, MAX_BATCHED_DECODE) + ds.block + 1);
    // The output head's device also holds the weights, the rings of the first
    // sequence and one checkpoint, and the commit's activations at a full ubatch
    // (the tapped rows gathered there and concatenated).
    const size_t commit = (size_t) m.n_ubatch * (size_t) (2 * ds.target_layers.size() + 2) * (size_t) E * sizeof(float);
    *placement_bytes = ds.weight_bytes + 2 * dspark_ring_bytes(ds) + commit;
    fprintf(stderr, "[minimax-m3] DSpark drafter %s: %d layers, block %d, taps %s, window %d, Markov rank %d%s; "
            "%.2f GiB beside the output head (%.2f GiB weights)\n", ds.path.c_str(), ds.n_layer, ds.block,
            list(ds.target_layers).c_str(), ds.window, ds.rank, has_conf ? ", confidence head" : "",
            *placement_bytes / 1073741824.0, ds.weight_bytes / 1073741824.0);
    return true;
}

static bool dspark_make_weights(model & m, const std::function<ggml_tensor * (int, const char *, bool)> & make)
{
    dspark_state & ds = m.ds;
    if (ds.path.empty()) return true;
    ds.dev = m.out_device;
    auto w = [&](const std::string & name, bool required) { return make(ds.dev, dspark_name(name).c_str(), required); };
    ds.fc = w("fc.weight", true);
    ds.enc_norm = w("enc.output_norm.weight", true);
    ds.out_norm = w("output_norm.weight", true);
    ds.markov_w1 = w("markov_w1.weight", true);
    ds.markov_w2 = w("markov_w2.weight", true);
    ds.conf_w = w("conf_proj.weight", false);
    ds.conf_b = ds.conf_w ? w("conf_proj.bias", false) : nullptr;
    ds.layers.resize((size_t) ds.n_layer);
    for (int il = 0; il < ds.n_layer; il++)
    {
        dspark_layer & L = ds.layers[(size_t) il];
        const std::string p = "blk." + std::to_string(il) + ".";
        L.attn_norm = w(p + "attn_norm.weight", true);
        L.wq = w(p + "attn_q.weight", true);
        L.wk = w(p + "attn_k.weight", true);
        L.wv = w(p + "attn_v.weight", true);
        L.wo = w(p + "attn_output.weight", true);
        L.q_norm = w(p + "attn_q_norm.weight", true);
        L.k_norm = w(p + "attn_k_norm.weight", true);
        L.ffn_norm = w(p + "ffn_norm.weight", true);
        L.ffn_gate = w(p + "ffn_gate.weight", true);
        L.ffn_up = w(p + "ffn_up.weight", true);
        L.ffn_down = w(p + "ffn_down.weight", true);
    }
    return true;
}

/// Token `id`'s embedding row as F32 (the trunk's own rows: the drafter has none).
static void dspark_embed_row(const model & m, int32_t id, float * out)
{
    const ggml_tensor * te = m.tok_embd;
    const size_t row = te->nb[1];
    std::vector<uint8_t> raw(row);
    ggml_backend_tensor_get(te, raw.data(), (size_t) id * row, row);
    if (te->type == GGML_TYPE_F32) memcpy(out, raw.data(), (size_t) te->ne[0] * sizeof(float));
    else ggml_get_type_traits(te->type)->to_float(raw.data(), out, te->ne[0]);
}

static bool dspark_finish_load(model & m)
{
    dspark_state & ds = m.ds;
    if (ds.path.empty()) return true;
    if (!m.tok_embd || (m.tok_embd->type != GGML_TYPE_F32 && !ggml_get_type_traits(m.tok_embd->type)->to_float))
        return dspark_refuse(ds.path, "the target's token_embd type has no row conversion");
    ds.mask_row.resize((size_t) ds.n_embd);
    dspark_embed_row(m, ds.mask_token, ds.mask_row.data());
    ds.graph_cap = (int) std::max(1L, env_long("TS_MINIMAX_M3_DSPARK_GRAPHS", 4));
    ds.loaded = true;
    const std::string where = ds.dev == m.host_dev() ? std::string("the host") : "GPU " + std::to_string(ds.dev);
    fprintf(stderr, "[minimax-m3] DSpark drafter on %s: %" PRId64 "-row context ring per sequence (%.0f MiB), "
            "committed inside every trunk graph\n", where.c_str(), ds.rows, dspark_ring_bytes(ds) / 1048576.0);
    return true;
}

static void dspark_slot_added(model & m, const slot & s)
{
    dspark_state & ds = m.ds;
    if (!ds.loaded) return;
    std::unique_ptr<dspark_ring> r(new dspark_ring());
    ggml_init_params p = { (size_t) (2 * ds.n_layer + 4) * ggml_tensor_overhead(), nullptr, true };
    r->ctx = ggml_init(p);
    for (int il = 0; il < ds.n_layer; il++)
    {
        ggml_tensor * k = ggml_new_tensor_2d(r->ctx, GGML_TYPE_F16, (int64_t) ds.n_head_kv * ds.head_dim, ds.rows);
        ggml_tensor * v = ggml_new_tensor_2d(r->ctx, GGML_TYPE_F16, (int64_t) ds.n_head_kv * ds.head_dim, ds.rows);
        ggml_format_name(k, "slot%d_dspark_k.%d", s.id, il);
        ggml_format_name(v, "slot%d_dspark_v.%d", s.id, il);
        r->k.push_back(k);
        r->v.push_back(v);
    }
    r->buf = ggml_backend_alloc_ctx_tensors(r->ctx, m.backends[ds.dev]);
    if (!r->buf)
    {
        if (!ds.warned_no_ring)
        {
            ds.warned_no_ring = true;
            fprintf(stderr, "[minimax-m3] no memory on device %d for a sequence's %.0f MiB DSpark context (slot %d); "
                    "sequences without one decode without drafts (reported once)\n", ds.dev,
                    dspark_ring_bytes(ds) / 1048576.0, s.id);
        }
        return;
    }
    // Zeroed: rows no position has written are read under a -inf mask, and 0 * -inf would be NaN.
    ggml_backend_buffer_clear(r->buf, 0);
    r->pos.assign((size_t) ds.rows, -1);
    ds.rings[s.id] = std::move(r);
}

static void dspark_slot_removed(model & m, int slot_id)
{
    dspark_state & ds = m.ds;
    for (auto it = ds.graphs.begin(); it != ds.graphs.end(); )
        it = (*it)->slot_id == slot_id ? ds.graphs.erase(it) : std::next(it);
    ds.rings.erase(slot_id);
}

static bool dspark_slot_copy(model & m, const slot & src, const slot & dst)
{
    dspark_state & ds = m.ds;
    if (!ds.loaded) return true;
    auto d = ds.rings.find(dst.id);
    if (d == ds.rings.end()) return true;
    auto s = ds.rings.find(src.id);
    if (s == ds.rings.end())
    {
        // Nothing to copy: the copy drafts from the positions it writes from now on.
        std::fill(d->second->pos.begin(), d->second->pos.end(), -1);
        return true;
    }
    for (int il = 0; il < ds.n_layer; il++)
    {
        ggml_backend_tensor_copy(s->second->k[(size_t) il], d->second->k[(size_t) il]);
        ggml_backend_tensor_copy(s->second->v[(size_t) il], d->second->v[(size_t) il]);
    }
    d->second->pos = s->second->pos;
    return true;
}

static void dspark_rows_written(model & m, const slot & s, int64_t p0, int64_t n)
{
    dspark_state & ds = m.ds;
    if (!ds.loaded) return;
    auto it = ds.rings.find(s.id);
    if (it == ds.rings.end()) return;
    std::vector<int64_t> & pos = it->second->pos;
    const int64_t mask = ds.rows - 1;
    for (int64_t p = std::max<int64_t>(p0, p0 + n - ds.rows); p < p0 + n; p++) pos[(size_t) (p & mask)] = p;
}

static bool dspark_taps_final(const model & m)
{
    const dspark_state & ds = m.ds;
    return ds.loaded && std::find(ds.target_layers.begin(), ds.target_layers.end(), m.hp.n_layer) != ds.target_layers.end();
}

static size_t dspark_graph_nodes(const model & m, size_t n_segments)
{
    const dspark_state & ds = m.ds;
    if (!ds.loaded) return 0;
    return 64 + 2 * ds.target_layers.size() + (size_t) ds.n_layer * (24 + 8 * n_segments);
}

static ggml_tensor * dspark_rms(ggml_context * ctx, const dspark_state & ds, ggml_tensor * x, ggml_tensor * w)
{
    x = ggml_rms_norm(ctx, x, ds.rms_eps);
    return w ? ggml_mul(ctx, x, w) : x;
}

/// NEOX RoPE over the first n_rot dims of each head (llama.cpp's dflash rope for a non-DSV4 drafter).
static ggml_tensor * dspark_rope(ggml_context * ctx, const dspark_state & ds, ggml_tensor * x, ggml_tensor * pos)
{
    return ggml_rope_ext(ctx, x, pos, nullptr, ds.n_rot, GGML_ROPE_TYPE_NEOX, 0, ds.rope_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
}

/// The context commit inside a trunk graph: the tapped rows of every token ->
/// fc -> enc.output_norm -> each draft layer's K and V, written to each
/// segment's slot ring at its positions modulo the ring.
static void dspark_commit(builder & b)
{
    model & m = b.m;
    const dspark_state & ds = m.ds;
    ggml_context * ctx = b.ctx;
    const int dev = ds.dev;
    ggml_tensor * pos = b.g.inp.pos[dev];
    if (!pos) return;   // the drafter sits with the last layer, which always has its positions
    const int64_t T = b.g.nt;
    const int64_t KV = (int64_t) ds.n_head_kv * ds.head_dim;

    ggml_tensor * feat = nullptr;
    for (ggml_tensor * t : b.ds_taps)
    {
        feat = feat ? ggml_concat(ctx, feat, t, 0) : t;   // [n_taps * n_embd, T]
        if (feat != t) b.pin(feat, dev);
    }
    ggml_tensor * gx = dspark_rms(ctx, ds, ggml_mul_mat(ctx, ds.fc, feat), ds.enc_norm);   // [n_embd, T]
#if defined(TSG_GGML_TEST_HOOKS)
    // Test hook: a slow drafter stage. Large F32 products of one commit value
    // ([16384, 4096] with itself, ~0.55 TFLOP in a few kernels each), on the
    // drafter's GPU, that nothing reads. Few kernels on purpose: a stage slowed
    // by many small ones fills CUDA's launch queue, whose backpressure then holds
    // the host, and so every stage, to that stage's pace.
    if (const int n = g_test_ds_slow_products.load(); n > 0)
    {
        ggml_tensor * big = ggml_repeat_4d(ctx, ggml_view_2d(ctx, gx, 1, 1, gx->nb[1], 0), 16384, 4096, 1, 1);
        b.pin(big, dev);
        for (int i = 0; i < n; i++)
        {
            ggml_tensor * p = ggml_mul_mat(ctx, big, big);
            b.pin(p, dev);
            ggml_build_forward_expand(b.gf, p);
        }
    }
#endif

    // ring row = position mod rows, exact in F32 (rows is a power of two)
    ggml_tensor * pf = ggml_cast(ctx, pos, GGML_TYPE_F32);
    b.pin(pf, dev);
    ggml_tensor * wraps = ggml_floor(ctx, ggml_scale(ctx, pf, 1.0f / (float) ds.rows));
    b.pin(wraps, dev);
    ggml_tensor * rf = ggml_add(ctx, pf, ggml_scale(ctx, wraps, -(float) ds.rows));
    b.pin(rf, dev);
    ggml_tensor * ridx = ggml_cast(ctx, rf, GGML_TYPE_I32);
    b.pin(ridx, dev);

    for (int il = 0; il < ds.n_layer; il++)
    {
        const dspark_layer & L = ds.layers[(size_t) il];
        ggml_tensor * k = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, L.wk, gx), ds.head_dim, ds.n_head_kv, T);
        k = dspark_rope(ctx, ds, dspark_rms(ctx, ds, k, L.k_norm), pos);
        ggml_tensor * k2 = ggml_reshape_2d(ctx, k, KV, T);
        ggml_tensor * v2 = ggml_mul_mat(ctx, L.wv, gx);                                       // [KV, T]
        int64_t row = 0;
        for (const bd_token & seg : b.g.bd)
        {
            auto it = ds.rings.find(seg.slot_id);
            if (it != ds.rings.end())
            {
                const dspark_ring & r = *it->second;
                ggml_tensor * ri = ggml_view_1d(ctx, ridx, seg.len, (size_t) row * ggml_element_size(ridx));
                ggml_tensor * ks = ggml_view_2d(ctx, k2, KV, seg.len, k2->nb[1], (size_t) row * k2->nb[1]);
                ggml_tensor * vs = ggml_view_2d(ctx, v2, KV, seg.len, v2->nb[1], (size_t) row * v2->nb[1]);
                ggml_tensor * wk = ggml_set_rows(ctx, r.k[(size_t) il], ks, ri);
                ggml_tensor * wv = ggml_set_rows(ctx, r.v[(size_t) il], vs, ri);
                b.pin(wk, dev);
                b.pin(wv, dev);
                ggml_build_forward_expand(b.gf, wk);
                ggml_build_forward_expand(b.gf, wv);
            }
            row += seg.len;
        }
    }
}

static void dspark_tap(builder & b, int il, ggml_tensor * x)
{
    const dspark_state & ds = b.m.ds;
    if (!ds.loaded) return;
    if (b.ds_taps.size() != ds.target_layers.size()) b.ds_taps.assign(ds.target_layers.size(), nullptr);
    bool mine = false;
    for (size_t j = 0; j < ds.target_layers.size(); j++)
        if (ds.target_layers[j] == il + 1) { b.ds_taps[j] = x; mine = true; }
    if (!mine) return;
    for (ggml_tensor * t : b.ds_taps)
        if (!t) return;
    dspark_commit(b);
}

// ---------------------------------------------------------------------------
// the draft
// ---------------------------------------------------------------------------
static dspark_graph * dspark_acquire_graph(model & m, int slot_id)
{
    dspark_state & ds = m.ds;
    for (auto it = ds.graphs.begin(); it != ds.graphs.end(); ++it)
    {
        if ((*it)->slot_id != slot_id) continue;
        if (it != ds.graphs.begin()) ds.graphs.splice(ds.graphs.begin(), ds.graphs, it);
        return ds.graphs.front().get();
    }
    auto rit = ds.rings.find(slot_id);
    if (rit == ds.rings.end()) return nullptr;
    const dspark_ring & ring = *rit->second;

    const int dev = ds.dev;
    const int64_t B = ds.block, R = ds.rows, E = ds.n_embd, HD = ds.head_dim;
    const int64_t KV = (int64_t) ds.n_head_kv * HD;
    const int64_t V = m.hp.n_vocab;
    const size_t n_nodes = 1024 + 64 * (size_t) ds.n_layer + 24 * (size_t) B;

    std::unique_ptr<dspark_graph> g(new dspark_graph());
    g->slot_id = slot_id;
    ggml_init_params gp = { ggml_tensor_overhead() * n_nodes + ggml_graph_overhead_custom(n_nodes, false), nullptr, true };
    g->ctx = ggml_init(gp);
    if (!g->ctx) return nullptr;
    g->gf = ggml_new_graph_custom(g->ctx, n_nodes, false);
    g->sched = ggml_backend_sched_new(m.sched_backends, m.sched_bufts, m.n_sched_backends, n_nodes, false, false);
    if (!g->sched) return nullptr;
    ggml_context * ctx = g->ctx;
    ggml_backend_t be = m.sched_backends[dev];
    auto input = [&](ggml_type type, int64_t ne0, int64_t ne1, const char * name)
    {
        ggml_tensor * t = ne1 > 0 ? ggml_new_tensor_2d(ctx, type, ne0, ne1) : ggml_new_tensor_1d(ctx, type, ne0);
        ggml_set_input(t);
        ggml_set_name(t, name);
        ggml_backend_sched_set_tensor_backend(g->sched, t, be);
        return t;
    };
    g->emb = input(GGML_TYPE_F32, E, B, "ds_emb");
    g->pos = input(GGML_TYPE_I32, B, 0, "ds_pos");
    g->ridx = input(GGML_TYPE_I32, B, 0, "ds_ridx");
    g->mask = input(GGML_TYPE_F16, R, B, "ds_mask");
    g->anchor = input(GGML_TYPE_I32, 1, 0, "ds_anchor");

    ggml_tensor * x = g->emb;
    for (int il = 0; il < ds.n_layer; il++)
    {
        const dspark_layer & L = ds.layers[(size_t) il];
        ggml_tensor * h = dspark_rms(ctx, ds, x, L.attn_norm);
        ggml_tensor * q = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, L.wq, h), HD, ds.n_head, B);
        ggml_tensor * k = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, L.wk, h), HD, ds.n_head_kv, B);
        ggml_tensor * v = ggml_mul_mat(ctx, L.wv, h);                                          // [KV, B]
        q = dspark_rope(ctx, ds, dspark_rms(ctx, ds, q, L.q_norm), g->pos);
        k = dspark_rope(ctx, ds, dspark_rms(ctx, ds, k, L.k_norm), g->pos);
        // The block's own K/V go into the ring at its positions, and attention
        // reads the ring through the writes: [context in window | the block].
        ggml_tensor * kw = ggml_set_rows(ctx, ring.k[(size_t) il], ggml_reshape_2d(ctx, k, KV, B), g->ridx);
        ggml_tensor * vw = ggml_set_rows(ctx, ring.v[(size_t) il], v, g->ridx);
        ggml_tensor * kc = ggml_view_3d(ctx, kw, HD, ds.n_head_kv, R, ggml_row_size(kw->type, HD), kw->nb[1], 0);
        ggml_tensor * vc = ggml_view_3d(ctx, vw, HD, ds.n_head_kv, R, ggml_row_size(vw->type, HD), vw->nb[1], 0);
        ggml_tensor * qp = ggml_permute(ctx, q, 0, 2, 1, 3);                                   // [HD, B, n_head]
        ggml_tensor * kp = ggml_permute(ctx, kc, 0, 2, 1, 3);                                  // [HD, R, n_head_kv]
        ggml_tensor * vp = ggml_permute(ctx, vc, 0, 2, 1, 3);
        ggml_tensor * fa = ggml_flash_attn_ext(ctx, qp, kp, vp, g->mask, 1.0f / sqrtf((float) HD), 0.0f, 0.0f);
        ggml_prec_set_acc(fa, GGML_PREC_F32);                                                   // [HD, n_head, B]
        ggml_tensor * o = ggml_mul_mat(ctx, L.wo, ggml_reshape_2d(ctx, fa, HD * ds.n_head, B));
        x = ggml_add(ctx, x, o);
        ggml_tensor * y = dspark_rms(ctx, ds, x, L.ffn_norm);
        ggml_tensor * up = ggml_mul_mat(ctx, L.ffn_up, y);
        ggml_tensor * gate = ggml_mul_mat(ctx, L.ffn_gate, y);
        ggml_tensor * f = ggml_mul_mat(ctx, L.ffn_down, ggml_swiglu_oai(ctx, gate, up, SWIGLU_ALPHA, SWIGLU_LIMIT));
        x = ggml_add(ctx, x, f);
    }
    ggml_tensor * hn = dspark_rms(ctx, ds, x, ds.out_norm);                                    // [E, B]
    ggml_tensor * base = ggml_mul_mat(ctx, m.output, hn);                                     // [V, B]
    ggml_prec_set_acc(base, GGML_PREC_F32);

    ggml_tensor * prev = g->anchor;
    ggml_tensor * toks = nullptr;
    ggml_tensor * conf = nullptr;
    for (int64_t i = 0; i < B; i++)
    {
        ggml_tensor * e = ggml_get_rows(ctx, ds.markov_w1, prev);                              // [rank, 1]
        ggml_tensor * bias = ggml_mul_mat(ctx, ds.markov_w2, e);                               // [V, 1]
        ggml_prec_set_acc(bias, GGML_PREC_F32);
        ggml_tensor * col = ggml_add(ctx, ggml_view_2d(ctx, base, V, 1, base->nb[1], (size_t) i * base->nb[1]), bias);
        ggml_tensor * tok = ggml_argmax(ctx, col);                                             // I32 [1]
        // token ids are below 2^24: exact in F32, which every backend concatenates
        ggml_tensor * tf = ggml_cast(ctx, tok, GGML_TYPE_F32);
        toks = toks ? ggml_concat(ctx, toks, tf, 0) : tf;
        if (ds.conf_w)
        {
            ggml_tensor * hi = ggml_cont(ctx, ggml_view_2d(ctx, hn, E, 1, hn->nb[1], (size_t) i * hn->nb[1]));
            ggml_tensor * c = ggml_mul_mat(ctx, ds.conf_w, ggml_concat(ctx, hi, e, 0));         // [1, 1]
            if (ds.conf_b) c = ggml_add(ctx, c, ds.conf_b);
            c = ggml_sigmoid(ctx, c);
            conf = conf ? ggml_concat(ctx, conf, c, 0) : c;
        }
        prev = tok;
    }
    ggml_set_output(toks);
    ggml_set_name(toks, "ds_toks");
    ggml_build_forward_expand(g->gf, toks);
    g->toks = toks;
    if (conf)
    {
        conf = ggml_reshape_1d(ctx, conf, B);
        ggml_set_output(conf);
        ggml_set_name(conf, "ds_conf");
        ggml_build_forward_expand(g->gf, conf);
        g->conf = conf;
    }
    if (!ggml_backend_sched_alloc_graph(g->sched, g->gf))
    {
        fprintf(stderr, "[minimax-m3] DSpark draft graph allocation failed on GPU %d\n", dev);
        return nullptr;
    }
    ds.n_graph_builds++;
    if (debug_on())
        fprintf(stderr, "[minimax-m3] built DSpark draft graph for slot %d: nodes=%d splits=%d compute MiB=%.1f\n", slot_id,
                ggml_graph_n_nodes(g->gf), ggml_backend_sched_get_n_splits(g->sched),
                ggml_backend_sched_get_buffer_size(g->sched, be) / 1048576.0);
    ds.graphs.push_front(std::move(g));
    while ((int) ds.graphs.size() > ds.graph_cap) ds.graphs.pop_back();
    return ds.graphs.front().get();
}

/// One draft for the active slot: the anchor at position n_past (sampled, not
/// yet in the trunk) and block-1 mask tokens after it. Returns the rows
/// drafted (the block size), 0 when this slot cannot draft now.
static int dspark_draft(model & m, int32_t anchor, int32_t * toks_out, float * conf_out)
{
    dspark_state & ds = m.ds;
    if (!ds.loaded || !m.active_slot || anchor < 0 || anchor >= m.hp.n_vocab) return 0;
    slot & s = *m.active_slot;
    auto rit = ds.rings.find(s.id);
    if (rit == ds.rings.end() || s.n_past <= 0) return 0;
    dspark_ring & ring = *rit->second;
    const auto t0 = std::chrono::steady_clock::now();
    dspark_graph * g = dspark_acquire_graph(m, s.id);
    if (!g) return 0;

    const int64_t n = s.n_past, B = ds.block, R = ds.rows, E = ds.n_embd, W = ds.window;
    std::vector<float> emb((size_t) (E * B));
    dspark_embed_row(m, anchor, emb.data());
    for (int64_t i = 1; i < B; i++) std::copy(ds.mask_row.begin(), ds.mask_row.end(), emb.begin() + (size_t) (i * E));
    std::vector<int32_t> pos((size_t) B), ridx((size_t) B);
    for (int64_t i = 0; i < B; i++)
    {
        pos[(size_t) i] = (int32_t) (n + i);
        ridx[(size_t) i] = (int32_t) ((n + i) & (R - 1));
        ring.pos[(size_t) ridx[(size_t) i]] = n + i;   // the block's rows: never context (positions >= n_past)
    }
    // Row r holds position t = ring.pos[r]: context when t < n and the query is
    // less than a window past it, the block's own row when t is in [n, n + B).
    const ggml_fp16_t NEG = ggml_fp32_to_fp16(-INFINITY), ZERO = ggml_fp32_to_fp16(0.0f);
    std::vector<ggml_fp16_t> mask((size_t) (R * B), NEG);
    for (int64_t r = 0; r < R; r++)
    {
        const int64_t t = ring.pos[(size_t) r];
        if (t < 0) continue;
        for (int64_t i = 0; i < B; i++)
        {
            const int64_t q = n + i;
            const bool vis = t < n ? q - t < W : (t < n + B && (!ds.causal || t <= q));
            if (vis) mask[(size_t) (i * R + r)] = ZERO;
        }
    }
    ggml_backend_tensor_set(g->emb, emb.data(), 0, emb.size() * sizeof(float));
    ggml_backend_tensor_set(g->pos, pos.data(), 0, pos.size() * sizeof(int32_t));
    ggml_backend_tensor_set(g->ridx, ridx.data(), 0, ridx.size() * sizeof(int32_t));
    ggml_backend_tensor_set(g->mask, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(g->anchor, &anchor, 0, sizeof(int32_t));
    if (ggml_backend_sched_graph_compute(g->sched, g->gf) != GGML_STATUS_SUCCESS)
    {
        fprintf(stderr, "[minimax-m3] DSpark draft compute failed\n");
        return 0;
    }
    std::vector<float> tf((size_t) B), cf((size_t) B, 1.0f);
    ggml_backend_tensor_get(g->toks, tf.data(), 0, tf.size() * sizeof(float));
    if (g->conf) ggml_backend_tensor_get(g->conf, cf.data(), 0, cf.size() * sizeof(float));
    for (int64_t i = 0; i < B; i++)
    {
        toks_out[i] = (int32_t) lrintf(tf[(size_t) i]);
        conf_out[i] = cf[(size_t) i];
    }
    ds.n_drafts++;
    ds.draft_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return (int) B;
}

} // namespace tsg_mm3

// ---------------------------------------------------------------------------
// C ABI (DSpark)
// ---------------------------------------------------------------------------
/// TSGgml_MiniMaxM3LoadModelTp with a DSpark drafter GGUF (null or empty: none).
/// The drafter is placed beside the output head; a drafter that does not match
/// the target is a load refusal.
TSG_EXPORT void * TSGgml_MiniMaxM3LoadModelDraft(const char * gguf_path, int n_gpu, int n_ctx, int n_ubatch, int n_threads,
                                                 int n_cpu_moe, const char * backend_name, int ctx_is_hard_limit,
                                                 const char * layer_split, int tp, const char * draft_path)
{
    try
    {
        return tsg_mm3::load(gguf_path, n_gpu, n_ctx, n_ubatch, n_threads, n_cpu_moe, backend_name, ctx_is_hard_limit != 0,
                             layer_split, tp, draft_path);
    }
    catch (const std::exception & e)
    {
        tsg::report_load_refusal("[minimax-m3] load failed: %s\n", e.what());
        return nullptr;
    }
}

/// Rows one draft proposes (the drafter's block), or 0 without a drafter.
TSG_EXPORT int TSGgml_MiniMaxM3DsparkBlockSize(void * handle)
{
    auto * m = (tsg_mm3::model *) handle;
    return m && m->ds.loaded ? m->ds.block : 0;
}

/// Draft for the active slot from `anchor` (the sampled token at position
/// n_past, not yet forwarded). Writes block-size tokens and confidences and
/// returns their count; 0 when the slot cannot draft (no drafter, no context
/// ring, an empty sequence, a failed graph).
TSG_EXPORT int TSGgml_MiniMaxM3DsparkDraft(void * handle, int anchor, int32_t * toks_out, float * conf_out)
{
    auto * m = (tsg_mm3::model *) handle;
    if (!m || !toks_out || !conf_out) return 0;
    std::lock_guard<std::mutex> lk(m->mu);
    try { return tsg_mm3::dspark_draft(*m, anchor, toks_out, conf_out); }
    catch (const std::exception & e) { fprintf(stderr, "[minimax-m3] DSpark draft failed: %s\n", e.what()); return 0; }
}

/// [0] block size (0: no drafter), [1] context ring rows per sequence, [2]
/// drafts run, [3] their total wall time in microseconds, [4] sequences with a
/// context ring, [5] the drafter's device, [6] draft graphs built, [7] draft
/// layers, [8] tapped target layers.
TSG_EXPORT int TSGgml_MiniMaxM3DsparkStats(void * handle, int64_t * out, int cap)
{
    auto * m = (tsg_mm3::model *) handle;
    if (!m || !out || cap < 9) return 0;
    std::lock_guard<std::mutex> lk(m->mu);
    const tsg_mm3::dspark_state & ds = m->ds;
    out[0] = ds.loaded ? ds.block : 0;
    out[1] = ds.rows;
    out[2] = (int64_t) ds.n_drafts;
    out[3] = (int64_t) (ds.draft_ms * 1000.0);
    out[4] = (int64_t) ds.rings.size();
    out[5] = ds.dev;
    out[6] = (int64_t) ds.n_graph_builds;
    out[7] = ds.n_layer;
    out[8] = (int64_t) ds.target_layers.size();
    return 9;
}

#if defined(TSG_GGML_TEST_HOOKS)
/// Test hook: draft layer `layer`'s K and V rows (F32, n_head_kv * head_dim each)
/// of position `position` in slot `slot_id`'s context ring. 1 when the ring
/// holds that position, 0 otherwise.
TSG_EXPORT int TSGgml_MiniMaxM3TestDsparkContext(void * handle, int slot_id, int layer, int64_t position,
                                                 float * k_out, float * v_out)
{
    auto * m = (tsg_mm3::model *) handle;
    if (!m || !k_out || !v_out) return 0;
    std::lock_guard<std::mutex> lk(m->mu);
    const tsg_mm3::dspark_state & ds = m->ds;
    auto it = ds.rings.find(slot_id);
    if (!ds.loaded || it == ds.rings.end() || layer < 0 || layer >= ds.n_layer || position < 0) return 0;
    const tsg_mm3::dspark_ring & r = *it->second;
    const int64_t row = position & (ds.rows - 1);
    if (r.pos[(size_t) row] != position) return 0;
    const int64_t KV = (int64_t) ds.n_head_kv * ds.head_dim;
    std::vector<ggml_fp16_t> h((size_t) KV);
    ggml_backend_tensor_get(r.k[(size_t) layer], h.data(), (size_t) row * r.k[(size_t) layer]->nb[1], h.size() * sizeof(ggml_fp16_t));
    ggml_fp16_to_fp32_row(h.data(), k_out, KV);
    ggml_backend_tensor_get(r.v[(size_t) layer], h.data(), (size_t) row * r.v[(size_t) layer]->nb[1], h.size() * sizeof(ggml_fp16_t));
    ggml_fp16_to_fp32_row(h.data(), v_out, KV);
    return 1;
}
#endif
