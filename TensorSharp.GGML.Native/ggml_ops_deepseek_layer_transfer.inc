// Included after graph_builder. Aggregate a stage's live state into one payload
// and transfer losslessly. Only newly committed compressed-cache rows cross the
// network; a context-length cache is never resent for each decoded token.
static void dsv4_pipeline_boundary(dsv4_model & m, graph_build_result & res, bool send)
{
    const int boundary = send ? m.layer_end : m.layer_begin;
    const int peer = m.pipeline.node + (send ? 1 : -1);
    struct range { ggml_tensor * tensor; size_t offset, bytes; bool cache; };
    std::vector<range> ranges;
    uint64_t layout = 1469598103934665603ull;
    auto mix = [&](uint64_t n) { layout = (layout ^ n) * 1099511628211ull; };
    auto append = [&](ggml_tensor * tensor, size_t offset, size_t bytes, bool cache) {
        if (!tensor || !tensor->buffer || offset > ggml_nbytes(tensor) || bytes > ggml_nbytes(tensor) - offset)
            throw std::runtime_error("Invalid distributed layer boundary tensor");
        mix(tensor->type); mix(offset); mix(bytes);
        for (int d = 0; d < 4; ++d) mix(tensor->ne[d]);
        ranges.push_back({tensor, offset, bytes, cache});
    };
    for (auto * tensor : send ? res.pipeline_out : res.pipeline_in)
        append(tensor, 0, ggml_nbytes(tensor), false);
    if (m.hp.v41)
    {
        for (int source : m.hp.kv_sources)
        {
            if (!m.cache_crosses(source, boundary)) continue;
            const auto & plan = m.hp.compress_ratios[source] == 1 ? res.plan_hca : res.plan_csa;
            auto & cache = m.active_slot->layers[source];
            mix(source);
            // Coalesce adjacent committed rows into bulk transfers.
            for (size_t i = 0; i < plan.state_write_idxs.size();)
            {
                size_t j = i + 1;
                while (j < plan.state_write_idxs.size() && plan.state_write_idxs[j] == plan.state_write_idxs[j-1] + 1) ++j;
                for (auto * tensor : {cache.csa_k, cache.lid_k})
                    append(tensor, (size_t) plan.state_write_idxs[i] * tensor->nb[1], (j-i) * tensor->nb[1], true);
                i = j;
            }
        }
    }
    size_t bytes = 0;
    for (const auto & r : ranges) {
        if (SIZE_MAX - bytes < r.bytes) throw std::runtime_error("Pipeline boundary size overflow");
        bytes += r.bytes;
    }
    uint64_t expected[] = {0x54534c4159455231ull, (uint64_t) boundary, (uint64_t) res.p0,
        (uint64_t) res.nt, (uint64_t) ranges.size(), (uint64_t) bytes, layout};
    uint64_t header[7];
    memcpy(header, expected, sizeof(header));
    m.pipeline.move(peer, send, header, sizeof(header));
    if (memcmp(header, expected, sizeof(header)) != 0)
        throw std::runtime_error("Distributed layer boundary metadata differs between nodes");
    m.pipeline.staging.resize(bytes);
    size_t offset = 0;
    if (send)
        for (const auto & r : ranges) {
            ggml_backend_tensor_get(r.tensor, m.pipeline.staging.data() + offset, r.offset, r.bytes);
            offset += r.bytes;
        }
    m.pipeline.move(peer, send, m.pipeline.staging.data(), bytes);
    if (!send)
    {
        for (const auto & r : ranges)
        {
            ggml_backend_tensor_set(r.tensor, m.pipeline.staging.data() + offset, r.offset, r.bytes);
            if (r.cache)
            {
                // Every local GPU reading this remote source has the same
                // committed rows, just like local source mirror writes.
                for (auto & cache : m.active_slot->layers)
                {
                    ggml_tensor ** mirrors = r.tensor == cache.csa_k ? cache.csa_k_mirror
                        : r.tensor == cache.lid_k ? cache.lid_k_mirror : nullptr;
                    if (!mirrors) continue;
                    for (int d = 0; d < m.n_gpu; ++d)
                        if (mirrors[d]) ggml_backend_tensor_set(mirrors[d],
                            m.pipeline.staging.data() + offset, r.offset, r.bytes);
                }
            }
            offset += r.bytes;
        }
    }
}
