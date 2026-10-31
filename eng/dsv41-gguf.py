"""Read V4.1 checkpoint metadata directly from the publisher's GGUF files.

This module never normalizes tokens or reconstructs Engram hash parameters.
"""
import struct

import numpy as np


def tokenizer_identity(reader):
    """Return vocabulary size and the raw-token identity used by vision files."""
    from gguf import GGUFValueType
    field = reader.fields["tokenizer.ggml.tokens"]
    if field.types != [GGUFValueType.ARRAY, GGUFValueType.STRING]:
        raise ValueError("GGUF tokenizer tokens must be a string array")
    tokens = field.contents()
    if not 0 < len(tokens) <= 1048576:
        raise ValueError("GGUF tokenizer vocabulary is outside supported bounds")
    fingerprint = 14695981039346656037
    for token in tokens:
        encoded = token.encode("utf-8")
        for byte in struct.pack("<Q", len(encoded)) + encoded:
            fingerprint = ((fingerprint ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return len(tokens), fingerprint


def load_engram(weights):
    """Read and validate embedded hash arrays; use tensor presence for cache sharing."""
    fields = weights.readers[0].fields

    def get(name, array=False, prefix="deepseek41.engram."):
        from gguf import GGUFValueType
        key = prefix + name
        if key not in fields:
            raise ValueError(f"Missing embedded GGUF metadata: {key}")
        integer_types = (GGUFValueType.UINT32, GGUFValueType.INT32, GGUFValueType.UINT64, GGUFValueType.INT64)
        types = fields[key].types
        if (array and (len(types) != 2 or types[0] != GGUFValueType.ARRAY or types[1] not in integer_types)
                or not array and (len(types) != 1 or types[0] not in integer_types)):
            raise ValueError(f"Invalid embedded GGUF metadata type: {key}")
        return fields[key].contents()

    mapping = np.asarray(get("token_map", array=True), dtype=np.int64)
    pad, ngram, heads, dim = (int(get(name)) for name in ("pad_id", "max_ngram_size", "head_count", "key_length"))
    layer_ids = np.asarray(get("layer_ids", array=True), dtype=np.int64)
    vocab, fingerprint = tokenizer_identity(weights.readers[0])
    if mapping.shape != (vocab,) or vocab == 0 or mapping.min() < 0 or mapping.max() >= vocab:
        raise ValueError("Invalid embedded Engram token map or padding ID")
    compressed = int(mapping.max()) + 1
    if not 0 <= pad < compressed:
        raise ValueError("Embedded Engram pad_id must already be a compressed token ID")
    if not np.array_equal(np.unique(mapping), np.arange(compressed)):
        raise ValueError("Embedded Engram compressed token IDs must be contiguous")
    layers = int(get("block_count", prefix="deepseek41."))
    if not 1 <= layers <= 128 or not 2 <= ngram <= 16 or not 1 <= heads <= 128 or not 1 <= dim <= 65536 or layer_ids.ndim != 1 or not 1 <= len(layer_ids) <= 128 or np.any(layer_ids < 0) or np.any(layer_ids >= layers) or np.any(np.diff(layer_ids) <= 0):
        raise ValueError("Invalid embedded Engram layer geometry")
    embedding = weights.shape("token_embd.weight") if "token_embd.weight" in weights.tensors else ()
    if len(embedding) < 2 or embedding[1] != vocab:
        raise ValueError("GGUF token embedding vocabulary does not match tokenizer")
    expected_tables = {f"blk.{layer}.engram_embd.weight" for layer in layer_ids}
    actual_tables = {name for name in weights.tensors if name.startswith("blk.") and name.endswith(".engram_embd.weight")}
    if expected_tables != actual_tables:
        raise ValueError("Embedded Engram layer IDs do not match GGUF table tensors")
    columns = (ngram - 1) * heads
    arrays = {}
    for name, width in (("multipliers", ngram), ("primes", columns), ("offsets", columns)):
        values = np.asarray(get(name, array=True), dtype=np.int64)
        if values.shape != (len(layer_ids) * width,):
            raise ValueError(f"Invalid embedded Engram {name} length")
        arrays[name] = values.reshape(len(layer_ids), width)
    layouts = []
    for index, layer in enumerate(layer_ids):
        multipliers, primes, offsets = (arrays[name][index] for name in ("multipliers", "primes", "offsets"))
        if np.any(multipliers <= 0) or np.any(multipliers % 2 != 1) or np.any(multipliers > (2**63 - 1) // compressed):
            raise ValueError("Invalid embedded Engram hash multipliers")
        expected_offsets = np.cumsum(np.concatenate(([0], primes[:-1])))
        if np.any(primes < 2) or np.any(primes > 2**32 - 1) or not np.array_equal(offsets, expected_offsets):
            raise ValueError("Invalid embedded Engram bucket offsets")
        rows = int(primes.sum())
        name = f"blk.{layer}.engram_embd.weight"
        shape = weights.shape(name) if name in weights.tensors else ()
        if rows > 2**31 - 1 or len(shape) < 2 or shape[:2] != (dim, rows) or any(size != 1 for size in shape[2:]):
            raise ValueError(f"Embedded Engram layout differs from tensor {name}")
        layouts.append(dict(id=int(layer), rows=rows, multipliers=multipliers,
                            primes=primes.reshape(ngram - 1, heads), offsets=offsets))
    ratios = get("attention.compress_ratios", array=True, prefix="deepseek41.")
    if not layers <= len(ratios) <= 256 or any(value < 0 for value in ratios):
        raise ValueError("Invalid GGUF compression ratios")
    ratios = ratios[:layers]
    kv_sources = [layer for layer in range(layers) if f"blk.{layer}.attn_compressor_kv.weight" in weights.tensors]
    index_sources = [layer for layer in range(layers) if f"blk.{layer}.indexer.attn_q_b.weight" in weights.tensors]
    if len(ratios) != layers or not kv_sources or not index_sources:
        raise ValueError("Invalid GGUF compression ratios or shared-cache source layers")
    candidate = next((layer for layer in index_sources if ratios[layer] == 1), -1)

    def candidate_value(name, default):
        from gguf import GGUFValueType
        key = "deepseek41.attention.indexer." + name
        if key not in fields:
            return default
        if fields[key].types not in ([GGUFValueType.UINT32], [GGUFValueType.INT32], [GGUFValueType.UINT64], [GGUFValueType.INT64]):
            raise ValueError(f"Invalid integer GGUF metadata: {key}")
        return int(fields[key].contents())

    source = candidate_value("candidate_source_layer", candidate)
    # Unsloth's 1017 GGUFs spell the budget candidate_top_k_blocks; TensorSharp's own files
    # candidate_top_k. A file with both must not let them disagree (as the native loader).
    topk = candidate_value("candidate_top_k_blocks", candidate_value("candidate_top_k", 2048)) if source >= 0 else 0
    if source >= 0 and candidate_value("candidate_top_k", topk) != topk:
        raise ValueError("deepseek41.attention.indexer.candidate_top_k_blocks disagrees with candidate_top_k")
    block = candidate_value("candidate_block_size", 8) if source >= 0 else 0
    if source < -1 or source >= layers or source >= 0 and (source not in index_sources or not 1 <= topk <= 2**31 - 1 or not 1 <= block <= 2**31 - 1):
        raise ValueError("Invalid GGUF candidate indexer geometry")
    return dict(token_map=mapping, pad=pad, max_ngram=ngram, heads=heads, dim=dim,
                layouts=layouts, kv_sources=kv_sources, index_sources=index_sources,
                candidate=source, candidate_topk=topk, candidate_block=block,
                fingerprint=fingerprint, compressed=compressed)


def load_config(weights):
    """Translate GGUF graph metadata into the independent reference's field names."""
    fields = weights.readers[0].fields
    names = {
        "num_hidden_layers": "block_count", "max_position_embeddings": "context_length",
        "hidden_size": "embedding_length", "num_attention_heads": "attention.head_count",
        "num_key_value_heads": "attention.head_count_kv", "head_dim": "attention.key_length",
        "qk_rope_head_dim": "rope.dimension_count", "q_lora_rank": "attention.q_lora_rank",
        "sliding_window": "attention.sliding_window", "o_lora_rank": "attention.output_lora_rank",
        "o_groups": "attention.output_group_count", "rms_norm_eps": "attention.layer_norm_rms_epsilon",
        "n_routed_experts": "expert_count", "n_shared_experts": "expert_shared_count",
        "num_experts_per_tok": "expert_used_count", "moe_intermediate_size": "expert_feed_forward_length",
        "norm_topk_prob": "expert_weights_norm", "routed_scaling_factor": "expert_weights_scale",
        "rope_theta": "rope.freq_base", "compress_rope_theta": "attention.compress_rope_freq_base",
        "compress_ratios": "attention.compress_ratios", "index_n_heads": "attention.indexer.head_count",
        "index_head_dim": "attention.indexer.key_length", "index_topk": "attention.indexer.top_k",
        "hc_mult": "hyper_connection.count", "hc_sinkhorn_iters": "hyper_connection.sinkhorn_iterations",
        "hc_eps": "hyper_connection.epsilon",
    }
    config = {name: fields["deepseek41." + key].contents() for name, key in names.items()}
    config["vocab_size"] = len(fields["tokenizer.ggml.tokens"].contents())
    config["compress_ratios"] = config["compress_ratios"][:config["num_hidden_layers"]]
    config["rope_scaling"] = {name: fields["deepseek41.rope.scaling." + key].contents()
                              for name, key in (("rope_type", "type"), ("factor", "factor"),
                                  ("beta_fast", "yarn_beta_fast"), ("beta_slow", "yarn_beta_slow"),
                                  ("original_max_position_embeddings", "original_context_length"))}
    for source, target in (("swiglu_clamp_exp", "swiglu_clamp_exp"), ("swiglu_clamp_shexp", "swiglu_clamp_shexp")):
        config[target] = fields["deepseek41." + source].contents()[:config["num_hidden_layers"]]
    return {"model_type": "deepseek_v41", "text_config": config}


def main():
    """Validate all shard metadata without reading or dequantizing tensor payloads."""
    import argparse
    import hashlib
    import json
    from pathlib import Path
    import re
    from types import SimpleNamespace
    from gguf import GGUFReader

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("model", type=Path, help="Model GGUF or the first split shard")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    match = re.fullmatch(r"(.*)-\d{5}-of-(\d{5})\.gguf", args.model.name)
    paths = [args.model] if not match else [args.model.with_name(f"{match[1]}-{i:05}-of-{int(match[2]):05}.gguf")
                                          for i in range(1, int(match[2]) + 1)]
    readers = [GGUFReader(str(path), mode="r") for path in paths]
    if ("general.architecture" not in readers[0].fields or
            any("general.architecture" in reader.fields and reader.fields["general.architecture"].contents() != "deepseek41" for reader in readers)):
        raise ValueError("Expected DeepSeek V4.1 GGUF shards")
    tensors = {tensor.name: tensor for reader in readers for tensor in reader.tensors}
    if len(tensors) != sum(len(reader.tensors) for reader in readers):
        raise ValueError("Duplicate GGUF tensor names across shards")
    weights = SimpleNamespace(readers=readers, tensors=tensors,
                              shape=lambda name: tuple(int(x) for x in tensors[name].shape))
    engram, config = load_engram(weights), load_config(weights)["text_config"]
    arrays = {key: dict(count=int(np.asarray(readers[0].fields["deepseek41.engram." + key].contents()).size),
                       int64_le_sha256=hashlib.sha256(np.asarray(readers[0].fields["deepseek41.engram." + key].contents(), dtype="<i8").tobytes()).hexdigest())
              for key in ("token_map", "multipliers", "primes", "offsets")}
    report = dict(embedded_engram_validated=True, tensor_payloads_verified=False,
                  files=[dict(path=str(path.resolve()), bytes=path.stat().st_size) for path in paths],
                  tensor_count=len(tensors), block_count=config["num_hidden_layers"],
                  vocabulary=config["vocab_size"], compressed_vocabulary=engram["compressed"],
                  compressed_pad_id=engram["pad"], tokenizer_fingerprint=engram["fingerprint"],
                  arrays=arrays, kv_sources=engram["kv_sources"], index_sources=engram["index_sources"],
                  candidate_source=engram["candidate"], candidate_topk=engram["candidate_topk"],
                  candidate_block_size=engram["candidate_block"],
                  layers=[dict(id=layer["id"], rows=layer["rows"]) for layer in engram["layouts"]])
    output = json.dumps(report, indent=2) + "\n"
    if args.report:
        args.report.write_text(output)
    print(output, end="")


if __name__ == "__main__":
    main()
