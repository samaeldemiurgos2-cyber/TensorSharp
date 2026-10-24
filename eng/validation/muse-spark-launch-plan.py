#!/usr/bin/env python3
"""Inspect GGUF headers and plan matching whole-layer placement without loading weights.

Requires llama.cpp's gguf Python module (PYTHONPATH=/path/llama.cpp/gguf-py).
This reads tensor descriptors only; it does not run inference or scan weights.
The algorithm matches MuseSparkModel.PlanLayerPlacement. llama.cpp counts the
output projection as an additional layer for --tensor-split, hence last+1.
"""
import argparse
import json
from pathlib import Path
import re


def partition(layers, embedding, head, devices, streamed=None):
    if not 1 <= devices <= len(layers) or min([embedding, head, *layers]) < 0:
        raise ValueError('Need nonnegative bytes and at least one layer per device')
    prefix = [0]
    for value in layers:
        prefix.append(prefix[-1] + value)
    size = len(layers)
    streamed = [0] * size if streamed is None else streamed
    if len(streamed) != size or min(streamed) < 0:
        raise ValueError('Stream reservations must match layers and be nonnegative')
    best = [[float("inf")] * (size + 1) for _ in range(devices + 1)]
    split = [[0] * (size + 1) for _ in range(devices + 1)]
    best[0][0] = 0
    for d in range(1, devices + 1):
        for n in range(d, size + 1):
            for start in range(d - 1, n):
                own = prefix[n] - prefix[start] + (embedding if d == 1 else 0) + (head if d == devices else 0) + max(streamed[start:n])
                cost = max(best[d - 1][start], own)
                if cost < best[d][n]:
                    best[d][n], split[d][n] = cost, start
    end = size
    counts = [0] * devices
    for d in range(devices, 0, -1):
        start = split[d][end]
        counts[d - 1], end = end - start, start
    return counts


def discover_shards(directory, quantization, manifest=None):
    if Path(quantization).name != quantization or '/' in quantization or '\\' in quantization:
        raise ValueError('quantization must be a single directory name')
    paths = sorted((directory / quantization).glob('*.gguf'))
    if not paths:
        raise ValueError(f'No shards found for {quantization}')
    parsed = [re.fullmatch(r'(.+)-(\d{5})-of-(\d{5})\.gguf', p.name) for p in paths]
    if any(m is None for m in parsed):
        raise ValueError('Every shard must use the numbered split GGUF filename format')
    prefixes = {m[1] for m in parsed}
    totals = {int(m[3]) for m in parsed}
    if len(prefixes) != 1 or len(totals) != 1:
        raise ValueError('Mixed shard prefixes or shard counts')
    count = totals.pop()
    if count < 1 or [int(m[2]) for m in parsed] != list(range(1, count + 1)):
        raise ValueError(f'Expected all {count} numbered {quantization} shards')
    if manifest is not None:
        expected = {Path(e['rfilename']).name: e for e in manifest['siblings']
                    if e['rfilename'].startswith(quantization + '/') and e['rfilename'].endswith('.gguf')}
        if set(expected) != {p.name for p in paths}:
            raise ValueError('Local shard filenames differ from the selected manifest quantization')
        for path in paths:
            size = expected[path.name].get('size')
            if size is not None and path.stat().st_size != size:
                raise ValueError(f'Incomplete/wrong-size shard: {path.name}')
    return paths


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cpu-moe", type=int, default=42)
    p.add_argument("--devices", type=int, default=5)
    p.add_argument("--quantization", default="UD-Q4_K_XL")
    p.add_argument("--manifest", type=Path)
    p.add_argument("--kv-capacity-tokens", type=int, default=0,
                   help="Aggregate physical KV token capacity; zero produces a weights-only plan")
    p.add_argument("--kv-dtype", choices=("f16", "f32"), default="f16")
    p.add_argument("--rank0-reserve-gib", type=float, default=0,
                   help="Additional rank0 reserve, e.g. vision encoder and image scratch")
    p.add_argument("--runtime-reserve-gib", type=float, default=0,
                   help="Reserve on every GPU for runtime and graph scratch, in addition to explicit streamed expert stacks")
    p.add_argument("--gpu-memory-gib", type=float, default=0)
    p.add_argument("--host-memory-gib", type=float, default=0)
    p.add_argument("--host-reserve-gib", type=float, default=0,
                   help="Host runtime/OS/cache reserve beyond explicit expert weights and host KV")
    p.add_argument("--host-kv-capacity-tokens", type=int, default=0,
                   help="Aggregate host KV mirror/snapshot capacity, budgeted as float32")
    p.add_argument("--cpu-quota-cores", type=float)
    p.add_argument("--stream-experts", action='store_true',
                   help="Reserve peak complete offloaded gate/up/down stacks on each layer-owning GPU")
    args = p.parse_args()
    import gguf
    from inference_resource_limits import cgroup_snapshot
    cgroup = cgroup_snapshot()
    observed_host_limit = cgroup.get('memory_limit_bytes')
    host_limit_gib = args.host_memory_gib
    if observed_host_limit:
        host_limit_gib = min(host_limit_gib, observed_host_limit / 1024**3) if host_limit_gib else observed_host_limit / 1024**3
    try:
        manifest = json.loads(args.manifest.read_text()) if args.manifest else None
        paths = discover_shards(args.directory, args.quantization, manifest)
    except ValueError as error:
        p.error(str(error))
    if min(args.cpu_moe, args.kv_capacity_tokens, args.rank0_reserve_gib, args.runtime_reserve_gib,
           args.gpu_memory_gib, args.host_memory_gib, args.host_reserve_gib, args.host_kv_capacity_tokens) < 0:
        p.error('Memory and layer limits must be nonnegative')
    layers = None
    streamed = None
    kv_heads = head_dim = None
    embedding = head = offloaded = 0
    tensors = []
    shards = []
    for path in paths:
        reader = gguf.GGUFReader(path)
        if layers is None:
            arch = reader.fields['general.architecture'].contents()
            layers = [0] * int(reader.fields[arch + '.block_count'].contents())
            streamed = [0] * len(layers)
            kv_heads = int(reader.fields[arch + '.attention.head_count_kv'].contents())
            head_dim = int(reader.fields[arch + '.embedding_length'].contents()) // int(reader.fields[arch + '.attention.head_count'].contents())
            if not 1 <= args.devices <= len(layers) or args.cpu_moe > len(layers):
                p.error('Invalid device count or CPU expert layer count')
        if 'split.count' in reader.fields and int(reader.fields['split.count'].contents()) != len(paths):
            p.error('GGUF split.count does not match complete filename shard set')
        shards.append({"file": path.name, "bytes": path.stat().st_size, "tensors": len(reader.tensors)})
        for tensor in reader.tensors:
            name, size = tensor.name, int(tensor.n_bytes)
            tensors.append({"name": name, "bytes": size, "type": tensor.tensor_type.name})
            match = re.match(r"blk\.(\d+)\.", name)
            if match:
                layer = int(match[1])
                if layer < args.cpu_moe and re.search(r"\.ffn_(gate|up|down)_exps\.weight$", name):
                    offloaded += size
                    streamed[layer] += size
                else:
                    layers[layer] += size
            elif name.startswith("output"):
                head += size
            else:
                embedding += size
    kv_per_layer = args.kv_capacity_tokens * 2 * kv_heads * head_dim * (4 if args.kv_dtype == 'f32' else 2)
    rank0_reserve = int(args.rank0_reserve_gib * 1024**3)
    runtime_reserve = int(args.runtime_reserve_gib * 1024**3)
    stream_reserves = streamed if args.stream_experts else [0] * len(layers)
    counts = partition([v + kv_per_layer for v in layers], embedding + rank0_reserve, head, args.devices, stream_reserves)
    llama = counts.copy()
    llama[-1] += 1
    ranks = []
    cursor = 0
    for rank, count in enumerate(counts):
        weights = sum(layers[cursor:cursor + count]) + (embedding if rank == 0 else 0) + (head if rank == args.devices - 1 else 0)
        stream_bytes = max(stream_reserves[cursor:cursor + count])
        ranks.append({"rank": rank, "first_layer": cursor, "layers": count, "resident_weight_bytes": weights,
                      "resident_weight_gib": weights / 1024**3,
                      "kv_bytes": count * kv_per_layer,
                      "reserved_bytes": runtime_reserve + (rank0_reserve if rank == 0 else 0),
                      "streamed_expert_bytes": stream_bytes,
                      "planned_total_gib": (weights + count * kv_per_layer + runtime_reserve +
                                            stream_bytes + (rank0_reserve if rank == 0 else 0)) / 1024**3})
        cursor += count
    vision = {}
    mmproj = args.directory / "mmproj-Q8_0.gguf"
    if mmproj.exists():
        reader = gguf.GGUFReader(mmproj)
        for key in ("clip.projector_type", "clip.vision.projector_type", "clip.has_vision_encoder", "clip.has_audio_encoder",
                    "clip.vision.projection_dim", "clip.audio.projector_type"):
            if key in reader.fields:
                vision[key] = reader.fields[key].contents()
    host_kv = args.host_kv_capacity_tokens * len(layers) * 2 * kv_heads * head_dim * 4
    host_total = (offloaded + host_kv) / 1024**3 + args.host_reserve_gib
    exceeded = []
    if args.gpu_memory_gib:
        exceeded += [f"gpu{r['rank']} exceeds {args.gpu_memory_gib} GiB" for r in ranks if r['planned_total_gib'] > args.gpu_memory_gib]
    if host_limit_gib and host_total > host_limit_gib:
        exceeded.append(f'Host resident experts plus KV and reserve exceed {host_limit_gib} GiB')
    report = {"quantization": args.quantization, "shard_count": len(paths), "first_shard": str(paths[0]),
              "cpu_moe_layers": args.cpu_moe, "devices": args.devices, "shards": shards,
              "tensors": tensors, "layer_resident_bytes": layers, "embedding_bytes": embedding,
              "head_bytes": head, "offloaded_expert_bytes": offloaded,
              "offloaded_expert_layer_bytes": streamed, "stream_experts": args.stream_experts,
              "tensorsharp_layer_counts": counts, "llama_tensor_split": llama, "ranks": ranks,
              "projector": vision,
              "kv_capacity_tokens": args.kv_capacity_tokens, "kv_dtype": args.kv_dtype,
              "runtime_reserve_gib": args.runtime_reserve_gib, "rank0_reserve_gib": args.rank0_reserve_gib,
              "host_reserve_gib": args.host_reserve_gib, "host_planned_total_gib": host_total,
              "host_kv_capacity_tokens": args.host_kv_capacity_tokens, "host_kv_bytes": host_kv,
              "host_memory_limit_gib": host_limit_gib, "gpu_memory_limit_gib": args.gpu_memory_gib,
              "observed_cgroup": cgroup,
              "cpu_quota_cores": cgroup.get('cpu_quota_cores') or args.cpu_quota_cores,
              "budget_feasible": not exceeded if args.gpu_memory_gib and host_limit_gib else None,
              "budget_status": 'exceeded' if exceeded else ('within_declared_reserves' if args.gpu_memory_gib and host_limit_gib else 'unbounded'),
              "budget_errors": exceeded,
              "scope": "Descriptor-based estimate with explicit reserves, not measured peak memory or proof of residency. Includes full linear KV for all layers; no SWA ring assumption.",
              "llama_embedding_override": "token_embd\\.weight=CUDA0"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("tensors", "shards", "layer_resident_bytes")}, indent=2))
    if exceeded:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
