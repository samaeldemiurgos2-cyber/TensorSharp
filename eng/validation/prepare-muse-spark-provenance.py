#!/usr/bin/env python3
"""Generate planned launch provenance after every checkpoint hash is verified.

These files describe prepared commands. Attach observed startup logs and change
launch_observed to true only after that exact command successfully starts.
"""
import hashlib
import json
import subprocess
import argparse
import re
import shlex
from pathlib import Path
from muse_spark_artifacts import shard_entries
from inference_resource_limits import cgroup_snapshot, stable_limits

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--workspace', type=Path, default=Path('/workspace'))
parser.add_argument('--evidence', type=Path)
parser.add_argument('--source-evidence', type=Path,
                    help='Source-verification and binary snapshot directory; defaults to --evidence')
parser.add_argument('--verification', type=Path, help='Completed all-shard hash report')
parser.add_argument('--manifest', type=Path, help='Pinned Hugging Face metadata manifest')
parser.add_argument('--quantization', help='Optional check on the quantization inferred from verified shards')
parser.add_argument('--cpu-moe', type=int, default=42)
parser.add_argument('--cpu-moe-f16', type=int, help='Override CPU expert layer count for the F16 profile')
parser.add_argument('--cpu-moe-f32', type=int, help='Override CPU expert layer count for the F32 profile')
parser.add_argument('--profiles', nargs='+', choices=['f16', 'f32'], default=['f16', 'f32'])
parser.add_argument('--batch-f16', type=int, default=256)
parser.add_argument('--batch-f32', type=int, default=128)
parser.add_argument('--devices', type=int, default=5)
parser.add_argument('--visible-devices', help='Distinct physical CUDA device indices, in visible rank order')
parser.add_argument('--cpu-threads', type=int, default=32)
parser.add_argument('--cpu-affinity', help='Comma-separated logical CPU indices, applied equally to both launch recipes')
parser.add_argument('--context', type=int, default=16384)
parser.add_argument('--kv-pool', type=int, default=16384)
parser.add_argument('--concurrency', type=int, default=4)
parser.add_argument('--prefix-cache', action=argparse.BooleanOptionalAction, default=True)
parser.add_argument('--tensorsharp-parallelism', choices=['layer', 'tensor'], default='layer')
parser.add_argument('--distribution-in-engine-options', action='store_true',
                    help='Keep intentional engine distribution differences out of shared work constraints')
parser.add_argument('--tensorsharp-options', type=Path, help='Explicit engine_options JSON, including environment controls')
parser.add_argument('--llama-options', type=Path, help='Explicit reference engine_options JSON')
parser.add_argument('--tensorsharp-launcher', type=Path)
parser.add_argument('--llama-launcher', type=Path)
parser.add_argument('--layer-counts-f16', help='One positive decoder layer count per device, totaling 62')
parser.add_argument('--layer-counts-f32', help='One positive decoder layer count per device, totaling 62')
parser.add_argument('--reference-template', type=Path,
                    help='Matched llama.cpp template created by prepare_muse_spark_llama_reference.py')
parser.add_argument('--tokenizer-metadata', type=Path,
                    help='Actual GGUF tokenizer metadata/hash capture; defaults to evidence/tokenizer-metadata.json')
parser.add_argument('--projector-verification', type=Path,
                    help='Reuse prior complete projector hash verification for the same unchanged path; recheck manifest and size')
args = parser.parse_args()
if not 1 <= args.devices <= 16 or min(args.cpu_threads, args.context, args.kv_pool, args.concurrency) <= 0 or args.kv_pool % 16:
    parser.error('Positive dimensions, 1-16 devices and a KV pool divisible by 16 are required')
if args.tensorsharp_parallelism == 'tensor' and (args.devices < 2 or not args.distribution_in_engine_options):
    parser.error('Tensor parallelism requires multiple devices and --distribution-in-engine-options')
def indices(text, name, expected=None):
    try:
        values = [int(value) for value in text.split(',')]
    except ValueError:
        parser.error(name + ' must be comma-separated integer indices')
    if not values or min(values) < 0 or len(set(values)) != len(values) or (expected is not None and len(values) != expected):
        parser.error(name + ' must contain distinct nonnegative indices with the expected count')
    return values
visible_devices = indices(args.visible_devices, '--visible-devices', args.devices) if args.visible_devices else None
cpu_affinity = indices(args.cpu_affinity, '--cpu-affinity') if args.cpu_affinity else None
workspace = args.workspace
ROOT = args.evidence or workspace / 'evidence/muse-spark-reference'
ROOT.mkdir(parents=True, exist_ok=True)
SOURCE = args.source_evidence or ROOT
tokenizer_metadata_path = args.tokenizer_metadata or ROOT / 'tokenizer-metadata.json'
tokenizer_metadata = json.loads(tokenizer_metadata_path.read_text())
fixture_path = workspace / 'TensorSharp/InferenceWeb.Tests/Fixtures/MuseSpark/unsloth_chat_template.jinja'
fixture_file_hash = hashlib.sha256(fixture_path.read_bytes()).hexdigest()
normalized_template_hash = hashlib.sha256(fixture_path.read_text(encoding='utf-8').encode('utf-8')).hexdigest()
embedded_hash = tokenizer_metadata.get('chat_template_sha256', tokenizer_metadata.get('template_utf8_sha256'))
assert embedded_hash == normalized_template_hash, 'Embedded template differs from the validated reference source'
if 'file' not in tokenizer_metadata and 'first_shard' in tokenizer_metadata:
    tokenizer_metadata = dict(tokenizer_metadata, file=tokenizer_metadata['first_shard'])
verification_path = args.verification or ROOT / 'download-full-hashes.json'
verification = json.loads(verification_path.read_text())
assert verification['status'] == 'pass'
manifest = json.loads((args.manifest or workspace / 'model-manifest.json').read_text())
if 'files' in verification:
    assert verification['files'] and all(row['status'] == 'pass' and row['sha256'] == row['expected_sha256']
        and row['bytes'] == row['expected_bytes'] for row in verification['files']), 'Incomplete full-hash verification'
    all_weights = {row['file']: row['sha256'] for row in verification['files']}
    assert len(all_weights) == len(verification['files']), 'Duplicate verified paths'
    weights = {name: digest for name, digest in all_weights.items() if '-of-' in name}
else:
    assert verification['hash_all'] and verification['shards'], 'No full checkpoint hash verification'
    assert all(s['sha256_status'] == 'pass' for s in verification['shards'])
    weights = {s['file']: s['actual_sha256'] for s in verification['shards']}
    all_weights = dict(weights)
assert weights, 'No verified checkpoint shards'
quantization = verification.get('quantization') or sorted(weights)[0].split('/')[0]
assert not args.quantization or args.quantization == quantization, 'Quantization disagrees with verified weights'
assert manifest['sha'] == verification['revision'] and manifest['id'] == verification['repo']
entries = shard_entries(manifest, quantization)
assert sorted(weights) == [entry['rfilename'] for entry in entries], 'Hash report is not the complete shard set'
assert all(weights[entry['rfilename']] == entry['lfs']['sha256'] for entry in entries)
cpu_counts = {'f16': args.cpu_moe if args.cpu_moe_f16 is None else args.cpu_moe_f16,
              'f32': args.cpu_moe if args.cpu_moe_f32 is None else args.cpu_moe_f32}
assert all(0 <= count <= 62 for count in cpu_counts.values())
batch_sizes = {'f16': args.batch_f16, 'f32': args.batch_f32}
assert all(batch_sizes[profile] > 0 for profile in args.profiles)

def placement(text, default):
    if text is None:
        if quantization != 'UD-Q4_K_XL':
            parser.error('Non-Q4 provenance requires explicit --layer-counts-f16 and --layer-counts-f32')
        text = default
    counts = [int(value) for value in text.split(',')]
    if len(counts) != args.devices or sum(counts) != 62 or min(counts) < 1:
        parser.error('Layer counts must have one positive entry per device, totaling 62')
    cursor, ranges = 0, []
    for count in counts:
        ranges.append([cursor, cursor + count - 1])
        cursor += count
    return ranges

placements = {}
for dtype in args.profiles:
    placements[dtype] = placement(getattr(args, 'layer_counts_' + dtype),
                                 '45,4,4,5,4' if dtype == 'f16' else '44,5,4,5,4')
entry = next(e for e in manifest['siblings'] if e['rfilename'] == 'mmproj-Q8_0.gguf')
path = workspace / 'models/muse-spark/mmproj-Q8_0.gguf'
assert path.stat().st_size == entry['size']
if args.projector_verification:
    previous_projector = json.loads(args.projector_verification.read_text())
    assert previous_projector['status'] == 'pass'
    assert Path(previous_projector['file']).resolve() == path.resolve()
    assert previous_projector['bytes'] == path.stat().st_size
    assert previous_projector['expected_sha256'] == entry['lfs']['sha256']
    digest = previous_projector['actual_sha256']
elif path.name in all_weights:
    digest = all_weights[path.name]
else:
    with path.open('rb') as source:
        hasher = hashlib.sha256()
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            hasher.update(block)
        digest = hasher.hexdigest()
assert digest == entry['lfs']['sha256']
weights[path.name] = digest
(ROOT / 'projector-verification.json').write_text(json.dumps({
    'file': str(path), 'bytes': path.stat().st_size, 'expected_sha256': entry['lfs']['sha256'],
    'actual_sha256': digest, 'status': 'pass',
    'verification_method': ('Prior complete hash verification reused; current file size and pinned manifest checked.'
                            if args.projector_verification or path.name in all_weights else 'Fresh complete SHA256 verification.'),
    'prior_verification': (str(args.projector_verification) if args.projector_verification else
                           str(verification_path) if path.name in all_weights else None)}, indent=2) + '\n')

def run(*command):
    return subprocess.check_output(command, text=True).strip()

hardware = {
    'gpu_inventory': run('nvidia-smi', '--query-gpu=index,name,uuid,memory.total,driver_version', '--format=csv,noheader'),
    'cpu': run('lscpu'),
    'kernel': run('uname', '-a'),
    'cgroup': cgroup_snapshot(),
}
# Only stable CPU/GPU identity is used to compare runs; uptime/free memory vary.
stable_cpu = '\n'.join(line for line in hardware['cpu'].splitlines() if 'MHz' not in line and 'BogoMIPS' not in line and 'scaling' not in line)
hardware_id = hashlib.sha256((hardware['gpu_inventory'] + stable_cpu +
                             json.dumps(stable_limits(hardware['cgroup']), sort_keys=True)).encode()).hexdigest()
llama_source = json.loads((SOURCE / 'llama-source-verification.json').read_text())
assert llama_source['status'] == 'pass' and llama_source['original_clean']
tensorsharp_source = json.loads((SOURCE / 'tensorsharp-source-verification.json').read_text())
assert tensorsharp_source['status'] == 'pass', 'Remote TensorSharp source snapshot is not fully verified'
binary_manifest = SOURCE / 'baseline-binaries.sha256'
assert binary_manifest.is_file(), 'Capture compiled binary hashes before preparing provenance'
binaries = {}
if binary_manifest.exists():
    for line in binary_manifest.read_text().splitlines():
        checksum, filename = line.split(maxsplit=1)
        binaries[filename.strip()] = checksum
for dtype in args.profiles:
    batch = batch_sizes[dtype]
    settings = {
        'quantization': quantization, 'context_per_request': args.context, 'aggregate_kv_pool_tokens': args.kv_pool,
        'max_running_requests': args.concurrency,
        'max_batched_tokens': batch, 'prefill_chunk_tokens': batch,
        'key_cache_dtype': dtype, 'value_cache_dtype': dtype, 'cpu_threads': args.cpu_threads,
        'cpu_expert_layer_count': cpu_counts[dtype],
        'layer_ranges': placements[dtype],
        'embedding_device': 0, 'output_device': args.devices - 1, 'vision_device': 0,
        'prefix_cache': args.prefix_cache, 'speculation': 'disabled', 'max_output_tokens': 512,
    }
    if args.distribution_in_engine_options:
        for key in ('layer_ranges', 'embedding_device', 'output_device', 'vision_device'):
            settings.pop(key)
        settings['gpu_count'] = args.devices
    if visible_devices is not None:
        settings['visible_gpu_devices'] = visible_devices
    if cpu_affinity is not None:
        settings['cpu_affinity'] = cpu_affinity
    for engine in ['tensorsharp', 'llamacpp']:
        tree = workspace / ('TensorSharp' if engine == 'tensorsharp' else 'llama.cpp')
        upstream = tree / 'ExternalProjects/ggml' if engine == 'tensorsharp' else tree
        # llama.cpp vendors its ggml sources in its own revision.
        script = ((args.tensorsharp_launcher if engine == 'tensorsharp' else args.llama_launcher) or
                  workspace / ('start-tensorsharp-matched.sh' if engine == 'tensorsharp' else 'start-llama-spark.sh'))
        options_path = args.tensorsharp_options if engine == 'tensorsharp' else args.llama_options
        options = json.loads(options_path.read_text()) if options_path else {}
        assert isinstance(options, dict), 'Engine options must be an object'
        environment = options.get('environment', {})
        assert isinstance(environment, dict) and all(isinstance(key, str) and re.fullmatch(r'[A-Z_][A-Z0-9_]*', key)
            and isinstance(value, str) for key, value in environment.items()), 'Invalid engine environment'
        common_env = {'OMP_NUM_THREADS': str(args.cpu_threads), 'SPARK_CPU_THREADS': str(args.cpu_threads)}
        if visible_devices is not None:
            common_env['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, visible_devices))
        assert all(key not in environment or environment[key] == value for key, value in common_env.items()), \
            'Engine environment conflicts with common hardware/work settings'
        environment_recipe = ''.join(f' {key}={shlex.quote(value)}' for key, value in sorted(environment.items()))
        if args.distribution_in_engine_options:
            mode = args.tensorsharp_parallelism if engine == 'tensorsharp' else 'layer'
            distribution = {'mode': mode, 'degree': args.devices}
            if mode == 'layer':
                distribution['layer_ranges'] = placements[dtype]
            assert 'distribution' not in options or options['distribution'] == distribution
            options['distribution'] = distribution
        if not args.prefix_cache and engine == 'llamacpp':
            assert options.get('prompt_cache_ram_mib', 0) == 0 and options.get('cache_idle_slots', False) is False
            options.update(prompt_cache_ram_mib=0, cache_idle_slots=False)
        script_text = script.read_text()
        template_path = args.reference_template if engine == 'llamacpp' else None
        template_extra = ''
        if engine == 'llamacpp':
            script_tokens = shlex.split(script_text, comments=True)
            if '--chat-template-file' in script_tokens:
                configured_template = Path(script_tokens[script_tokens.index('--chat-template-file') + 1])
                if template_path and template_path != configured_template:
                    raise ValueError('Reference template does not match the embedded launcher setting')
                template_path = configured_template
            elif template_path:
                template_extra = f' --chat-template-file {shlex.quote(str(template_path))}'
        data = {
            'launch_observed': False,
            'status': 'planned command; attach actual startup and placement evidence after launch',
            'hardware_id': hardware_id, 'hardware': hardware, 'weights_sha256': weights,
            'weight_verification': str(verification_path),
            'quantization': quantization, 'shard_count': len(entries),
            'projector_verification': str(ROOT / 'projector-verification.json'),
            'model_revision': verification['revision'], 'settings': settings, 'engine_options': options,
            'tokenizer_metadata': tokenizer_metadata,
            'tokenizer_metadata_file': str(tokenizer_metadata_path),
            'upstream_ggml_revision': (run('git', '-C', str(upstream), 'rev-parse', 'HEAD') if engine == 'tensorsharp' else llama_source['revision']),
            'upstream_ggml_clean': (run('git', '-C', str(upstream), 'status', '--porcelain') == '' if engine == 'tensorsharp' else True),
            'server_revision': tensorsharp_source['revision'] if engine == 'tensorsharp' else llama_source['revision'],
            'server_dirty': not tensorsharp_source['original_clean'] if engine == 'tensorsharp' else False,
            'reference_source_verification': llama_source,
            'tensorsharp_source_verification': tensorsharp_source,
            'source_evidence_directory': str(SOURCE),
            'server_command': (f'SPARK_QUANTIZATION={shlex.quote(quantization)} '
                               f'SPARK_CPU_MOE={cpu_counts[dtype]} '
                               'SPARK_LAYER_SPLIT=' + ','.join(str(last - first + 1) for first, last in placements[dtype]) +
                               f' SPARK_KV_DTYPE={dtype} SPARK_BATCH={batch} SPARK_KV_POOL_TOKENS={args.kv_pool}'
                               f' SPARK_CONTEXT={args.context} SPARK_PARALLEL={args.concurrency} SPARK_CPU_THREADS={args.cpu_threads}'
                               f' SPARK_DEGREE={args.devices} SPARK_PARALLELISM={args.tensorsharp_parallelism if engine == "tensorsharp" else "layer"}' +
                               (f' CUDA_VISIBLE_DEVICES={",".join(map(str, visible_devices))}' if visible_devices is not None else '') +
                               (' TS_MUSE_SPARK_STREAM_EXPERTS=0' if engine == 'tensorsharp' and 'TS_MUSE_SPARK_STREAM_EXPERTS' not in environment else '') +
                               environment_recipe +
                               (f' taskset -c {",".join(map(str, cpu_affinity))}' if cpu_affinity is not None else '') +
                               f' bash {shlex.quote(str(script))}' +
                               (' --no-spec' if engine == 'tensorsharp' else '') + template_extra),
            'server_url': 'http://127.0.0.1:8083' if engine == 'tensorsharp' else 'http://127.0.0.1:8082',
            'server_script': script_text,
            'server_script_sha256': hashlib.sha256(script.read_bytes()).hexdigest(),
            'binary_hashes': {name: checksum for name, checksum in binaries.items()
                              if ('/TensorSharp/' in name) == (engine == 'tensorsharp')},
            'binary_hash_scope': 'Snapshot at capture time; recapture if rebuilt before measured launch.',
            'server_log': None,
            'physical_kv_note': (f'{args.kv_pool}-token aggregate scheduler pool ({args.kv_pool//16} blocks of 16); F16 per-sequence restore; F32 packed paged pool'
                                if engine == 'tensorsharp' else f'Unified aggregate pool: {args.kv_pool} tokens; {args.context} per-request ceiling; up to {args.concurrency} active slots'),
            'scope': 'Common hardware and work constraints; engine_options records intentional distribution differences when requested. Concurrent requests share the aggregate KV capacity; physical allocation differs by engine.',
        }
        if not args.prefix_cache:
            data['server_command'] += ' --no-prefix-cache' if engine == 'tensorsharp' else ' --no-cache-prompt'
        if engine == 'tensorsharp':
            data['request_prompt_adapter'] = {
                'skills': [], 'skills_discovery': False, 'multi_agent': False,
                'scope': 'Every validation request disables implicit server skill/delegation tool injection; caller-declared tools remain unchanged.',
            }
        if template_path:
            data['reference_prompt_adapter'] = {
                'template': str(template_path),
                'template_sha256': hashlib.sha256(template_path.read_bytes()).hexdigest(),
                'released_template_sha256': normalized_template_hash,
                'fixture_file_sha256': fixture_file_hash,
                'template_hash_note': 'Released/embedded hash is UTF-8 text after newline normalization; fixture_file_sha256 hashes checkout bytes including any CRLF.',
                'current_date': '',
                'answer_suffix': ' to=user<|message|> only when thinking=false and no tools',
                'request_parser': 'reasoning_format=none only when thinking=false and no tools; default otherwise',
                'request_stop': ['<|eot|>'],
                'request_stop_scope': 'Only when thinking=false and no tools; preserves actual usage counts.',
            }
        (ROOT / f'provenance-{engine}-{dtype}-planned.json').write_text(json.dumps(data, indent=2) + '\n')
        print(engine, dtype, 'prepared', flush=True)
