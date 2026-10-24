#!/usr/bin/env python3
"""Verify an actual Linux llama-server launch and capture read-only provenance.

Run after launching the prepared Muse Spark reference server. This tool checks
the live process, exact profile arguments, template and executable/library
hashes, then waits for GET /v1/models. It never submits inference requests.
Use a new output file for each launch; startup/prewarm are not benchmark time.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import time
import urllib.request
from urllib.parse import urlsplit


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def argument(argv, name):
    positions = [index for index, value in enumerate(argv) if value == name]
    require(len(positions) == 1 and positions[0] + 1 < len(argv),
            f'Expected one explicit {name} argument')
    return argv[positions[0] + 1]


def distribution_profile(planned):
    """Keep the intentionally different distribution outside shared work settings.

    Older five-GPU reports store layer_ranges in settings; retain their schema.
    New profiles name the actual strategy and partitions in engine_options.
    """
    settings = planned['settings']
    value = planned.get('engine_options', {}).get('distribution')
    if value is None:
        ranges = settings.get('layer_ranges')
        require(isinstance(ranges, list) and bool(ranges), 'Missing distribution profile')
        value = {'mode': 'layer', 'degree': len(ranges), 'layer_ranges': ranges}
    require(isinstance(value, dict), 'Distribution must be an object')
    mode, degree = value.get('mode'), value.get('degree')
    require(mode in ('layer', 'tensor'), 'Unsupported distribution mode')
    require(type(degree) is int and 1 <= degree <= 16, 'Invalid GPU degree')
    if 'gpu_count' in settings:
        require(type(settings['gpu_count']) is int and settings['gpu_count'] == degree,
                'Distribution degree differs from common GPU count')
    visible = settings.get('visible_gpu_devices')
    if visible is not None:
        require(isinstance(visible, list) and len(visible) == degree and
                all(type(device) is int and device >= 0 for device in visible) and
                len(set(visible)) == degree, 'Expected distinct physical GPU identities')
    layers = None
    if mode == 'layer':
        ranges = value.get('layer_ranges')
        require(isinstance(ranges, list) and len(ranges) == degree, 'Layer ranges differ from degree')
        cursor, layers = 0, []
        for pair in ranges:
            require(isinstance(pair, list) and len(pair) == 2 and
                    all(type(n) is int for n in pair), 'Invalid layer range')
            first, last = pair
            require(first == cursor and last >= first, 'Layer ranges must be contiguous and nonempty')
            layers.append(last - first + 1)
            cursor = last + 1
        require(cursor == 62, 'Layer ranges must cover all 62 decoder layers exactly once')
    else:
        require(degree >= 2 and 'layer_ranges' not in value,
                'Tensor parallelism cannot masquerade as whole-layer placement')
    return mode, degree, layers


def common_environment(settings):
    result = {'OMP_NUM_THREADS': str(settings['cpu_threads'])}
    if 'visible_gpu_devices' in settings:
        result['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, settings['visible_gpu_devices']))
    return result


def verify_no_speculation(argv, planned, environment=None):
    require(planned['settings'].get('speculation') == 'disabled',
            'This observer requires disabled speculation')
    # llama accepts underscores in long options and accumulates repeated
    # --spec-type values, including values supplied by its environment.
    normalized = [value.replace('_', '-') if value.startswith('--') else value for value in argv]
    explicit = '--spec-type' in normalized
    if planned.get('engine_options', {}).get('distribution') is not None:
        require(explicit, 'New reference profiles require explicit --spec-type none')
    if explicit:
        require(argument(normalized, '--spec-type') == 'none', 'Reference speculation must be none')
    for value in normalized:
        if not value.startswith('-') or value == '--spec-type':
            continue
        require(not (value.startswith(('--spec-', '--draft', '--lookup')) or
                     'draft' in value or value in ('-md', '-hfd', '-hfrd', '--mtp', '--dflash', '--eagle3')),
                'Unexpected speculative/draft option in no-spec profile: ' + value)
    for key, value in (environment or {}).items():
        if not value:
            continue
        if key == 'LLAMA_ARG_SPEC_TYPE':
            require(value == 'none', 'Inherited reference speculation is enabled')
        elif key.startswith(('LLAMA_ARG_SPEC_', 'LLAMA_ARG_DRAFT_', 'LLAMA_ARG_LOOKUP_')) or key.endswith('_DRAFT'):
            require(False, 'Unexpected inherited draft configuration: ' + key)
    # Legacy five-GPU records omitted --spec-type. They remain inspectable only
    # with no active CLI or inherited draft configuration; new plans are explicit.


def verify_arguments(argv, planned):
    settings = planned['settings']
    require(Path(argv[0]).name == 'llama-server', 'PID is not llama-server')
    verify_no_speculation(argv, planned)
    mode, degree, layers = distribution_profile(planned)
    require(mode == 'layer', 'Reference observer currently validates the measured layer-split baseline only')
    layers = layers.copy()
    layers[-1] += 1  # llama.cpp also places the output projection.
    url = urlsplit(planned['server_url'])
    expected = {
        '--n-cpu-moe': settings['cpu_expert_layer_count'],
        '--threads': settings['cpu_threads'], '--threads-batch': settings['cpu_threads'],
        '--cache-type-k': settings['key_cache_dtype'], '--cache-type-v': settings['value_cache_dtype'],
        '--parallel': settings['max_running_requests'], '--ctx-size': settings['aggregate_kv_pool_tokens'],
        '--kv-unified-per-slot': settings['context_per_request'],
        '--batch-size': settings['max_batched_tokens'], '--ubatch-size': settings['prefill_chunk_tokens'],
        '--predict': settings['max_output_tokens'], '--tensor-split': ','.join(map(str, layers)),
        '--device': ','.join(f'CUDA{i}' for i in range(degree)), '--gpu-layers': 'all', '--split-mode': 'layer',
        '--mmproj-device': 'CUDA0', '--override-tensor': r'token_embd\.weight=CUDA0',
        '--fit': 'off', '--flash-attn': 'on', '--reasoning': 'off',
        '--host': url.hostname, '--port': url.port,
    }
    for name, value in expected.items():
        require(argument(argv, name) == str(value), f'{name} differs from the planned profile')
    for flag in ('--kv-unified', '--cont-batching', '--jinja'):
        require(argv.count(flag) == 1, f'Missing or repeated {flag}')
    cache_flags = [value for value in argv if value in ('--cache-prompt', '--no-cache-prompt')]
    require(type(settings.get('prefix_cache')) is bool, 'Explicit prefix_cache setting required')
    expected_cache_flag = '--cache-prompt' if settings['prefix_cache'] else '--no-cache-prompt'
    require(bool(cache_flags) and cache_flags[-1] == expected_cache_flag,
            'Final effective prefix-cache flag differs from planned profile')
    if settings['prefix_cache'] is False:
        # cache_prompt=false skips prefix reuse, but llama's independent RAM
        # prompt cache can still save/load idle slot state. These copies are
        # unnecessary in this profile and are not measured by cached_tokens.
        options = planned.get('engine_options', {})
        require(type(options.get('prompt_cache_ram_mib')) is int and options['prompt_cache_ram_mib'] == 0
                and options.get('cache_idle_slots') is False,
                'No-cache reference requires explicit disabled RAM/idle-slot cache provenance')
        require(argument(argv, '--cache-ram') == '0', 'No-cache reference must disable the RAM prompt cache')
        idle_flags = [value for value in argv if value in ('--cache-idle-slots', '--no-cache-idle-slots')]
        require(bool(idle_flags) and idle_flags[-1] == '--no-cache-idle-slots',
                'No-cache reference must disable idle-slot state saves')
        expected['--cache-ram'] = 0
        expected['--no-cache-idle-slots'] = True
    # Reject aliases/negative flags that could override a checked long option.
    conflicting = {'-c', '-b', '-ub', '-t', '-tb', '-np', '-ngl', '-ts',
                   '--no-cont-batching', '--no-kv-unified'}
    require(not conflicting.intersection(argv), 'Conflicting profile override/alias in argv')
    first = [name for name in planned['weights_sha256'] if '-00001-of-' in name]
    require(len(first) == 1, 'Expected one verified first shard')
    require(argument(argv, '--model').replace('\\', '/').endswith('/' + first[0]),
            'Actual model path does not select the verified checkpoint')
    first_path = Path(planned['tokenizer_metadata']['file'])
    require(Path(argument(argv, '--model')).as_posix() == first_path.as_posix(),
            'Actual model path differs from the verified metadata source')
    require(Path(argument(argv, '--mmproj')).as_posix() == (first_path.parent.parent / 'mmproj-Q8_0.gguf').as_posix(),
            'Unexpected projector path')
    adapter = planned['reference_prompt_adapter']
    require(argument(argv, '--chat-template-file') == adapter['template'], 'Unexpected template path')
    return expected


def process(pid):
    directory = Path('/proc') / str(pid)
    argv = (directory / 'cmdline').read_bytes().decode().strip('\0').split('\0')
    environment = dict(item.split('=', 1) for item in
        (directory / 'environ').read_bytes().decode().split('\0') if '=' in item)
    stat = (directory / 'stat').read_text().rsplit(')', 1)[1].split()
    return argv, environment, int(stat[19])


def verify_binaries(planned):
    verified, resolved = {}, {}
    require(bool(planned.get('binary_hashes')), 'No compiled binary manifest')
    for name, expected in planned['binary_hashes'].items():
        path = Path(name)
        target = path.resolve(strict=True)
        if target not in resolved:
            resolved[target] = sha256(target)
        actual = resolved[target]
        require(actual == expected, f'Compiled binary changed: {name}')
        verified[name] = actual
    return verified


def owns_listen_port(pid, port, proc_root=Path('/proc')):
    """Do not accept another process's healthy endpoint while this one loads."""
    directory = proc_root / str(pid)
    inodes = set()
    for descriptor in (directory / 'fd').iterdir():
        try:
            target = os.readlink(descriptor)
        except OSError:
            continue
        if target.startswith('socket:[') and target.endswith(']'):
            inodes.add(target[8:-1])
    for name in ('tcp','tcp6'):
        path = directory / 'net' / name
        if not path.exists():
            continue
        for row in path.read_text().splitlines()[1:]:
            fields = row.split()
            if len(fields) >= 10 and fields[3] == '0A' and fields[9] in inodes:
                if int(fields[1].rsplit(':',1)[1],16) == port:
                    return True
    return False


def main(engine='llamacpp', profile_verifier=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--planned', type=Path, required=True)
    parser.add_argument('--pid', type=int, required=True)
    parser.add_argument('--server-log', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timeout', type=float, default=1800)
    args = parser.parse_args()
    require(args.timeout > 0 and args.pid > 0, 'Positive timeout and PID required')
    require(not args.output.exists(), 'Use a new output path; observed launch evidence is not overwritten')
    planned = json.loads(args.planned.read_text(encoding='utf-8'))
    require(planned.get('launch_observed') is False, 'Input must be planned provenance')
    argv, env, start = process(args.pid)
    if profile_verifier is None:
        checked_arguments = verify_arguments(argv, planned)
        verify_no_speculation(argv, planned, env)
        checked_environment = common_environment(planned['settings'])
        for key, value in planned.get('engine_options', {}).get('environment', {}).items():
            require(isinstance(key, str) and isinstance(value, str), 'Engine environment must contain string values')
            require(key not in checked_environment or checked_environment[key] == value,
                    'Engine option conflicts with common setting: ' + key)
            checked_environment[key] = value
        for key, value in checked_environment.items():
            require(env.get(key) == value, 'Observed environment differs: ' + key)
    else:
        checked_arguments, checked_environment = profile_verifier(argv, env, planned)
    require(args.server_log.is_file() and args.server_log.stat().st_size > 0, 'Startup log missing/empty')
    template = planned.get('reference_prompt_adapter')
    if template:
        require(sha256(Path(template['template'])) == template['template_sha256'], 'Reference template changed')
    verified = verify_binaries(planned)
    executable = Path(f'/proc/{args.pid}/exe').resolve(strict=True)
    if engine == 'tensorsharp':
        host_dll = next(Path(value) for value in argv if Path(value).name == 'TensorSharp.Server.Host.dll')
        cwd = Path(f'/proc/{args.pid}/cwd').resolve(strict=True)
        target = host_dll if host_dll.is_absolute() else cwd / host_dll
        require(any(Path(name).resolve() == target.resolve() for name in verified),
                'Live managed host is absent from binary manifest')
    else:
        require(any(Path(name).resolve() == executable for name in verified), 'Live executable is absent from binary manifest')
    runtime_executable_hash = sha256(executable)
    affinity = sorted(os.sched_getaffinity(args.pid))
    expected_affinity = planned['settings'].get('cpu_affinity')
    if expected_affinity is not None:
        require(isinstance(expected_affinity, list) and bool(expected_affinity) and
                all(type(cpu) is int and cpu >= 0 for cpu in expected_affinity) and
                len(set(expected_affinity)) == len(expected_affinity), 'Invalid planned CPU affinity')
        require(affinity == sorted(expected_affinity), 'Observed CPU affinity differs from common setting')
    deadline = time.monotonic() + args.timeout
    last_error = None
    while time.monotonic() < deadline:
        current_argv, current_environment, current_start = process(args.pid)
        require(start == current_start and argv == current_argv, 'Process identity changed')
        require(all(current_environment.get(key) == value for key, value in checked_environment.items()),
                'Checked environment changed before readiness')
        if engine == 'llamacpp':
            verify_no_speculation(current_argv, planned, current_environment)
        require(sorted(os.sched_getaffinity(args.pid)) == affinity, 'CPU affinity changed before readiness')
        try:
            require(owns_listen_port(args.pid,urlsplit(planned['server_url']).port),
                    'Observed PID does not own the expected listening port')
            with urllib.request.urlopen(planned['server_url'].rstrip('/') + '/v1/models', timeout=3) as response:
                models = json.load(response)
            require(any('muse-spark' in str(model.get('id', '')).lower() for model in models.get('data', [])),
                    'Ready endpoint did not advertise Muse Spark')
            break
        except (OSError, ValueError) as error:
            last_error = str(error)
            time.sleep(min(5, max(0, deadline - time.monotonic())))
    else:
        raise TimeoutError('Readiness not observed: ' + str(last_error))
    observed = datetime.now(timezone.utc).isoformat()
    boot = next(int(line.split()[1]) for line in Path('/proc/stat').read_text().splitlines() if line.startswith('btime '))
    started = datetime.fromtimestamp(boot + start / os.sysconf('SC_CLK_TCK'), timezone.utc).isoformat()
    result = dict(planned)
    result.update(launch_recipe=planned['server_command'], launch_observed=True,
        launch_observed_utc=observed, server_pid=args.pid, server_command=shlex.join(argv),
        server_start_ticks=start, observed_cpu_affinity=affinity,
        server_log=str(args.server_log), observed_environment=checked_environment,
        observed_checked_arguments=checked_arguments, ready_response=models,
        binaries_reverified_at_launch=verified,
        runtime_executable={'path':str(executable), 'sha256':runtime_executable_hash},
        phase_timestamps={'process_start_kernel_estimate_utc': started, 'http_ready_first_observed_utc': observed},
        phase_note='Readiness polling and binary verification precede inference. Record CPU expert prewarm separately; no inference requests were sent by this observer.',
        status='Actual arguments, binary/template identity and HTTP readiness verified; quality and performance not implied.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'pid': args.pid, 'launch_observed': True, 'output': str(args.output)}))


if __name__ == '__main__':
    main()
