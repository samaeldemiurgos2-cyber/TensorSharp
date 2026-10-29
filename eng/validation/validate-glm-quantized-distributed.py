#!/usr/bin/env python3
"""Run one opt-in GLM fixture through unsplit and loopback distributed topologies.

Uses an already built, isolated probe runtime. Four processes sharing one GPU
qualify global-rank arithmetic, not physical multi-GPU performance. No binaries
or fixture files are modified, and every output directory must be fresh.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import time


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', type=Path, required=True)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=['tp', 'layer'], default='tp')
    parser.add_argument('--nodes', type=int, default=4)
    parser.add_argument('--first-port', type=int, default=29571)
    parser.add_argument('--prefix', type=int)
    parser.add_argument('--chunk', type=int, default=64)
    parser.add_argument('--ubatch', type=int, default=128)
    parser.add_argument('--decode', type=int, default=8)
    parser.add_argument('--timeout', type=int, default=600)
    parser.add_argument('--require-no-expert-fallback', action='store_true',
                        help='Require every TP node to log exact row projections with zero full-weight fallbacks.')
    parser.add_argument('--byte-reference', type=Path,
                        help='Also require every vector to equal a saved probe.json reference byte-for-byte.')
    args = parser.parse_args()
    if args.nodes < 2 or min(args.chunk, args.ubatch, args.decode, args.timeout) < 1:
        parser.error('Need at least two nodes and positive chunk/ubatch/decode/timeout.')
    if not 1024 <= args.first_port <= 65535 - args.nodes:
        parser.error('Invalid loopback port range.')
    if args.require_no_expert_fallback and args.mode != 'tp':
        parser.error('Expert row-strip diagnostics apply to TP mode only.')
    if args.byte_reference and not args.byte_reference.is_file():
        parser.error('Saved byte-reference probe.json does not exist.')
    models = list(args.fixture.glob('*.gguf'))
    if len(models) != 1:
        parser.error('Fixture directory must contain exactly one GGUF.')
    model = models[0].resolve()
    probe = args.probe.resolve()
    source_tokens = json.loads((args.fixture / 'tokens.json').read_text())
    prefix = args.prefix if args.prefix is not None else len(source_tokens) - args.decode
    if prefix < 2 or prefix + args.decode > len(source_tokens):
        parser.error('Fixture must contain prefix + decode tokens, with prefix >= 2.')
    tokens = source_tokens[:prefix + args.decode]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    token_path = output / 'tokens.json'
    token_path.write_text(json.dumps(tokens) + '\n')
    binary_names = ['GgmlOps.dll', 'libGgmlOps.so', 'TensorSharp.Models.dll',
                    'TensorSharp.Distributed.dll', probe.name]
    binaries = {name: sha(probe.parent / name) for name in binary_names if (probe.parent / name).is_file()}
    if not any(name in binaries for name in ('GgmlOps.dll', 'libGgmlOps.so')):
        parser.error('Probe directory must contain its explicit native runtime library.')
    settings = dict(MAX_CONTEXT='1024', KV_CACHE_DTYPE='F16', TS_GLM_UBATCH=str(args.ubatch),
                    TS_GLM_THREADS='2', TS_PREFILL_WARMUP='0', TS_GLM_TP_OVERSUBSCRIBE='1', TS_N_CPU_MOE='0')
    env = os.environ.copy()
    env.update(settings)
    model_sha = sha(model)
    byte_reference = args.byte_reference.resolve() if args.byte_reference else None
    byte_reference_sha = sha(byte_reference) if byte_reference else None
    base = ['dotnet', str(probe), '--model', str(model), '--tokens', str(token_path),
            '--weights-id', model_sha, '--chunk', str(args.chunk), '--decode-count', str(args.decode),
            '--backend', 'ggml_cuda']
    report = dict(status='running', fixture=str(model), model_sha256=model_sha, binaries=binaries,
                  settings=settings, prefix=prefix, decode=args.decode, chunk=args.chunk,
                  inherited_numerical_settings={name: value for name, value in env.items()
                      if name.startswith(('TS_GLM_', 'GGML_CUDA_', 'TS_SPEC_', 'TS_CPU_MOE'))},
                  nodes=args.nodes, local_degree=1, mode=args.mode, runs=[],
                  require_no_expert_fallback=args.require_no_expert_fallback,
                  byte_reference=str(byte_reference) if byte_reference else None,
                  byte_reference_report_sha256=byte_reference_sha,
                  limitations=['Loopback processes may share one GPU; this is not physical multi-GPU qualification.',
                               'Synthetic fixed-token numerical evidence, not model quality or performance.'])
    report_path = output / 'run.json'

    def save():
        report_path.write_text(json.dumps(report, indent=2) + '\n')

    def execute(commands):
        children = []
        try:
            for name, command in commands:
                entry = dict(name=name, command=command, started_unix=time.time())
                report['runs'].append(entry)
                log = (output / (name + '.log')).open('w', encoding='utf-8')
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
                entry['pid'] = child.pid
                children.append((child, log, entry))
                save()
            deadline = time.monotonic() + args.timeout
            for child, _, entry in children:
                entry['exit_code'] = child.wait(timeout=max(0.1, deadline - time.monotonic()))
                entry['finished_unix'] = time.time()
                save()
                if entry['exit_code']:
                    raise RuntimeError(f"{entry['name']} exited {entry['exit_code']}; see its preserved log.")
        finally:
            for child, log, entry in children:
                if child.poll() is None:
                    child.kill()
                    child.wait(timeout=30)
                    entry['forced_stop'] = True
                log.close()
            save()

    try:
        execute([('local', base + ['--layer-split', '1', '--output', str(output / 'local')])])
        peers = ','.join(f'127.0.0.1:{args.first_port + node}' for node in range(args.nodes))
        mode = '--tp' if args.mode == 'tp' else '--layer-split'
        execute([(f'node{node}', base + [mode, '1', '--tp-node-id', str(node), '--tp-peers', peers,
                    '--output', str(output / f'node{node}')]) for node in range(args.nodes)])
        fallback_counts = {}
        for node in range(args.nodes):
            worker = json.loads((output / f'node{node}/probe.json').read_text())
            expected_status = 'completed' if node == 0 else 'worker_loop_returned'
            if worker.get('status') != expected_status or worker.get('disposed') is not True:
                raise RuntimeError(f'Node {node} did not complete its expected lifecycle.')
            diagnostics = (output / f'node{node}.log').read_text(encoding='utf-8', errors='replace')
            counts = [int(value) for value in re.findall(
                r'exact distributed expert projections: full-row MMQ geometry; (\d+) full-weight/crop fallbacks',
                diagnostics)]
            fallback_counts[f'node{node}'] = counts
            if args.require_no_expert_fallback and (not counts or any(counts)):
                raise RuntimeError(f'Node {node} did not confirm exact projections with zero full-weight fallbacks: {counts}.')
        report['expert_full_weight_fallback_counts'] = fallback_counts
        if binaries != {name: sha(probe.parent / name) for name in binaries}:
            raise RuntimeError('Probe runtime changed during validation.')
        spec = importlib.util.spec_from_file_location('distributed_compare',
            Path(__file__).with_name('compare-deepseek41-distributed.py'))
        compare = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(compare)
        result = compare.compare(output / 'local/probe.json', output / 'node0/probe.json',
            max_absolute_error=2e-5, max_rmse=2e-5, max_relative_l2=2e-5,
            max_normalized_error=2e-5, min_cosine=0.999999, require_same_argmax=True)
        (output / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
        report['status'] = result['status']
        report['max_absolute_error'] = max(v['max_absolute_error'] for v in result['vectors'])
        report['max_relative_l2_error'] = max(v['relative_l2_error'] for v in result['vectors'])
        report['vectors'] = len(result['vectors'])
        reference_steps = [step for step in json.loads((output / 'local/probe.json').read_text())['steps'] if step.get('logits')]
        candidate_steps = [step for step in json.loads((output / 'node0/probe.json').read_text())['steps'] if step.get('logits')]
        report['vector_hashes'] = [dict(operation=reference['operation'],
            reference_sha256=sha(output / 'local' / reference['logits']),
            candidate_sha256=sha(output / 'node0' / candidate['logits']))
            for reference, candidate in zip(reference_steps, candidate_steps)]
        report['all_vector_bytes_identical'] = len(reference_steps) == len(candidate_steps) and all(
            pair['reference_sha256'] == pair['candidate_sha256'] for pair in report['vector_hashes'])
        if byte_reference:
            saved_steps = [step for step in json.loads(byte_reference.read_text())['steps'] if step.get('logits')]
            expected_schedule = [(step['operation'], step['tokens'], step.get('count')) for step in saved_steps]
            actual_schedule = [(step['operation'], step['tokens'], step.get('count')) for step in reference_steps]
            if expected_schedule != actual_schedule or sha(byte_reference) != byte_reference_sha:
                raise RuntimeError('Saved byte-reference schedule differs or its report changed during validation.')
            report['saved_reference_vector_hashes'] = [sha(byte_reference.parent / step['logits']) for step in saved_steps]
            report['saved_reference_bytes_identical'] = all(
                pair['reference_sha256'] == pair['candidate_sha256'] == saved
                for pair, saved in zip(report['vector_hashes'], report['saved_reference_vector_hashes']))
            if not report['saved_reference_bytes_identical']:
                raise RuntimeError('Candidate or fresh unsplit vector differs byte-for-byte from the saved reference.')
        save()
        print(json.dumps({key: report[key] for key in ('status', 'vectors', 'max_absolute_error', 'max_relative_l2_error')}))
        return 0 if result['status'] == 'passed' else 1
    except Exception as error:
        report.update(status='error', error=str(error))
        save()
        raise


if __name__ == '__main__':
    raise SystemExit(main())
