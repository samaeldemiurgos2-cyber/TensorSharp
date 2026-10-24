#!/usr/bin/env python3
"""Exact-copy requests plus fresh server telemetry. Never launches a server.

After the corresponding profile is ready:
 python eng/validation/muse-spark-runtime-probe.py --mode batch --server-log /workspace/tensorsharp-f32-batch.log --out docs/validation/muse-spark/batch-runtime.json
 python eng/validation/muse-spark-runtime-probe.py --mode ngram --server-log /workspace/tensorsharp-ngram.log --out docs/validation/muse-spark/ngram-runtime.json

Use --mode plain on the baseline server for the identical solo request; pass its
result as --reference to the ngram run to check exact greedy output parity.
These mechanism probes are not a broad quality or throughput parity benchmark.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'benchmarks/engine_comparison'))
import engines


def telemetry(log):
    steps = []
    for match in re.finditer(r'\[cb\] step#(\d+) BatchedPaged scheduled=(\d+) served=(\d+)([^\r\n]*)', log):
        steps.append({'step': int(match[1]), 'scheduled': int(match[2]), 'served': int(match[3]),
                      'mixed_prefill_decode': ':P ' in match[4] and ':D ' in match[4],
                      'error': 'error=' in match[4]})
    stats = []
    for match in re.finditer(r'Speculative decoding stats for ([^:]+): drafted=(\d+) accepted=(\d+) acceptance=.*?verifySteps=(\d+) plainSteps=(\d+) rollbacks=(\d+)', log):
        stats.append(dict(request_id=match[1], drafted=int(match[2]), accepted=int(match[3]),
                          verify_steps=int(match[4]), plain_steps=int(match[5]), rollbacks=int(match[6])))
    return {'batch_steps': steps, 'speculation_stats': stats,
            'ngram_armed': bool(re.search(r'Speculative decoding armed .*algorithm=ngram\(', log))}


def valid_copy(metrics, wire, expected):
    output = metrics.get('assistant_message', {}).get('content') or ''
    normalized = '\n'.join(part.rstrip() for part in output.strip().splitlines())
    done = any(line.strip() == 'data: [DONE]' for line in wire.get('sse_lines', []))
    return (normalized == expected and done and metrics.get('usage_present', False)
            and metrics.get('finish_reason') == 'stop'
            and metrics.get('prompt_tokens', 0) > 0 and metrics.get('completion_tokens', 0) > 0)


def mechanism_failures(mode, observed):
    failures = []
    if mode == 'batch' and not any(
            s['served'] >= 2 and s['scheduled'] >= s['served'] and not s['error']
            for s in observed['batch_steps']):
        failures.append('No successful BatchedPaged step actually served >=2 requests.')
    if mode == 'ngram' and not (observed['ngram_armed'] and any(
            0 < s['accepted'] <= s['drafted'] and s['verify_steps'] > 0
            for s in observed['speculation_stats'])):
        failures.append('No observed ngram draft acceptance and verification; configured flags alone do not pass.')
    return failures


def run(args):
    log_path = Path(args.server_log)
    start_offset = log_path.stat().st_size
    tags = ['Aqua17', 'Bronze29', 'Coral43'] if args.mode == 'batch' else ['Aqua17']
    barrier = threading.Barrier(len(tags))

    def one(tag):
        # Distinct numbered rows still exercise prompt-copy ngram proposals,
        # without looking like an accidental generation loop to serving guards.
        expected = '\n'.join(f'{i:02d} {tag} lanterns glow beside quiet rivers.' for i in range(1, 13))
        body = {'model': args.model, 'think': False, 'stream': True,
                'skills': [], 'skills_discovery': False, 'multi_agent': False,
                'stream_options': {'include_usage': True}, 'timings_per_token': True,
                'temperature': 0, 'top_p': 1, 'top_k': 0, 'seed': 42,
                'repeat_penalty': 1, 'presence_penalty': 0, 'frequency_penalty': 0,
                'max_tokens': 512,
                'messages': [{'role': 'user', 'content':
                    'Copy the text between <copy> and </copy> verbatim. Preserve all 12 lines. '
                    'Output only the copied text, without introduction, explanation, or code fences.\n'
                    f'<copy>\n{expected}\n</copy>'}]}
        wire = {}
        barrier.wait()
        metrics = engines._run_streaming(args.url.rstrip('/') + '/v1/chat/completions', body, args.timeout, wire)
        output = metrics.get('assistant_message', {}).get('content') or ''
        return {'tag': tag, 'request': body, 'expected': expected, 'metrics': metrics, 'wire': wire,
                'correct': valid_copy(metrics, wire, expected),
                'output': output}

    result = {'mode': args.mode, 'log_start_offset': start_offset, 'requests': [], 'failures': [],
              'request_adapter': {'skills': [], 'skills_discovery': False, 'multi_agent': False}}
    try:
        with ThreadPoolExecutor(max_workers=len(tags)) as pool:
            result['requests'] = list(pool.map(one, tags))
    except Exception as error:
        result['failures'].append(f'{type(error).__name__}: {error}')
    # Completion may arrive just before its final structured log line is flushed.
    for _ in range(30):
        with log_path.open('rb') as stream:
            stream.seek(start_offset)
            log = stream.read().decode('utf-8', errors='replace')
        observed = telemetry(log)
        if args.mode != 'ngram' or observed['speculation_stats']:
            break
        time.sleep(0.1)
    result['telemetry'] = observed
    result['fresh_server_log'] = log
    if len(result['requests']) != len(tags) or not all(item['correct'] for item in result['requests']):
        result['failures'].append('Exact copy, SSE completion, or usage check failed.')
    result['failures'].extend(mechanism_failures(args.mode, observed))
    if args.reference:
        baseline = json.loads(Path(args.reference).read_text())
        if not baseline.get('passed'):
            result['failures'].append('The reference run did not pass.')
        result['exact_reference_match'] = [r['output'] for r in result['requests']] == [r['output'] for r in baseline['requests']]
        if not result['exact_reference_match']:
            result['failures'].append('Greedy output differs from the reference request.')
    result['passed'] = not result['failures']
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'passed': result['passed'], 'failures': result['failures'], 'telemetry': observed}, indent=2))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['plain', 'batch', 'ngram'], required=True)
    parser.add_argument('--url', default='http://127.0.0.1:8083')
    parser.add_argument('--model', default='muse-spark')
    parser.add_argument('--timeout', type=float, default=7200)
    parser.add_argument('--server-log', required=True)
    parser.add_argument('--reference')
    parser.add_argument('--out', required=True)
    raise SystemExit(run(parser.parse_args()))
