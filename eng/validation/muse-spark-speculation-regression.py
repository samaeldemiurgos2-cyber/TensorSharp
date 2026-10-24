#!/usr/bin/env python3
"""Bounded F32 serving-configuration regression: edited copy, follow-up, limit17.

Run --phase nospec first, then --phase ngram --reference <nospec.json> against
separate observed launches. Never launches or reconfigures a server. The plain
and speculative attention/cache paths differ; this is exact greedy serving
correctness, NOT isolated speculation speed or broad model quality evidence.
No retry is made. A correct answer without actual rejection/rollback is recorded
as not_exercised, fails the requested mechanism gate, and is never called a pass.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shlex
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'benchmarks/engine_comparison'))
import engines

SCOPE = ('Exact greedy comparison of complete F32 serving configurations, including '
         'different attention/cache paths. No isolated speculation speed claim. '
         'Public chatcmpl IDs are not mapped to internal chat IDs; attribution uses '
         'fresh log segments, exact request counts, internal telemetry joins and idle queue counters.')


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def fresh_cursor(path):
    stat = Path(path).stat()
    return dict(device=stat.st_dev, inode=stat.st_ino, offset=stat.st_size)


def fresh_log(path, cursor):
    current = fresh_cursor(path)
    if any(current[k] != cursor[k] for k in ('device', 'inode')) or current['offset'] < cursor['offset']:
        raise ValueError('Server log rotated or was truncated during the request')
    with Path(path).open('rb') as stream:
        stream.seek(cursor['offset'])
        return stream.read().decode('utf-8', errors='replace')


def telemetry(text):
    steps = []
    for match in re.finditer(r'\[cb\] step#(\d+) (\w+) scheduled=(\d+) served=(\d+)([^\r\n]*)', text):
        members = [dict(request_id=m[1], kind=m[2], forwarded=int(m[3])) for m in
                   re.finditer(r'\|\s+(\S+):([PD])\s+fwd=(\d+)', match[5])]
        steps.append(dict(step=int(match[1]), path=match[2], scheduled=int(match[3]),
                          served=int(match[4]), members=members, error='error=' in match[5]))
    armed = re.findall(r'Speculative decoding armed for (\S+) \(algorithm=ngram\(', text)
    stats = [dict(request_id=m[1], drafted=int(m[2]), accepted=int(m[3]),
                  verify_steps=int(m[4]), plain_steps=int(m[5]), rollbacks=int(m[6])) for m in
             re.finditer(r'Speculative decoding stats for ([^:\s]+): drafted=(\d+) accepted=(\d+) '
                         r'acceptance=.*?verifySteps=(\d+) plainSteps=(\d+) rollbacks=(\d+)', text)]
    return dict(chat_starts=len(re.findall(r'\bchat\.start\s+arch=', text)), steps=steps,
                armed_ids=armed, speculation_stats=stats,
                actual_paths=sorted({s['path'] for s in steps}),
                internal_ids=sorted({m['request_id'] for s in steps for m in s['members']}))


def telemetry_failures(observed, count, mode):
    failures = []
    if observed['chat_starts'] != count or len(observed['internal_ids']) != count:
        failures.append('Fresh chat.start/internal request counts do not match the submitted workload')
    for step in observed['steps']:
        ids = [m['request_id'] for m in step['members']]
        if (step['error'] or not 0 < step['served'] <= step['scheduled']
                or len(ids) != step['served'] or len(set(ids)) != len(ids)
                or any(m['forwarded'] <= 0 for m in step['members'])):
            failures.append('Malformed, failed or empty actual scheduler step')
            break
    if mode in ('plain', 'batch', 'mixed') and (observed['armed_ids'] or observed['speculation_stats']):
        failures.append('Non-speculative workload unexpectedly speculated')
    if mode in ('batch', 'mixed'):
        matches = [s for s in observed['steps'] if s['path'] == 'BatchedPaged' and s['served'] >= 2
                   and (mode != 'mixed' or {m['kind'] for m in s['members']} == {'P', 'D'})]
        if not matches:
            failures.append('No actual packed' + (' mixed prefill/decode' if mode == 'mixed' else '') + ' step')
    if mode.startswith('ngram'):
        ids = observed['internal_ids']
        stats = observed['speculation_stats']
        # Current linear ngram prefill may arm/dispose the same request on
        # several chunks before the final decode context. Join unique IDs;
        # duplicate same-ID arming is not another request or completion.
        if (count != 1 or not observed['armed_ids'] or sorted(set(observed['armed_ids'])) != ids or len(stats) != 1
                or stats[0]['request_id'] not in ids):
            failures.append('Arming, completion statistics and actual scheduler request IDs do not correlate')
        elif not (0 <= stats[0]['accepted'] <= stats[0]['drafted']
                  and stats[0]['rollbacks'] <= stats[0]['verify_steps']):
            failures.append('Impossible speculative counters')
        elif mode != 'ngram_observe':
            stat = stats[0]
            if not (stat['accepted'] > 0 and stat['verify_steps'] > 0):
                failures.append('not_exercised: no accepted and verified drafts')
            if mode == 'ngram_reject' and not (stat['drafted'] > stat['accepted'] and stat['rollbacks'] > 0):
                failures.append('not_exercised: rejected drafts and actual rollback were not both observed')
        if 'SpeculativePerSequence' not in observed['actual_paths']:
            failures.append('No actual SpeculativePerSequence execution observed')
    return failures


def queue_status(url):
    with urllib.request.urlopen(url.rstrip('/') + '/api/queue/status', timeout=10) as response:
        status = json.load(response)
    for name in ('processing', 'pending_requests', 'total_processed'):
        if type(status.get(name)) is not int or status[name] < 0:
            raise ValueError('Missing/invalid live engine counter: ' + name)
    return status


def require_idle(status):
    if status['processing'] or status['pending_requests'] or status.get('busy') is not False:
        raise ValueError('Server has active or queued traffic; use a dedicated idle server')


def wait_idle(url, expected_total=None, timeout=10):
    deadline = time.monotonic() + timeout
    while True:
        status = queue_status(url)
        if expected_total is not None and status['total_processed'] > expected_total:
            raise ValueError('Unexpected concurrent request completions')
        if status['processing'] == status['pending_requests'] == 0:
            require_idle(status)
            if expected_total is None or status['total_processed'] == expected_total:
                return status
        if time.monotonic() >= deadline:
            raise ValueError('Engine did not become idle at the exact expected request completion count')
        time.sleep(.1)


def live_identity(provenance):
    directory = Path('/proc') / str(provenance['server_pid'])
    argv = (directory / 'cmdline').read_bytes().decode().strip('\0').split('\0')
    stat = (directory / 'stat').read_text().rsplit(')', 1)[1].split()
    if stat[0] == 'Z' or shlex.join(argv) != provenance['server_command']:
        raise ValueError('Observed server exited or its command changed')
    start = int(stat[19])
    if provenance.get('server_start_ticks', start) != start:
        raise ValueError('Observed server PID was reused')
    return dict(pid=provenance['server_pid'], start_ticks=start, argv=argv,
                executable=str((directory / 'exe').resolve(strict=True)))


def fixtures(model):
    original = '\n'.join(f'{i:02d} Aqua17 lanterns glow beside the quiet silver river.' for i in range(1, 9))
    edited = original.replace('river.', 'harbor.')
    bodies = [
        ('edited_copy', 'Copy all eight numbered lines between <copy> and </copy>, but replace '
         'the final word river with harbor in EVERY line. Preserve every other character. '
         'Output only the eight changed lines, without explanation or code fences.\n<copy>\n' + original + '\n</copy>', 256, edited),
        ('after_rollback', 'Return exactly the two digits 57. No other text.', 16, '57'),
        ('limit17', 'Copy the text between <copy> and </copy> verbatim. Output only the copied '
         'text, without introduction, explanation, or code fences.\n<copy>\n' + original + '\n</copy>', 17, original),
    ]
    return [dict(name=name, expected=expected, request=dict(model=model, think=False, stream=True,
        skills=[], skills_discovery=False, multi_agent=False, stream_options={'include_usage': True},
        timings_per_token=True, temperature=0, top_p=1, top_k=0, seed=42, repeat_penalty=1,
        presence_penalty=0, frequency_penalty=0, max_tokens=maximum,
        messages=[dict(role='user', content=prompt)])) for name, prompt, maximum, expected in bodies]


def content_failures(case, metrics, wire):
    failures = []
    output = metrics.get('assistant_message', {}).get('content') or ''
    if not any(line.strip() == 'data: [DONE]' for line in wire.get('sse_lines', [])):
        failures.append('Missing SSE completion')
    if not metrics.get('usage_present') or metrics.get('prompt_tokens', 0) <= 0 or metrics.get('completion_tokens', 0) <= 0:
        failures.append('Missing/invalid positive token usage')
    cached = metrics.get('raw_usage', {}).get('prompt_tokens_details', {}).get('cached_tokens')
    if type(cached) is not int or cached != 0:
        failures.append('Fresh-KV run requires measured zero cached tokens')
    if case['name'] == 'limit17':
        if (metrics.get('finish_reason') != 'length' or metrics.get('completion_tokens') != 17
                or not output or not case['expected'].startswith(output)):
            failures.append('Limit17 must return an exact nonempty prefix with exactly 17 tokens and length finish')
    elif output.strip() != case['expected'] or metrics.get('finish_reason') != 'stop':
        failures.append('Exact expected content/normal stop check failed')
    return failures


def reference_failures(reference, actual, phase):
    if phase == 'nospec':
        return []
    failures = []
    if not reference or not reference.get('passed') or reference.get('phase') != 'nospec':
        return ['A fully passing fresh F32 nospec reference is required']
    reference_cases = reference.get('cases', [])
    if len(reference_cases) != 3:
        return ['Reference must cover all three cases']
    try:
        expected_cases = fixtures(reference_cases[0]['request']['model'])
        for baseline, expected in zip(reference_cases, expected_cases):
            if (baseline.get('status') != 'passed' or baseline['name'] != expected['name']
                    or baseline['request'] != expected['request'] or baseline['expected'] != expected['expected']
                    or baseline['output'] != baseline['metrics'].get('assistant_message',{}).get('content')
                    or content_failures(expected, baseline['metrics'], baseline['wire'])
                    or telemetry_failures(baseline['telemetry'], 1, 'plain')):
                failures.append('Reference content/protocol/telemetry did not independently pass: ' + expected['name'])
    except (KeyError, TypeError, ValueError):
        return ['Reference is missing required per-case evidence']
    before, after = reference['provenance'], actual['provenance']
    for key in ('binary_hashes', 'weights_sha256'):
        if not before.get(key) or before[key] != after.get(key):
            failures.append('Reference identity differs: ' + key)
    if {k:v for k,v in before['settings'].items() if k != 'speculation'} != {k:v for k,v in after['settings'].items() if k != 'speculation'}:
        failures.append('Reference serving settings differ beyond speculation')
    if ({k:v for k,v in before.get('engine_options', {}).get('environment', {}).items() if k != 'TS_SPEC'} !=
            {k:v for k,v in after.get('engine_options', {}).get('environment', {}).items() if k != 'TS_SPEC'}):
        failures.append('Reference optimization environment differs beyond TS_SPEC')
    for row, baseline in zip(actual['cases'], reference['cases']):
        if row['request'] != baseline['request']:
            failures.append('Exact request differs: ' + row['name'])
        if row.get('output') != baseline.get('output'):
            failures.append('Exact greedy output differs: ' + row['name'])
        for key in ('prompt_tokens', 'completion_tokens', 'finish_reason'):
            if row.get('metrics', {}).get(key) != baseline.get('metrics', {}).get(key):
                failures.append('Token workload/finish differs: ' + row['name'] + '/' + key)
    return failures


def run(args):
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x', encoding='utf-8') as stream:
        result = dict(phase=args.phase, started_utc=utc(), status='running', scope=SCOPE,
                      tool_sha256=digest(__file__), cases=[], failures=[], passed=False)
        def save():
            stream.seek(0)
            json.dump(result, stream, indent=2, ensure_ascii=False)
            stream.write('\n'); stream.truncate(); stream.flush()
        save()
        try:
            provenance = json.loads(Path(args.provenance).read_text())
            settings = provenance['settings']
            if (not provenance.get('launch_observed') or settings['key_cache_dtype'] != 'f32'
                    or settings['value_cache_dtype'] != 'f32' or settings['prefix_cache'] is not False
                    or settings['speculation'] != ('disabled' if args.phase == 'nospec' else 'ngram')):
                raise ValueError('Requires actually observed, no-prefix-cache F32 phase profile')
            if provenance['server_url'].rstrip('/') != args.url.rstrip('/') or Path(provenance['server_log']) != Path(args.server_log):
                raise ValueError('URL/log differ from observed profile')
            result['provenance'] = provenance
            result['provenance_sha256'] = digest(args.provenance)
            result['server_identity'] = live_identity(provenance)
            reference = json.loads(Path(args.reference).read_text()) if args.reference else None
            if args.phase == 'ngram':
                initial = reference_failures(reference, result, args.phase)
                if initial:
                    raise ValueError('; '.join(initial))
            for case in fixtures(args.model):
                if live_identity(provenance) != result['server_identity']:
                    raise ValueError('Server identity changed between cases')
                row = dict(case, started_utc=utc(), wire={}, failures=[])
                result['cases'].append(row)
                row['queue_before'] = wait_idle(args.url)
                row['log_cursor'] = fresh_cursor(args.server_log)
                save()
                row['metrics'] = engines._run_streaming(args.url.rstrip('/') + '/v1/chat/completions',
                    case['request'], args.timeout, row['wire'])
                row['output'] = row['metrics'].get('assistant_message', {}).get('content') or ''
                row['queue_after'] = wait_idle(args.url, row['queue_before']['total_processed'] + 1)
                if live_identity(provenance) != result['server_identity']:
                    raise ValueError('Server identity changed during request')
                # Final completion telemetry can flush just after the HTTP body.
                for attempt in range(30):
                    row['fresh_server_log'] = fresh_log(args.server_log, row['log_cursor'])
                    row['telemetry'] = telemetry(row['fresh_server_log'])
                    if args.phase == 'nospec' or row['telemetry']['speculation_stats']:
                        break
                    time.sleep(.1)
                mode = 'plain' if args.phase == 'nospec' else {
                    'edited_copy':'ngram_reject', 'after_rollback':'ngram_observe', 'limit17':'ngram_accept'}[case['name']]
                row['failures'] = content_failures(case, row['metrics'], row['wire']) + telemetry_failures(row['telemetry'], 1, mode)
                row['status'] = ('not_exercised' if row['failures'] and all(x.startswith('not_exercised:') for x in row['failures'])
                                 else 'failed' if row['failures'] else 'passed')
                row['finished_utc'] = utc()
                save()
            result['failures'].extend(reference_failures(reference, result, args.phase))
        except (Exception, KeyboardInterrupt) as error:
            result['failures'].append(f'{type(error).__name__}: {error}')
        result['failures'].extend(row['name'] + ': ' + error for row in result['cases'] for error in row['failures'])
        if len(result['cases']) != 3 or any(row.get('status') is None for row in result['cases']):
            result['failures'].append('All three cases did not complete; no retry was attempted')
        result.update(passed=not result['failures'], status='passed' if not result['failures'] else 'failed', finished_utc=utc())
        save()
    print(json.dumps(dict(passed=result['passed'], failures=result['failures'], output=str(destination))))
    return 0 if result['passed'] else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['nospec', 'ngram'], required=True)
    parser.add_argument('--url', default='http://127.0.0.1:8083')
    parser.add_argument('--model', default='muse-spark')
    parser.add_argument('--server-log', required=True)
    parser.add_argument('--provenance', required=True)
    parser.add_argument('--reference')
    parser.add_argument('--out', required=True)
    parser.add_argument('--timeout', type=float, default=3600)
    args = parser.parse_args(argv)
    if args.timeout <= 0 or (args.phase == 'ngram') != bool(args.reference):
        parser.error('Positive timeout required; only ngram must receive --reference')
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
