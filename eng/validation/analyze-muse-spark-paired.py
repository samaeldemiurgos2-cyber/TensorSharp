"""Read-only analysis of this campaign's paired reports; never changes verdicts.

Usage: python analyze-muse-spark-paired.py --candidate TS.json --reference LLAMA.json
        --output NEW-ANALYSIS.json

The unchanged reusable comparator supplies the campaign verdict. Per-scenario
projections retain every repeat and expose which scenarios have equal measured
work. A descriptive task latency ratio with unequal completion counts never
becomes a parity verdict. Raw reports are hashed and never rewritten.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
HARNESS_PATH = ROOT / 'eng/muse-spark-validate.py'
spec = importlib.util.spec_from_file_location('spark_comparison', HARNESS_PATH)
harness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(harness)


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def distribution(values):
    return {'samples': values, 'count': len(values), 'median': statistics.median(values),
            'minimum': min(values), 'maximum': max(values)} if values else None


def project(report, scenario):
    result = dict(report)
    result['scenarios'] = [scenario]
    result['cases'] = [case for case in report['cases'] if case['scenario'] == scenario]
    return result


def campaign_errors(candidate, reference):
    errors = []
    if candidate.get('engine') != 'tensorsharp' or reference.get('engine') != 'llamacpp':
        errors.append('Expected TensorSharp candidate and llama.cpp reference')
    for name in ('weights_id', 'profile', 'thinking', 'sampling', 'scenarios', 'repeats',
                 'concurrency', 'harness_sha256', 'cache_policy'):
        if candidate.get(name) != reference.get(name):
            errors.append('Original campaign field differs: ' + name)
    for label, report in (('candidate', candidate), ('reference', reference)):
        scenarios, repeats = report.get('scenarios', []), report.get('repeats', 0)
        cases = report.get('cases', [])
        if not report.get('run_complete') or len(cases) != len(scenarios) * repeats:
            errors.append(label + ': original campaign is incomplete')
        keys = [(case['scenario'], case['tag']) for case in cases]
        if len(set(keys)) != len(keys):
            errors.append(label + ': duplicate original case keys')
        if any(case['scenario'] not in scenarios for case in cases):
            errors.append(label + ': unexpected original scenario')
        for scenario in scenarios:
            if sum(case['scenario'] == scenario for case in cases) != repeats:
                errors.append(label + ': wrong repeat count for ' + scenario)
    return errors


def observed_case(case):
    turns = []
    for turn in case.get('turns', []):
        metrics = turn.get('metrics') or {}
        request = turn.get('request') or {}
        start, first, last, end = (metrics.get(key) for key in
            ('t_start_abs', 't_first_abs', 't_last_abs', 't_end_abs'))
        streamed = request.get('stream') is True
        valid_clock = streamed and all(finite(x) for x in (start, first, last, end)) and start <= first <= last <= end
        completion = metrics.get('completion_tokens')
        window = last - first if valid_clock else None
        sustained = valid_clock and window > 0 and type(completion) is int and completion >= 512
        turns.append({'prompt_tokens': metrics.get('prompt_tokens'), 'completion_tokens': completion,
            'cached_tokens': ((metrics.get('raw_usage') or {}).get('prompt_tokens_details') or {}).get('cached_tokens'),
            'finish_reason': metrics.get('finish_reason'), 'streamed': streamed,
            'valid_client_clock_order': valid_clock,
            'client_first_generated_event_ms': (first - start) * 1000 if valid_clock else None,
            'client_generated_event_window_ms': window * 1000 if valid_clock else None,
            'client_event_window_tokens_per_second_estimate': (completion - 1) / window if sustained else None,
            'original_reported_decode_tps': metrics.get('decode_tps'),
            'reported_decode_timing_source': metrics.get('decode_timing_source'),
            'server_timings': metrics.get('server_timings', {}),
            'output_sha256': hashlib.sha256((metrics.get('output_text') or '').encode()).hexdigest()})
    return {'tag': case['tag'], 'status': case['status'], 'wall_ms': case.get('total_wall_ms'), 'turns': turns}


def metric_summary(cases, metric):
    values = []
    for case in cases:
        if metric == 'wall_ms':
            value = case[metric]
        elif len(case['turns']) == 1:
            value = case['turns'][0].get(metric)
        else:
            return None
        if not finite(value) or value < 0:
            return None
        values.append(value)
    return distribution(values)


def analyze(candidate, reference, tolerance=0.05, additional_global_errors=()):
    full = harness.comparison(candidate, reference, tolerance)
    global_errors = campaign_errors(candidate, reference) + list(additional_global_errors)
    result = {'original_harness_campaign_verdict': full, 'original_campaign_errors': global_errors,
              'acceptance_rules_changed': False, 'scenarios': {}}
    for scenario in dict.fromkeys(candidate.get('scenarios', []) + reference.get('scenarios', [])):
        left, right = project(candidate, scenario), project(reference, scenario)
        verdict = harness.comparison(left, right, tolerance)
        reasons = global_errors + verdict.get('reasons', [])
        equal_work = not global_errors and verdict['status'] in ('pass', 'fail')
        # The unchanged comparator emits this exact reason for output-work
        # mismatch. No quality/cache/prompt/provenance error may be removed.
        only_completion_differs = (not global_errors and verdict['status'] == 'inconclusive'
            and bool(reasons) and all(reason.endswith(': generated token counts differ') for reason in reasons))
        eligible_task = equal_work or only_completion_differs
        observations = {'candidate': [observed_case(c) for c in left['cases']],
                        'reference': [observed_case(c) for c in right['cases']]}
        walls = {engine: metric_summary(cases, 'wall_ms') for engine, cases in observations.items()}
        valid_walls = all(item and item['minimum'] > 0 for item in walls.values())
        if not valid_walls:
            eligible_task = equal_work = False
            reasons.append('Missing/nonpositive conversation wall measurements')
        task_ratio = walls['reference']['median'] / walls['candidate']['median'] if eligible_task else None
        item = {'all_original_repeats_retained': True, 'strict_scenario_projection_verdict': verdict,
            'equal_work_eligible': equal_work, 'eligibility_reasons': reasons,
            'task_latency': {'status': ('equal_work_descriptive' if equal_work else
                                       'descriptive_only_unequal_completion_work' if eligible_task else 'unavailable'),
                'conversation_wall_ms': walls, 'reference_over_candidate_median_wall_ratio': task_ratio,
                'parity_claim': False,
                'note': 'Task latency is time to the validated answer. Unequal output work prevents equal-work parity; no ratio is emitted for other validation/provenance/cache mismatches.'},
            'common_client_metrics': {}, 'observations': observations}
        for metric in ('client_first_generated_event_ms', 'client_event_window_tokens_per_second_estimate'):
            summaries = {engine: metric_summary(cases, metric) for engine, cases in observations.items()}
            available = equal_work and all(value and value['minimum'] > 0 for value in summaries.values())
            rate = metric.endswith('_estimate')
            ratio = ((summaries['candidate']['median'] / summaries['reference']['median']) if rate else
                     (summaries['reference']['median'] / summaries['candidate']['median'])) if available else None
            item['common_client_metrics'][metric] = {'status': 'descriptive' if available else 'unavailable',
                'observed_distributions': summaries, 'candidate_speed_ratio': ratio,
                'parity_claim': False}
        result['scenarios'][scenario] = item
    return result


LIMITATIONS = [
    'The original full-campaign verdict remains authoritative; per-scenario projections are diagnostics, not a replacement overall pass.',
    'All repeats are retained. No favorable-repeat selection or averaging away of a failed quality/cache observation is permitted.',
    'Client conversation wall time includes transport, queueing, prompt processing, generation and validation overhead.',
    'Client TTFT is the first nonempty generated SSE event (content, reasoning or tool fragment), not a server compute boundary. Planned no-tools, no-thinking requests make this a visible-content latency.',
    'The common event-window estimate is (completion_tokens-1)/(last-generated-event-first-generated-event), recomputed from the same client clock for both engines. It is emitted only for at least512 output tokens, valid timestamps and equal validated work.',
    'SSE events are not guaranteed to equal tokenizer tokens. Buffering, grouped frames and nonvisible special tokens can bias the event-window estimate. No per-token timestamps or inter-token latency distribution are recorded.',
    'Raw decode_tps may prefer a server timer on one engine and the client stream window on the other. These raw rates are not compared.',
    'prompt_tokens/TTFT is an end-to-end prefill proxy including queue/transport/first generation, not prompt-kernel throughput.',
    'Three sequential repeats provide a small descriptive sample, not statistical equivalence. Engine order, thermal state, page residency and competing work require separate phase/resource evidence.',
    'Matching counts does not prove matching token identities or numerical equivalence. Fixture-quality passes are bounded integration assertions, not broad model-quality parity.',
    'Reanalysis requires the same local harness/client/validator source hashes recorded by both reports. A source mismatch suppresses diagnostic ratios; the original recorded reports and acceptance rules are unchanged.',
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--tolerance', type=float, default=.05)
    args = parser.parse_args()
    if not all((args.candidate, args.reference, args.output)):
        parser.error('--candidate, --reference and --output are required')
    if args.output.exists():
        parser.error('Use a new output path; preserve prior evidence')
    if not 0 <= args.tolerance < 1:
        parser.error('tolerance must be in [0,1)')
    sources = {name: {'path': str(path.resolve()), 'sha256': checksum(path)}
               for name,path in [('candidate',args.candidate),('reference',args.reference)]}
    reports = {name: json.loads(Path(meta['path']).read_text(encoding='utf-8')) for name,meta in sources.items()}
    current_sources = {Path(path).name: checksum(Path(path)) for path in
        (HARNESS_PATH, harness.validation.__file__, harness.engines.__file__)}
    source_errors = [name + ': analysis source hashes differ from recorded harness/client/validator'
        for name, report in reports.items() if report.get('harness_sha256') != current_sources]
    result = analyze(reports['candidate'], reports['reference'], args.tolerance, source_errors)
    result.update(created_utc=datetime.now(timezone.utc).isoformat(), sources=sources,
        analyzer_sha256=checksum(Path(__file__)), comparator_source_sha256=checksum(HARNESS_PATH),
        analysis_only=True, limitations=LIMITATIONS,
        current_analysis_source_hashes=current_sources,
        original_recorded_comparison={name: report.get('comparison') for name,report in reports.items()},
        reported_harness_hashes={name: report.get('harness_sha256') for name,report in reports.items()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    for name,meta in sources.items():
        assert checksum(Path(meta['path'])) == meta['sha256'], name + ' source changed during analysis'
    print(json.dumps({'output':str(args.output), 'original_verdict':result['original_harness_campaign_verdict']['status']}))


if __name__ == '__main__':
    main()
