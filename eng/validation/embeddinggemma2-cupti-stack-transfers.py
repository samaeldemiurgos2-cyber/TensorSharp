#!/usr/bin/env python3
"""Attribute the single activation pair in a forced 16-block CUDA vision stack."""
import argparse
import hashlib
import json
import re
from pathlib import Path


def attribute(report, probe, patches, blocks=16, width=768):
    # This scope deliberately remains separate from the retained block-loop
    # parser: a capture of that route must fail this stack qualification.
    if (report.get('schema_version') != 2 or report.get('completed') is not True or
            report.get('diagnostic') is not True or report.get('hardware_counters') is not False or
            not isinstance(report.get('cupti_api_version'), int) or report['cupti_api_version'] < 130000 or
            report.get('errors') != []):
        raise ValueError('Require a complete CUDA 13 activity-only capture')
    for field in ('dropped', 'incomplete', 'callback_exceptions', 'truncated', 'invalid_buffers'):
        if report.get(field) != 0:
            raise ValueError(field + ' must be zero')
    if patches <= 0 or blocks != 16 or width != 768:
        raise ValueError('Require positive patch count and qualified 16-block width768 geometry')
    if (probe.get('completed') is not True or probe.get('backend') != 'GGML_CUDA' or
            probe.get('require_validated_float32_precision') is not True or
            probe.get('require_ggml_cuda_stack') is not True or
            probe.get('ggml_cuda_stack_observation_available') is not True or
            probe.get('ggml_cuda_stack_encode_count') != 1 or
            probe.get('retained_block_loop_encode_count') != 0 or
            probe.get('raw_tower_benchmark') is not None or
            not probe.get('cupti_activity_output')):
        raise ValueError('Require a forced, observed single precise CUDA stack forward')
    if any(probe.get(key) is not None for key in ('vision_stage', 'vision_trace', 'vision_profile')):
        raise ValueError('Per-block diagnostics would change the stack route')
    pw, ph = probe.get('width'), probe.get('height')
    if type(pw) is not int or type(ph) is not int or pw <= 0 or ph <= 0 or (pw // 16) * (ph // 16) != patches:
        raise ValueError('Patch count must match the observed16-pixel floor image grid')

    kernels, copies = report['kernel_events'], report['memcpy_events']
    if report.get('kernel_count') != len(kernels) or report.get('memcpy_record_count') != len(copies):
        raise ValueError('Activity record counts mismatch')
    markers = sorted((k for k in kernels if 'vision_kernel<' in k['name']), key=lambda k: k['start_ns'])
    if len(markers) != blocks or any(re.search(r'\bvision_kernel<64>(?:\(|$)', k['name']) is None for k in markers):
        raise ValueError('Exactly one qualified head64 vision attention marker per block required')
    size = patches * width * 4
    uploads = sorted((c for c in copies if c['direction'] == 'HtoD' and c['bytes'] == size), key=lambda c: c['start_ns'])
    downloads = sorted((c for c in copies if c['direction'] == 'DtoH' and c['bytes'] == size), key=lambda c: c['start_ns'])
    if len(uploads) != 2 or len(downloads) != 2:
        raise ValueError('Require one initial patch pair and one complete stack activation pair')
    for event in [*markers, *uploads, *downloads]:
        start, end = event['start_ns'], event['end_ns']
        if type(start) is not int or type(end) is not int or not 0 < start <= end:
            raise ValueError('Invalid event timestamp')
    if any(c['copy_count'] != 1 for c in [*uploads, *downloads]):
        raise ValueError('Batched copy records are outside this attribution scope')
    if not uploads[0]['end_ns'] <= downloads[0]['start_ns'] <= downloads[0]['end_ns'] <= uploads[1]['start_ns']:
        raise ValueError('Initial patch-projection pair must finish before the stack upload')
    upload, download = uploads[1], downloads[1]
    if not upload['end_ns'] <= markers[0]['start_ns'] or not markers[-1]['end_ns'] <= download['start_ns']:
        raise ValueError('One stack copy pair must bracket all sixteen attention markers')
    if any(a['end_ns'] > b['start_ns'] for a, b in zip(markers, markers[1:])):
        raise ValueError('Consecutive attention markers overlap or are reordered')
    if len({(e['device'], e['context']) for e in [*markers, *uploads, *downloads]}) != 1:
        raise ValueError('Stack kernels, activation copies and excluded patch pair must share a device/context')
    if len({k['stream'] for k in markers}) != 1:
        raise ValueError('All sixteen dependent stack attention kernels must share a stream')
    sums = {direction: dict(count=1, bytes=size,
        sum_device_interval_ms=(event['end_ns'] - event['start_ns']) / 1e6)
        for direction, event in [('HtoD', upload), ('DtoH', download)]}
    return dict(completed=True, diagnostic=True, paired_performance_claim=False,
        attribution='Forced stack dispatch plus exact hidden byte size, all sixteen sequential attention markers bracketed by one upload/readback pair and shared CUDA context; initial patch pair excluded.',
        patches=patches, hidden_width=width, blocks=blocks, bytes_per_copy=size,
        stack_transfers=sums, total_stack_copy_bytes=2 * size,
        excluded_initial_pair=dict(upload=uploads[0], download=downloads[0]),
        excluded_hidden_sized_d2d_records=sum(c['direction'] == 'DtoD' and c['bytes'] == size for c in copies),
        stack=dict(upload=upload, attention_markers=markers, download=download),
        limitations=['CUPTI records expose logical copy sizes and ordering, not host tensor identities.',
            'This single cold forward includes weight loading; only the selected activation pair is attributed.',
            'Summed GPU copy intervals are neither wall time nor physical PCIe hardware counters; no service gain is inferred.',
            'Numerical oracle checks and immutable runtime/source guards are separate mandatory qualification evidence.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--probe', required=True, type=Path)
    parser.add_argument('--patches', required=True, type=int)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--expected-source-sha256', required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    source_hash = hashlib.sha256(args.source.read_bytes()).hexdigest()
    if source_hash != args.expected_source_sha256:
        raise ValueError('Actual executed native source identity mismatch')
    probe_raw, activity_raw = args.probe.read_bytes(), args.input.read_bytes()
    probe = json.loads(probe_raw)
    if Path(probe.get('cupti_activity_output', '')).resolve() != args.input.resolve():
        raise ValueError('Probe must identify this exact CUPTI capture path')
    result = attribute(json.loads(activity_raw), probe, args.patches)
    result['activity'] = dict(path=str(args.input.resolve()), sha256=hashlib.sha256(activity_raw).hexdigest())
    result['probe'] = dict(path=str(args.probe.resolve()), sha256=hashlib.sha256(probe_raw).hexdigest())
    result['owned_fused_source'] = dict(path=str(args.source.resolve()), sha256=source_hash)
    # Exclusive creation preserves prior raw evidence even across a duplicate run.
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
