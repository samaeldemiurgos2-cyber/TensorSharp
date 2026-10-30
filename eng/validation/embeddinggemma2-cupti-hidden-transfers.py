#!/usr/bin/env python3
"""Attribute hidden-sized transfers by the qualified 16-block vision call order."""
import argparse
import hashlib
import json
from pathlib import Path


def attribute(report, patches, blocks=16, width=768):
    if (report.get('schema_version') != 2 or report.get('completed') is not True or
            report.get('diagnostic') is not True or report.get('hardware_counters') is not False or
            not isinstance(report.get('cupti_api_version'), int) or report['cupti_api_version'] < 130000 or
            report.get('errors') != []):
        raise ValueError('Require a complete CUDA 13 activity-only capture')
    for field in ('dropped','incomplete','callback_exceptions','truncated','invalid_buffers'):
        if report.get(field) != 0:
            raise ValueError(field + ' must be zero')
    if patches <= 0 or blocks <= 0 or width <= 0:
        raise ValueError('Positive dimensions and block count required')
    kernels, copies = report['kernel_events'], report['memcpy_events']
    if report.get('kernel_count') != len(kernels) or report.get('memcpy_record_count') != len(copies):
        raise ValueError('Activity record counts mismatch')
    markers = sorted((k for k in kernels if 'vision_kernel<' in k['name']), key=lambda k:k['start_ns'])
    if len(markers) != blocks:
        raise ValueError('Exactly one qualified head64 vision attention marker per block required')
    size = patches * width * 4
    uploads = sorted((c for c in copies if c['direction']=='HtoD' and c['bytes']==size), key=lambda c:c['start_ns'])
    downloads = sorted((c for c in copies if c['direction']=='DtoH' and c['bytes']==size), key=lambda c:c['start_ns'])
    if len(uploads) != blocks+1 or len(downloads) != blocks+1:
        raise ValueError('Require one initial non-block pair and exactly one pair per block')
    for event in [*markers,*uploads,*downloads]:
        if not 0 < event['start_ns'] <= event['end_ns']:
            raise ValueError('Invalid event timestamp')
    if not uploads[0]['end_ns'] <= downloads[0]['start_ns'] <= downloads[0]['end_ns'] <= uploads[1]['start_ns']:
        raise ValueError('Initial patch-projection pair is not separate from the first block')
    pairs=[]
    for i,(kernel,upload,download) in enumerate(zip(markers,uploads[1:],downloads[1:])):
        if not upload['end_ns'] <= kernel['start_ns'] <= kernel['end_ns'] <= download['start_ns']:
            raise ValueError('Block copy pair must bracket its attention marker')
        if i and downloads[i]['end_ns'] > upload['start_ns']:
            raise ValueError('Consecutive block copy pairs overlap or are reordered')
        if len({(e['device'],e['context']) for e in (kernel,upload,download)}) != 1:
            raise ValueError('Block kernels and copies must share a device/context')
        if upload['copy_count'] != 1 or download['copy_count'] != 1:
            raise ValueError('Batched copy records are outside this attribution scope')
        pairs.append(dict(block=i,attention=kernel,upload=upload,download=download))
    sums={direction:dict(count=blocks,bytes=blocks*size,
        sum_device_interval_ms=sum((p[key]['end_ns']-p[key]['start_ns'])/1e6 for p in pairs))
        for direction,key in [('HtoD','upload'),('DtoH','download')]}
    return dict(completed=True,diagnostic=True,paired_performance_claim=False,
        attribution='Hidden byte size plus exact sequential per-block upload/attention/readback order and shared CUDA context; initial patch-projection pair excluded.',
        patches=patches,hidden_width=width,blocks=blocks,bytes_per_copy=size,
        block_transfers=sums,total_block_copy_bytes=2*blocks*size,
        excluded_initial_pair=dict(upload=uploads[0],download=downloads[0]),
        excluded_hidden_sized_d2d_records=sum(c['direction']=='DtoD' and c['bytes']==size for c in copies),
        pairs=pairs,limitations=['Source call sequence identifies logical hidden transfers; CUPTI records do not expose host tensor identities.',
            'This cold process includes weight loading, which is excluded from the selected hidden copy pairs.',
            'Summed GPU copy activity intervals are neither wall time nor physical PCIe hardware counters; no service gain is inferred.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True,type=Path)
    parser.add_argument('--patches',required=True,type=int)
    parser.add_argument('--source',required=True,type=Path)
    parser.add_argument('--expected-source-sha256',required=True)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    if args.output.exists():
        raise ValueError('Output must be new')
    source_hash=hashlib.sha256(args.source.read_bytes()).hexdigest()
    if source_hash != args.expected_source_sha256:
        raise ValueError('Actual executed native source identity mismatch')
    result=attribute(json.loads(args.input.read_text()),args.patches)
    result['activity']=dict(path=str(args.input.resolve()),sha256=hashlib.sha256(args.input.read_bytes()).hexdigest())
    result['owned_fused_source']=dict(path=str(args.source.resolve()),sha256=source_hash)
    args.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main()
