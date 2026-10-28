#!/usr/bin/env python3
"""Make a byte-preserving GLM prefix for numerical diagnosis, not model quality.

Requires a prior full-shard verification report. Retains the original tokenizer,
embedding, output head and norm, plus the first N trunk layers. Model layer
metadata is truncated and NextN removed. Every selected tensor is hashed before
writing and checked byte-for-byte by hash after reopening the derived GGUF.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import time

from gguf import GGUFReader, GGUFValueType, GGUFWriter


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def tensor_sha(tensor):
    return hashlib.sha256(memoryview(tensor.data).cast('B')).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verified-model', type=Path, required=True)
    parser.add_argument('--layers', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix('.manifest.json').exists():
        parser.error('Output and manifest must be fresh; preserve earlier evidence.')
    source_report = json.loads(args.verified_model.read_text())
    assert source_report['status'] == 'verified' and not source_report.get('failures')
    assert source_report['repo'] == 'unsloth/GLM-5.3-Flash-GGUF'
    assert source_report['revision'] == 'a38483c8cd5df544f53d70fb281afe97369d5ab6'
    shards = source_report['shards']
    assert len(shards) == 4 and all(x['verified'] and x['sha256'] == x['actual_sha256'] for x in shards)
    readers = []
    source_files = []
    for shard in shards:
        path = Path(shard['path'])
        stat = path.stat()
        assert stat.st_size == shard['bytes']
        readers.append(GGUFReader(str(path)))
        source_files.append(dict(path=str(path), bytes=stat.st_size, sha256=shard['sha256'],
                                 mtime_ns=stat.st_mtime_ns))
    fields = readers[0].fields
    arch = fields['general.architecture'].contents()
    assert arch == 'glm5-next'
    blocks = int(fields[arch + '.block_count'].contents())
    nextn = int(fields[arch + '.nextn_predict_layers'].contents())
    assert 0 < args.layers <= blocks - nextn
    tensors = []
    for reader in readers:
        for tensor in reader.tensors:
            match = re.fullmatch(r'blk\.(\d+)\..+', tensor.name)
            if tensor.name in {'token_embd.weight', 'output.weight', 'output_norm.weight'} or (match and int(match[1]) < args.layers):
                tensors.append(tensor)
    names = {t.name for t in tensors}
    assert len(names) == len(tensors)
    assert {'token_embd.weight', 'output.weight', 'output_norm.weight'} <= names
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tensor_bytes = sum(t.n_bytes for t in tensors)
    assert shutil.disk_usage(args.output.parent).free >= tensor_bytes + 2 * 1024**3, 'Preserve at least2GiB output-volume reserve.'
    started = time.time()
    writer = GGUFWriter(args.output, arch)
    changes = {}
    for name, field in fields.items():
        if name.startswith(('GGUF.', 'split.')) or name == 'general.architecture':
            continue
        value = field.contents()
        original = value
        if name == arch + '.block_count':
            value = args.layers
        elif name == arch + '.nextn_predict_layers':
            value = 0
        elif name.startswith(arch + '.') and field.types[0] == GGUFValueType.ARRAY and len(value) in (blocks, blocks - nextn):
            value = value[:args.layers]
        if value != original:
            changes[name] = dict(original=original, derived=value)
        writer.add_key_value(name, value, field.types[0],
                             field.types[1] if field.types[0] == GGUFValueType.ARRAY else None)
    tensor_records = []
    for tensor in tensors:
        tensor_records.append(dict(name=tensor.name, type=tensor.tensor_type.name,
                                   shape=[int(x) for x in tensor.shape], bytes=int(tensor.n_bytes),
                                   sha256=tensor_sha(tensor)))
        writer.add_tensor(tensor.name, tensor.data, raw_dtype=tensor.tensor_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    derived = GGUFReader(str(args.output))
    assert int(derived.fields[arch + '.block_count'].contents()) == args.layers
    assert int(derived.fields[arch + '.nextn_predict_layers'].contents()) == 0
    assert not any(key.startswith('split.') for key in derived.fields)
    assert len(derived.tensors) == len(tensor_records)
    for original, tensor in zip(tensor_records, derived.tensors):
        assert original['name'] == tensor.name and original['type'] == tensor.tensor_type.name
        assert original['shape'] == [int(x) for x in tensor.shape]
        assert original['sha256'] == tensor_sha(tensor), tensor.name
    for source in source_files:
        stat = Path(source['path']).stat()
        assert stat.st_size == source['bytes'] and stat.st_mtime_ns == source['mtime_ns']
    report = dict(status='verified', diagnostic_only=True, layers=args.layers,
                  repo=source_report['repo'], revision=source_report['revision'], quant=source_report['quant'],
                  source_verification=str(args.verified_model), source_verification_sha256=sha(args.verified_model),
                  source_shards=source_files, generator_sha256=sha(__file__), output=str(args.output),
                  output_bytes=args.output.stat().st_size, output_sha256=sha(args.output),
                  metadata_changes=changes, tensors=tensor_records, started_unix=started, finished_unix=time.time(),
                  limitations=['Original full-shard SHA verification is linked, not repeated here; every selected tensor is rehashed.',
                               'Truncated trained checkpoint is for numerical diagnosis, not answer quality or throughput qualification.'])
    args.output.with_suffix('.manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in ('status', 'layers', 'output', 'output_bytes', 'output_sha256')}))


if __name__ == '__main__':
    main()
