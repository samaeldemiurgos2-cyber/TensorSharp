#!/usr/bin/env python3
"""Create a diagnostic GLM prefix header that shares verified, unmodified shards.

The pinned checkpoint has a metadata-only first shard. A fresh first-shard
header changes only layer metadata; sibling symlinks resolve to the original
weight files. This avoids copying tensor payloads under a storage quota. Extra
layers remain in the shared files but are not loaded by TensorSharp's prefix
executor. Validate a view against a materialized prefix before using it to
localize a numerical discrepancy. This is not a model-quality benchmark.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import time

from gguf import GGUFReader, GGUFValueType, GGUFWriter


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(data)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def identity(path):
    stat = path.stat()
    return dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
                device=stat.st_dev, inode=stat.st_ino)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verified-model', type=Path, required=True)
    parser.add_argument('--layers', type=int, required=True)
    parser.add_argument('--output-directory', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output_directory.exists() and not args.output_directory.is_symlink(),
            'Output directory must be fresh; preserve existing evidence.')
    source_report = json.loads(args.verified_model.read_text())
    require(source_report.get('status') == 'verified' and not source_report.get('failures'),
            'A passing full-shard verification report is required.')
    require(source_report['repo'] == 'unsloth/GLM-5.3-Flash-GGUF' and
            source_report['revision'] == 'a38483c8cd5df544f53d70fb281afe97369d5ab6',
            'This diagnostic is restricted to the documented checkpoint revision.')
    shards = sorted(source_report['shards'], key=lambda item: Path(item['path']).name)
    require(len(shards) == 4, 'Expected four verified shards.')
    readers, originals = [], []
    for index, shard in enumerate(shards, 1):
        path = Path(shard['path']).resolve(strict=True)
        expected_name = f'GLM-5.3-Flash-UD-IQ3_XXS-{index:05d}-of-00004.gguf'
        require(Path(shard['path']).name == expected_name, 'Unexpected shard filename.')
        require(shard['verified'] and shard['sha256'] == shard['actual_sha256'],
                'Every source shard must have passed its full-file hash check.')
        before = identity(path)
        require(before['bytes'] == shard['bytes'], 'Source size changed since verification.')
        reader = GGUFReader(str(path))
        require(int(reader.fields['split.no'].contents()) == index - 1 and
                int(reader.fields['split.count'].contents()) == 4, 'Unexpected split metadata.')
        readers.append(reader)
        originals.append(dict(path=str(path), link_name=expected_name,
                              sha256=shard['sha256'], identity=before))
    require(not readers[0].tensors, 'First shard must be metadata-only; never discard weights.')
    require(sha(originals[0]['path']) == originals[0]['sha256'], 'First-shard hash changed.')
    fields = readers[0].fields
    arch = fields['general.architecture'].contents()
    require(arch == 'glm5-next', 'Unexpected architecture.')
    blocks = int(fields[arch + '.block_count'].contents())
    nextn = int(fields[arch + '.nextn_predict_layers'].contents())
    require(0 < args.layers <= blocks - nextn, 'Prefix must contain only original trunk layers.')
    all_tensors = [tensor for reader in readers for tensor in reader.tensors]
    require(len({tensor.name for tensor in all_tensors}) == len(all_tensors), 'Duplicate tensor names.')
    require(int(fields['split.tensors.count'].contents()) == len(all_tensors),
            'Split tensor count must describe all shared weight files.')
    started = time.time()
    selected = []
    for tensor in all_tensors:
        match = re.fullmatch(r'blk\.(\d+)\..+', tensor.name)
        if tensor.name in {'token_embd.weight', 'output.weight', 'output_norm.weight'} or (
                match and int(match[1]) < args.layers):
            selected.append(dict(name=tensor.name, type=tensor.tensor_type.name,
                                 shape=[int(size) for size in tensor.shape], bytes=int(tensor.n_bytes),
                                 sha256=hashlib.sha256(memoryview(tensor.data).cast('B')).hexdigest()))
    require({'token_embd.weight', 'output.weight', 'output_norm.weight'} <=
            {item['name'] for item in selected}, 'Missing root tensors.')
    # No source path is opened for writing. The fresh directory contains only
    # our new metadata header, symlinks and a manifest, and is disposable.
    args.output_directory.mkdir(parents=True, exist_ok=False)
    output = args.output_directory / originals[0]['link_name']
    writer = GGUFWriter(output, arch)
    changes, expected = {}, {}
    for name, field in fields.items():
        if name.startswith('GGUF.') or name == 'general.architecture':
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
        expected[name] = value
        writer.add_key_value(name, value, field.types[0],
                             field.types[1] if field.types[0] == GGUFValueType.ARRAY else None)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    derived = GGUFReader(str(output))
    require(not derived.tensors, 'Derived first shard unexpectedly contains tensor payloads.')
    require(derived.fields['general.architecture'].contents() == arch, 'Architecture changed.')
    require({key for key in derived.fields if not key.startswith('GGUF.')} ==
            set(expected) | {'general.architecture'}, 'Metadata fields changed unexpectedly.')
    for name, value in expected.items():
        require(derived.fields[name].contents() == value, f'Metadata round-trip mismatch: {name}')
    for original in originals[1:]:
        link = args.output_directory / original['link_name']
        link.symlink_to(original['path'])
        require(link.is_symlink() and link.resolve(strict=True) == Path(original['path']),
                'Sibling link target mismatch.')
    for original in originals:
        require(identity(Path(original['path'])) == original['identity'], 'Source file identity changed.')
    report = dict(status='verified', diagnostic_only=True, format='shared-shard-prefix-view-v1',
                  layers=args.layers, repo=source_report['repo'], revision=source_report['revision'],
                  quant=source_report['quant'], source_verification=str(args.verified_model.resolve()),
                  source_verification_sha256=sha(args.verified_model), source_shards=originals,
                  generator_sha256=sha(__file__), output=str(output.resolve()),
                  output_bytes=output.stat().st_size, output_sha256=sha(output),
                  metadata_changes=changes, selected_tensors=selected,
                  shared_tensor_count=len(all_tensors), started_unix=started, finished_unix=time.time(),
                  limitations=['Full source-file hashes are linked to the prior verification report; selected tensor bytes are freshly hashed.',
                               'This view depends on the original shards remaining present and unchanged.',
                               'Extra layer tensors remain in shared files; only the prefix layer count is executed.',
                               'Require numerical equivalence to a materialized prefix before localization.',
                               'A truncated trained checkpoint does not establish answer quality or throughput.'])
    (args.output_directory / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in ('status', 'layers', 'output', 'output_bytes', 'output_sha256')}))


if __name__ == '__main__':
    main()
