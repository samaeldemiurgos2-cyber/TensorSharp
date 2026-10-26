#!/usr/bin/env python3
"""Add missing synthetic BPE merges metadata without changing inference tensors.

The original fixture declares gpt2 but omits the required merges array. This
derived fixture uses the original synthetic token IDs and an empty merge table;
it is not a language tokenizer or a trained-model-quality fixture.
--byte-tokenizer creates a separate HTTP-only byte vocabulary; all model
tensor bytes and the default numerical fixture derivation remain unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from gguf import GGUFReader, GGUFWriter, GGUFValueType


class ExplicitEmptyStringArrayWriter(GGUFWriter):
    # The installed writer refuses every empty array even with an explicit
    # element type. Emit the standard typed zero-length encoding in this local
    # fixture helper; the dependency and all nonempty serialization are intact.
    def _pack_val(self, val, vtype, add_vtype, sub_type=None):
        if vtype == GGUFValueType.ARRAY and sub_type == GGUFValueType.STRING and len(val) == 0:
            prefix = self._pack("I", vtype) if add_vtype else b""
            return prefix + self._pack("I", GGUFValueType.STRING) + self._pack("Q", 0)
        return super()._pack_val(val, vtype, add_vtype, sub_type)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def byte_vocabulary():
    """GPT-2's reversible byte alphabet, with fixture token ID b for byte b."""
    visible = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    codepoints = dict(zip(visible, visible))
    extra = 256
    for value in range(256):
        if value not in codepoints:
            codepoints[value] = extra
            extra += 1
    return [chr(codepoints[value]) for value in range(256)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True, help="Pin the synthetic GGUF input bytes")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--byte-tokenizer", action="store_true",
                        help="Separate HTTP-only fixture: map 256 tokens to GPT-2 byte alphabet without changing model tensors")
    args = parser.parse_args()
    if args.output_dir.exists(): parser.error("Output must be fresh; preserve prior fixture evidence")
    source_hash = sha(args.source)
    if source_hash != args.source_sha256:
        parser.error("Unexpected source fixture")
    manifest = json.loads((args.source.parent / "deepseek41.config.json").read_text())
    if not manifest.get("fixture"):
        parser.error("Only generated synthetic fixtures are supported")
    source = GGUFReader(args.source)
    if "deepseek41.engram.token_map" not in source.fields:
        parser.error("Regenerate the fixture with embedded Engram metadata")
    assert "tokenizer.ggml.merges" not in source.fields
    byte_tokens = byte_vocabulary() if args.byte_tokenizer else None
    if byte_tokens is not None and len(source.fields["tokenizer.ggml.tokens"].contents()) != 256:
        parser.error("The HTTP byte tokenizer requires exactly 256 source vocabulary rows")
    args.output_dir.mkdir(parents=True)
    output = args.output_dir / args.source.name
    writer = ExplicitEmptyStringArrayWriter(output, source.fields["general.architecture"].contents())
    for name, field in source.fields.items():
        if name.startswith("GGUF.") or name == "general.architecture": continue
        value = byte_tokens if byte_tokens is not None and name == "tokenizer.ggml.tokens" else field.contents()
        writer.add_key_value(name, value, field.types[0],
            field.types[1] if field.types[0] == GGUFValueType.ARRAY else None)
    writer.add_key_value("tokenizer.ggml.merges", [], GGUFValueType.ARRAY, GGUFValueType.STRING)
    for tensor in source.tensors:
        writer.add_tensor(tensor.name, tensor.data, raw_dtype=tensor.tensor_type)
    writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_tensors_to_file(); writer.close()
    derived = GGUFReader(output)
    assert derived.fields["tokenizer.ggml.merges"].contents() == []
    assert len(source.tensors) == len(derived.tensors)
    tensors = []
    for old, new in zip(source.tensors, derived.tensors):
        old_hash = hashlib.sha256(old.data.tobytes()).hexdigest()
        new_hash = hashlib.sha256(new.data.tobytes()).hexdigest()
        assert old.name == new.name and list(old.shape) == list(new.shape) and old.tensor_type == new.tensor_type
        assert old_hash == new_hash
        tensors.append(dict(name=old.name, shape=[int(n) for n in old.shape], dtype=old.tensor_type.name,
                            bytes=old.n_bytes, sha256=old_hash))
    sidecars = {}
    for name in ("deepseek41.config.json", "tokens.json"):
        src, dst = args.source.parent / name, args.output_dir / name
        shutil.copyfile(src, dst)
        assert sha(src) == sha(dst)
        sidecars[name] = sha(dst)
    changes = {"tokenizer.ggml.merges": []}
    if byte_tokens is not None:
        assert derived.fields["tokenizer.ggml.tokens"].contents() == byte_tokens
        changes["tokenizer.ggml.tokens"] = byte_tokens
    report = dict(schema_version=1, fixture=True, http_only_byte_tokenizer=args.byte_tokenizer,
        scope=__doc__, generator_sha256=sha(__file__),
        source=str(args.source.resolve()), source_sha256=source_hash, target=str(output.resolve()),
        target_sha256=sha(output), metadata_change=changes,
        unchanged_tensor_count=len(tensors), tensors=tensors, sidecars=sidecars)
    (args.output_dir / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("tensors", "scope")}, indent=2))


if __name__ == "__main__": main()
