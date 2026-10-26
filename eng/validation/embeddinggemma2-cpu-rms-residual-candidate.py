#!/usr/bin/env python3
"""Prepare an isolated whole-encoder RMS+gain+ADD candidate; never compile it."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BUILDER = HERE / "embeddinggemma2-cpu-rms-arm-candidate.py"
BUILDER_SHA = "5c0cff9cbae38eb2ba944fd458c1f7d424fa15cd40595373edfe34b366adebb2"
COMPONENT = HERE / "embeddinggemma2-cpu-rms-residual-probe.cpp"
COMPONENT_SHA = "37516bef33eb7b063af197cd6e77854e5eb9b0a26f5693c9c8b6379492bdc23f"
SOURCE_SHA = "a0a18b3f3fcc11b3ac7094bd87c72e8a4a7938351a82f725b3e23f0df9df62d1"
NATIVE_SHA = "936168c6ab4036fbb0692dfc0a5bb94cc0b23e2f2ebdc2575de1600c2a7bf5dc"
REVISION = "3612c0af93f2cfc8f048295300a61187723b12ae"
SCOPE = ("Isolated Apple ARM validated Gemma2 CPU all-F32 width512 two-dimensional "
         "RMS+gain+residual ADD at seven/eight workers; only three norm/add sites, "
         "rows1..8192/shared gain; no production build, GPU, scale or GELU change.")
BEGIN = "// BEGIN STAGED CPU RMS GAIN RESIDUAL"
END = "// END STAGED CPU RMS GAIN RESIDUAL"
CALLS = {
    '            x = ggml_add(ctx, rms(ctx, x, p + "post_attention_norm"), residual);':
        '            x = rms_residual(g, x, residual, p + "post_attention_norm");',
    '            x = ggml_add(ctx, rms(ctx, linear(ctx, x, p + "ffn_down"), p + "post_ffw_norm"), residual);':
        '            x = rms_residual(g, linear(ctx, x, p + "ffn_down"), residual, p + "post_ffw_norm");',
    '            x = ggml_add(ctx, rms(ctx, linear(ctx, x, p + "proj"), p + "post_norm"), residual);':
        '            x = rms_residual(g, linear(ctx, x, p + "proj"), residual, p + "post_norm");',
}
PRAGMA = """#if defined(__clang__)
#pragma clang fp contract(off)
#endif
"""
ATTRIBUTE = """#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("fp-contract=off")))
#endif
"""
FACTORY = r"""
bool staged_rms_residual_layout(const ggml_tensor * value) {
    if (!value || value->type != GGML_TYPE_F32 || value->ne[0] != 512 ||
        value->ne[1] < 1 || value->ne[1] > 8192 || value->ne[2] != 1 || value->ne[3] != 1 ||
        value->nb[0] != sizeof(float) || value->nb[1] < 512 * sizeof(float) ||
        value->nb[1] % sizeof(float) != 0) return false;
    return uint64_t(value->ne[1] - 1) <= (size_t(-1) - 512 * sizeof(float)) / value->nb[1];
}
void staged_rms_residual_callback(ggml_tensor * output, const ggml_tensor * input,
        const ggml_tensor * gain, const ggml_tensor * residual, int ith, int nth, void * userdata) {
    const float eps = *static_cast<const float *>(userdata);
    for (int64_t token = ith; token < input->ne[1]; token += nth) {
        staged_rms_residual_row(
            reinterpret_cast<float *>(static_cast<char *>(output->data) + token * output->nb[1]),
            reinterpret_cast<const float *>(static_cast<const char *>(input->data) + token * input->nb[1]),
            static_cast<const float *>(gain->data),
            reinterpret_cast<const float *>(static_cast<const char *>(residual->data) + token * residual->nb[1]), eps);
    }
}
ggml_tensor * staged_rms_residual(ggml_context * ctx, ggml_tensor * input,
        ggml_tensor * gain, ggml_tensor * residual, const float * eps) {
    if (!eps || !std::isfinite(*eps) || *eps <= 0 || !staged_rms_residual_layout(input) ||
        !staged_rms_residual_layout(gain) || !staged_rms_residual_layout(residual) ||
        gain->ne[1] != 1 || residual->ne[1] != input->ne[1]) return nullptr;
    // The encoder owns immutable eps until both cached graph plans are freed.
    // No descriptor/input/output pointer cache and no callback instrumentation.
    return ggml_map_custom3(ctx, input, gain, residual, staged_rms_residual_callback,
        GGML_N_TASKS_MAX, const_cast<float *>(eps));
}
"""
METHOD = r"""    ggml_tensor * rms_residual(graph & g, ggml_tensor * input, ggml_tensor * residual,
                               const std::string & name) const {
        auto * ctx = g.ctx;
        auto * result = cpu_fused_rms_residual ? staged_rms_residual(ctx, input,
            weight(name + ".weight", 512, 1), residual, &eps) : nullptr;
        if (!result) return ggml_add(ctx, rms(ctx, input, name), residual);
        ++g.rms_residual_nodes; // Graph-build diagnostic only; never a timed worker counter.
        ggml_set_name(result, (name + ".cpu_rms_residual").c_str());
        return result;
    }
"""
ADMISSION = r"""#if defined(__APPLE__) && defined(__aarch64__)
    // Initial whole-encoder screen is limited to the component-qualified platform.
    // Other CPU platforms, GPU, nonvalidated geometry and mixed precision keep the original chain.
    const int rms_residual_workers = threads > 0 ? threads : 4;
    e->cpu_fused_rms_residual = validated_cpu_geometry && e->cpu_float_linears &&
        (rms_residual_workers == 7 || rms_residual_workers == 8);
#endif
"""
PROFILE = r"""                std::fprintf(stderr, "EMBEDDING_CPU_RMS_RESIDUAL_PLAN nodes=%d upper_bound=%d rows_limit=8192\n",
                    g->rms_residual_nodes, cpu_fused_rms_residual ? 3 * layers : 0);
"""


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def component_row(text):
    require(text.count("void row(float * y,") == 1, "Require the reviewed component row function")
    start = text.index("void row(float * y,")
    end = text.index("// Independent real-valued", start)
    result = text[start:end]
    require(result.endswith("}\n"), "Component row boundary changed")
    return result


def injected_row(text):
    original = component_row(text)
    renamed = original.replace("void row(", "void staged_rms_residual_row(", 1).replace("width", "staged_rms_residual_width")
    return ATTRIBUTE + renamed.replace("{\n", "{\n" + PRAGMA, 1)


def transform(source, component):
    require(hashlib.sha256(source.encode("utf-8")).hexdigest() == SOURCE_SHA and
            hashlib.sha256(component.encode("utf-8")).hexdigest() == COMPONENT_SHA,
            "Transform requires the exact reviewed production/component bytes")
    require(BEGIN not in source and END not in source and "cpu_fused_rms_residual" not in source,
            "Preserve existing staged sources; transformation is not repeatable")
    anchors = ["struct source_file {\n", "    bool cpu_flat_qkv_rms = false;\n", "        bool packed = false;\n",
               "    void validate() const {\n", "    e->weights_buffer = ggml_backend_alloc_ctx_tensors(e->weights_ctx, e->backend);\n",
               "            if (cpu_profile_plan::enabled()) {\n", *CALLS]
    require(all(source.count(anchor) == 1 for anchor in anchors), "Every owned insertion/callsite anchor must occur exactly once")
    require(source.index("e->cpu_float_linears = std::all_of") < source.index("    e->weights_buffer ="),
            "Final F32 storage validation must precede admission")
    require(source.index("cached.reset();") < source.index("if (cpu_attention_scratch) std::fprintf"),
            "Encoder graph plans must be destroyed before immutable epsilon ownership ends")
    helper = BEGIN + "\nconstexpr int staged_rms_residual_width = 512;\n" + injected_row(component) + FACTORY + END + "\n"
    output = source.replace("struct source_file {\n", helper + "struct source_file {\n", 1)
    output = output.replace("    bool cpu_flat_qkv_rms = false;\n", "    bool cpu_flat_qkv_rms = false;\n    bool cpu_fused_rms_residual = false;\n", 1)
    output = output.replace("        bool packed = false;\n", "        bool packed = false;\n        int rms_residual_nodes = 0;\n", 1)
    output = output.replace("    void validate() const {\n", METHOD + "    void validate() const {\n", 1)
    output = output.replace("    e->weights_buffer = ggml_backend_alloc_ctx_tensors(e->weights_ctx, e->backend);\n",
                            ADMISSION + "    e->weights_buffer = ggml_backend_alloc_ctx_tensors(e->weights_ctx, e->backend);\n", 1)
    output = output.replace("            if (cpu_profile_plan::enabled()) {\n", "            if (cpu_profile_plan::enabled()) {\n" + PROFILE, 1)
    for before, after in CALLS.items():
        output = output.replace(before, after, 1)
    require(all(before not in output and output.count(after) == 1 for before, after in CALLS.items()), "Three-callsite contract differs")
    require(output.count(BEGIN) == output.count(END) == 1, "Injected block boundaries differ")
    return output


def builder():
    require(sha(BUILDER) == BUILDER_SHA, "Reviewed Mac builder changed")
    spec = importlib.util.spec_from_file_location("rms_residual_mac_builder", BUILDER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=ROOT / "TensorSharp.GGML.Native/ggml_ops_embeddings.cpp")
    p.add_argument("--native-library", type=Path, default=ROOT / "TensorSharp.GGML.Native/build/libGgmlOps.dylib")
    p.add_argument("--build-dir", type=Path, default=ROOT / "TensorSharp.GGML.Native/build")
    p.add_argument("--ggml-dir", type=Path, default=ROOT / "ExternalProjects/ggml")
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--stage", action="store_true", help="Generate private sources/reference/guarded build recipes only; never compile")
    args = p.parse_args(argv)
    pinned = {args.source.absolute(): SOURCE_SHA, args.native_library.absolute(): NATIVE_SHA,
              COMPONENT: COMPONENT_SHA, BUILDER: BUILDER_SHA, Path(__file__).absolute(): sha(__file__),
              HERE / "embeddinggemma2-cpu-rms-residual-candidate.md": sha(HERE / "embeddinggemma2-cpu-rms-residual-candidate.md")}
    require(all(sha(path) == digest for path, digest in pinned.items()), "Known source/component/native/builder identity differs")
    raw_source = args.source.read_bytes(); raw_component = COMPONENT.read_bytes()
    require(hashlib.sha256(raw_source).hexdigest() == SOURCE_SHA and hashlib.sha256(raw_component).hexdigest() == COMPONENT_SHA,
            "Parsed source/component bytes changed")
    prepared = transform(raw_source.decode("utf-8"), raw_component.decode("utf-8"))
    if not args.stage:
        print(json.dumps({"source_only": True, "staged": False, "compiled": False,
                          "source_sha256": SOURCE_SHA, "candidate_source_sha256": hashlib.sha256(prepared.encode()).hexdigest(),
                          "planned_callsite_nodes_upper_bound": 72, "scope": SCOPE}, indent=2))
        return
    require(args.output_dir is not None, "Stage requires a fresh ignored output directory")
    args.execute = False  # There is deliberately no --execute or native build option in this tool.
    args.expected_source_sha256 = SOURCE_SHA; args.expected_native_sha256 = NATIVE_SHA; args.expected_ggml_revision = REVISION
    report = builder().build_candidate(args, source_transform=lambda text: transform(text, raw_component.decode("utf-8")),
                                       scope=SCOPE, additional_guard_paths=pinned)
    require(report["completed"] and not report["built"] and not report["runs"], "Source-only staging must never compile")
    require(all(report["before"][str(path)]["sha256"] == digest and sha(path) == digest for path, digest in pinned.items()),
            "Source transform/component binding changed or was rebased")
    require(report["staged_sources"]["candidate"]["sha256"] == hashlib.sha256(prepared.encode()).hexdigest(), "Staged candidate bytes differ")
    report["planned_callsite_nodes_upper_bound"] = 72
    report["actual_native_dispatch_observed"] = False
    report["component_source_sha256"] = COMPONENT_SHA
    report["source_transform_guarded_inputs"] = {str(path): digest for path, digest in pinned.items()}
    report["limitations"][1] = "Both future rebuilds share captured nonencoder objects/archives/flags. Rebuilt baseline must equal the supplied native936 or be separately qualified before candidate comparisons."
    (args.output_dir / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("completed", "built", "scope", "staged_sources", "planned_callsite_nodes_upper_bound")}, indent=2))


if __name__ == "__main__":
    main()
