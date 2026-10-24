// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Globalization;
using System.Text.RegularExpressions;
using TensorSharp.GGML;
using TensorSharp.Models;

namespace InferenceWeb.Tests;

/// <summary>
/// Drift guards for the two DeepSeek V4.1 prefill defaults, in both directions:
/// what the native loader and the managed resolver implement must be what
/// docs/env_var_feature_matrix(.md, _zh-cn.md) and the V4.1 model cards say,
/// and every value those rows promise must still exist in the code.
///
/// <list type="bullet">
/// <item><c>TS_DSV41_SPARSE_FA</c>: sparse prefill on the owned F32 CUDA path for
/// launches of more than TSG_PRECISION_DECODE_COLUMNS queries over at least
/// TSG_DSV41_SPARSE_MIN_KEYS keys, <c>0</c> restoring tiled prefill.</item>
/// <item><c>TS_DSV4_UBATCH</c>: unset on a ggml GPU backend means the loader's
/// automatic 1024/512/256 choice, logged as <c>[dsv4] prefill ubatch: N (auto; ...)</c>.</item>
/// </list>
/// </summary>
public class DeepSeek41PrefillDefaultsDocsTests
{
    private static readonly string RepoRoot = FindRepoRoot();

    private static string FindRepoRoot()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir != null && !File.Exists(Path.Combine(dir.FullName, "TensorSharp.slnx")))
            dir = dir.Parent;
        return dir?.FullName;
    }

    private static string Read(params string[] parts) => File.ReadAllText(Path.Combine([RepoRoot, .. parts]));

    private static string Row(string markdown, string variable, string file)
    {
        string[] rows = markdown.Split('\n').Where(line => line.StartsWith($"| `{variable}` |", StringComparison.Ordinal)).ToArray();
        Assert.True(rows.Length == 1, $"{file} must document {variable} in exactly one row; found {rows.Length}.");
        return rows[0];
    }

    private static long NativeConstant(string source, string name)
    {
        Match match = Regex.Match(source, $@"\b{name}\s*=\s*(\d+)\s*;");
        Assert.True(match.Success, $"native constant {name} not found");
        return long.Parse(match.Groups[1].Value, CultureInfo.InvariantCulture);
    }

    [Fact]
    public void SparsePrefillGate_CodeAndDocsAgree()
    {
        if (RepoRoot is null) return;   // running outside a source checkout
        string policy = Read("TensorSharp.GGML.Native", "ggml_ops_precision_policy.h");
        string loader = Read("TensorSharp.GGML.Native", "ggml_ops_deepseek4.cpp");
        long queries = NativeConstant(policy, "TSG_PRECISION_DECODE_COLUMNS");
        long keys = NativeConstant(policy, "TSG_DSV41_SPARSE_MIN_KEYS");
        Assert.Equal(8, queries);
        Assert.Equal(2048, keys);
        // accepted -> documented: the attention dispatch and the banner both go through the gate.
        Assert.Contains("inline int tsg_dsv41_owned_sparse_capacity(", policy);
        Assert.Contains("std::atoi(sparse_env) == 0", policy);
        Assert.True(Regex.Matches(loader, @"tsg_dsv41_owned_sparse_capacity\(").Count >= 2,
            "attn_mha and the V4.1 precision banner must share the gate");
        Assert.Contains("getenv(\"TS_DSV41_SPARSE_FA\")", loader);
        Assert.Contains("sparse prefill for queries>" + queries.ToString(CultureInfo.InvariantCulture) + ", keys>=" +
                        keys.ToString(CultureInfo.InvariantCulture), loader);

        // documented -> accepted
        string keysText = keys.ToString("N0", CultureInfo.InvariantCulture);
        string en = Row(Read("docs", "env_var_feature_matrix.md"), "TS_DSV41_SPARSE_FA", "env_var_feature_matrix.md");
        Assert.Contains($"more than {queries} queries", en);
        Assert.Contains($"at least {keysText} keys", en);
        Assert.Contains("on by default", en);
        Assert.Contains("`0` restores tiled prefill", en);
        Assert.Contains("16,384", en);
        string zh = Row(Read("docs", "env_var_feature_matrix_zh-cn.md"), "TS_DSV41_SPARSE_FA", "env_var_feature_matrix_zh-cn.md");
        Assert.Contains($"超过 {queries} 个 query", zh);
        Assert.Contains($"至少 {keysText} 个 key", zh);
        Assert.Contains("默认开启", zh);
        Assert.Contains("`0` 恢复分块 prefill", zh);
        Assert.Contains("16,384", zh);
        foreach (string card in new[] { "deepseek41.md", "deepseek41_zh-cn.md" })
        {
            string text = Read("docs", "models", card);
            Assert.Contains("TS_DSV41_SPARSE_FA=0", text);
            Assert.Contains(keysText, text);
            Assert.Contains("--benchmark-dsv41-prefill", text);
        }
        Assert.Contains("--benchmark-dsv41-prefill", Read("TensorSharp.GGML.Native", "tests", "attention_precision_test.cpp"));
    }

    [Fact]
    public void YocoBoundedReplay_DefaultOn_CodeAndDocsAgree()
    {
        if (RepoRoot is null) return;
        string policy = Read("TensorSharp.GGML.Native", "dsv41_yoco.h");
        string loader = Read("TensorSharp.GGML.Native", "ggml_ops_deepseek4.cpp");
        long keep = NativeConstant(policy, "default_keep");
        Assert.Equal(140, keep);
        Assert.Contains("return n_swa + 8;", policy);
        // accepted -> documented: one spelling, every banner state.
        Assert.Equal("TS_DSV41_YOCO_KEEP", DeepSeek41Architecture.YocoKeepVariable);
        Assert.Contains("tsg_dsv41_yoco::parse(getenv(\"TS_DSV41_YOCO_KEEP\"), hp.n_swa)", loader);
        Assert.Contains("YOCO bounded replay%s: layers %d..%d prefill only the last %lld positions", loader);
        Assert.Contains("\" (default)\"", loader);
        Assert.Contains("YOCO bounded replay: off (TS_DSV41_YOCO_KEEP=0)", loader);
        Assert.Contains("YOCO bounded replay: off for this checkpoint (%s)", loader);
        Assert.Contains("YOCO bounded replay: inactive with a 1-token prefill micro-batch", loader);

        // documented -> accepted
        string en = Row(Read("docs", "env_var_feature_matrix.md"), "TS_DSV41_YOCO_KEEP", "env_var_feature_matrix.md");
        string zh = Row(Read("docs", "env_var_feature_matrix_zh-cn.md"), "TS_DSV41_YOCO_KEEP", "env_var_feature_matrix_zh-cn.md");
        Assert.Contains("On by default", en);
        Assert.Contains("| `140` (on) |", en);
        Assert.Contains("`0` runs every layer over every position", en);
        Assert.Contains("at least the sliding window + 8 (136)", en);
        Assert.Contains("--backend cuda", en);
        Assert.Contains("TS_DSV4_UBATCH=1", en);
        Assert.Contains("默认开启", zh);
        Assert.Contains("| `140`（开） |", zh);
        Assert.Contains("`0` 让每一层处理每个位置", zh);
        Assert.Contains("136", zh);
        Assert.Contains("--backend cuda", zh);
        Assert.Contains("TS_DSV4_UBATCH=1", zh);
        foreach (string card in new[] { "deepseek41.md", "deepseek41_zh-cn.md" })
        {
            string text = Read("docs", "models", card);
            Assert.Contains("TS_DSV41_YOCO_KEEP=0", text);
            Assert.Contains("140", text);
        }
    }

    [Fact]
    public void AutomaticPrefillWidth_CodeAndDocsAgree()
    {
        if (RepoRoot is null) return;
        string plan = Read("TensorSharp.GGML.Native", "dsv4_ubatch_plan.h");
        string loader = Read("TensorSharp.GGML.Native", "ggml_ops_deepseek4.cpp");
        Match candidates = Regex.Match(plan, @"AUTO_UBATCH_CANDIDATES\[\]\s*=\s*\{([^}]*)\}");
        Assert.True(candidates.Success, "dsv4_ubatch_plan.h must declare AUTO_UBATCH_CANDIDATES");
        int[] widths = candidates.Groups[1].Value.Split(',').Select(v => int.Parse(v.Trim(), CultureInfo.InvariantCulture)).ToArray();
        Assert.Equal(new[] { 1024, 512, 256 }, widths);
        Assert.Contains($"constexpr int UBATCH_AUTO = {GgmlDeepSeek4Native.UBatchAuto};", plan);
        Assert.Contains("prefill ubatch: %d (auto; %s)", loader);
        Assert.Contains("TSG_EXPORT int TSGgml_Dsv4UBatch(void * handle)", loader);
        Assert.Contains("Re-run with --n-cpu-moe %d", plan);

        // The managed default the rows describe.
        Assert.Equal(GgmlDeepSeek4Native.UBatchAuto, DeepSeek4Model.ResolveNativeUbatch(true, BackendType.GgmlCuda, null, out _));
        Assert.Equal(256, DeepSeek4Model.ResolveNativeUbatch(true, BackendType.GgmlCpu, null, out _));
        Assert.Equal(256, DeepSeek4Model.ResolveNativeUbatch(true, BackendType.Cpu, null, out _));
        Assert.Equal(256, DeepSeek4Model.ResolveNativeUbatch(true, BackendType.Cuda, null, out _));
        Assert.Equal(1024, DeepSeek4Model.ResolveNativeUbatch(false, BackendType.GgmlCuda, null, out _));
        Assert.Equal(512, DeepSeek4Model.ResolveNativeUbatch(false, BackendType.Cpu, null, out _));

        string en = Row(Read("docs", "env_var_feature_matrix.md"), "TS_DSV4_UBATCH", "env_var_feature_matrix.md");
        string zh = Row(Read("docs", "env_var_feature_matrix_zh-cn.md"), "TS_DSV4_UBATCH", "env_var_feature_matrix_zh-cn.md");
        Assert.Contains($"{widths[0]}, {widths[1]} or {widths[2]}", en);
        Assert.Contains($"{widths[0]}、{widths[1]}、{widths[2]}", zh);
        foreach (string row in new[] { en, zh })
        {
            Assert.Contains("[dsv4] prefill ubatch: N (auto; ...)", row);
            Assert.Contains("`256`", row);
            Assert.Contains("`1024`", row);
            Assert.Contains("`512`", row);
        }
        Assert.Contains("V4.1: auto on ggml GPU backends", en);
        Assert.Contains("used verbatim", en);
        Assert.Contains("ggml GPU 后端上自动", zh);
        Assert.Contains("原样使用", zh);
        foreach (string card in new[] { "deepseek41.md", "deepseek41_zh-cn.md" })
        {
            string text = Read("docs", "models", card);
            Assert.Contains("[dsv4] prefill ubatch: N (auto; ...)", text);
            Assert.Contains("TS_DSV4_UBATCH=256", text);
            Assert.Contains("TSGgml_Dsv4UBatch", text);
            Assert.Contains("Re-run with --n-cpu-moe N", text);
            Assert.Contains("2,318 / 2,588 / 3,128", text);
        }
    }
}
