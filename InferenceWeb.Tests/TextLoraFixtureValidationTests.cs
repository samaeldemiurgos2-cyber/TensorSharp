// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text.Json;
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Models.Lora;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>
/// Validates a fixture written by <c>eng/text-lora-fixture.py</c> on a real (or the script's
/// self-test) base GGUF: the GGUF adapter and the safetensors adapter, applied at run time,
/// must give the logits of the merged-weights GGUF through a prompt prefill and greedy decode.
/// Runs only with <c>TS_TEXT_LORA_FIXTURE_DIR</c> (the script's output directory);
/// <c>TS_TEXT_LORA_FIXTURE_BACKEND</c> picks the backend (default cpu; a GGML backend must be the
/// one this process pins), <c>TS_TEXT_LORA_FIXTURE_TOLERANCE</c> the allowed max |difference|
/// relative to the largest merged logit (default 1e-3, for an F32 base, where the merged GGUF is
/// an exact oracle: the script's <c>--self-test-f32</c> fixture matches to ~5e-7). A quantized
/// base differs from its F32 merged twin by its own matmuls' rounding (Q8_0 rounds the
/// activations too): about 2.4e-3 on the script's mixed F16/Q8_0 self-test base, more on Q4
/// checkpoints, so raise the tolerance there and read the reported top-1 agreement.
/// </summary>
public sealed class TextLoraFixtureValidationTests(ITestOutputHelper output)
{
    private const string DirVar = "TS_TEXT_LORA_FIXTURE_DIR";

    [ModelFact(DirVar)]
    public void FixtureAdapters_MatchTheMergedWeights()
    {
        string dir = Environment.GetEnvironmentVariable(DirVar)!;
        using var manifest = JsonDocument.Parse(File.ReadAllText(Path.Combine(dir, "fixture.json")));
        var root = manifest.RootElement;
        string basePath = root.GetProperty("base").GetString()!;
        if (!Path.IsPathRooted(basePath)) basePath = Path.Combine(dir, basePath);
        string merged = Path.Combine(dir, root.GetProperty("merged").GetString()!);
        BackendType backend = Enum.Parse<BackendType>(
            (Environment.GetEnvironmentVariable("TS_TEXT_LORA_FIXTURE_BACKEND") ?? "cpu").Replace("_", ""), ignoreCase: true);
        string? pinSkip = TestGates.GgmlPinSkip(backend);
        Assert.True(pinSkip == null, pinSkip);
        float tolerance = float.Parse(Environment.GetEnvironmentVariable("TS_TEXT_LORA_FIXTURE_TOLERANCE") ?? "1e-3",
            CultureInfo.InvariantCulture);

        const int decode = 16;
        int[] prompt;
        float[][] expected;
        int[] tokens;
        using (var twin = ModelBase.Create(merged, backend))
        {
            prompt = twin.Tokenizer.Encode("Explain low-rank adaptation in one sentence.", addSpecial: false).ToArray();
            twin.ResetKVCache();
            expected = new float[decode + 1][];
            tokens = new int[decode];
            expected[0] = (float[])twin.ForwardRefill(prompt).Clone();
            for (int i = 0; i < decode; i++)
            {
                tokens[i] = ArgMax(expected[i]);
                expected[i + 1] = (float[])twin.Forward(new[] { tokens[i] }).Clone();
            }
        }

        using var model = ModelBase.Create(basePath, backend);
        model.ResetKVCache();
        float[] baseLogits = (float[])model.ForwardRefill(prompt).Clone();
        foreach (string key in new[] { "adapter_gguf", "adapter_safetensors" })
        {
            var adapter = TextLoraLoader.Load(key, Path.Combine(dir, root.GetProperty(key).GetString()!));
            var plan = model.PlanTextLora(adapter);
            Assert.True(plan.IsApplicable, plan.RefusalMessage);
            Assert.True(model.TryBindAdapter(adapter, out string? error), error);
            model.SelectStepAdapter(adapter);
            try
            {
                model.ResetKVCache();
                float worst = 0, scale = 0;
                int agree = 0;
                float[] logits = (float[])model.ForwardRefill(prompt).Clone();
                for (int i = 0; i <= decode; i++)
                {
                    scale = Math.Max(scale, expected[i].Max(Math.Abs));
                    worst = Math.Max(worst, expected[i].Zip(logits, (e, a) => Math.Abs(e - a)).Max());
                    if (ArgMax(logits) == ArgMax(expected[i])) agree++;
                    if (i < decode) logits = (float[])model.Forward(new[] { tokens[i] }).Clone();
                }
                float moved = baseLogits.Zip(expected[0], (b, e) => Math.Abs(b - e)).Max();
                output.WriteLine($"{key} on {backend}: {plan.AppliedTargets.Count} targets on {plan.AppliedWeights.Count} weights; " +
                                 $"max |lora - merged| = {worst:G4} (largest logit {scale:G4}), top-1 agreement {agree}/{decode + 1}, " +
                                 $"base vs merged moved {moved:G4}");
                Assert.True(worst <= tolerance * scale, $"{key}: max |lora - merged| {worst} exceeds {tolerance} x {scale}");
                Assert.True(moved > 10 * tolerance * scale, "the adapter must move the logits well past the tolerance");
            }
            finally
            {
                model.SelectStepAdapter(null);
                model.ReleaseAdapter(adapter);
            }
        }
    }

    private static int ArgMax(float[] logits)
    {
        int best = 0;
        for (int v = 1; v < logits.Length; v++) if (logits[v] > logits[best]) best = v;
        return best;
    }
}
