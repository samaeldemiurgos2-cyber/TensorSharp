// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Text.Json.Nodes;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Models.Lora;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Speculative;
using TensorSharp.Server;

namespace InferenceWeb.Tests;

/// <summary>
/// The model side of per-request text LoRA (PR #282 review items 1, 7 and 11): one
/// <see cref="TextLoraBinding"/> per (model, adapter), materialized once; the
/// <c>LinearForward</c> hook applies only the adapter selected for the step; an adapter the
/// model cannot apply in full is refused, never applied in part; reload and unload retire;
/// and <see cref="ModelService"/> resolves a request's adapter without touching any other
/// request. Runs on the pure-C# cpu backend over a probe GGUF.
/// </summary>
public sealed class TextLoraBindingTests : IDisposable
{
    private readonly string _dir = Directory.CreateTempSubdirectory("ts-lora-binding-").FullName;
    private readonly string _gguf;

    public TextLoraBindingTests()
    {
        _gguf = Path.Combine(_dir, "probe.gguf");
        BackendFailureWarmupTests.WriteProbeGguf(_gguf);
    }

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch { /* best effort */ }
    }

    // ------------------------------------------------------------------ LinearForward hook

    [Fact]
    public void LinearForward_AppliesOnlyTheSelectedAdapter_MatchingANaiveReference()
    {
        const int inDim = 3, outDim = 4, rank = 2, rows = 2;
        float[] w = { 0.5f, -1f, 2f, 1f, 0f, -0.5f, 0.25f, 3f, -2f, 1.5f, 0.75f, -1f };       // [out, in]
        float[] a = { 1f, -2f, 0.5f, 0.25f, 1f, -1f };                                       // [rank, in]
        float[] b = { 2f, -1f, 0.5f, 1f, -1.5f, 0.25f, 1f, 3f };                             // [out, rank]
        float[] x = { 1f, -0.5f, 2f, 0.25f, 1.5f, -1f };                                     // [rows, in]
        const float alpha = 6f;
        float scale = alpha / rank;

        using var model = new LoraProbeModel(_gguf, w, outDim, inDim);
        var adapter = Adapter("canary", (LoraProbeModel.Target, a, b, rank, inDim, outDim, scale));
        float[] baseBefore = model.Linear(x, rows);

        Assert.True(model.TryBindAdapter(adapter, out string? error), error);
        float[] baseWhileBound = model.Linear(x, rows);
        model.SelectStepAdapter(adapter);
        float[] adapted = model.Linear(x, rows);
        model.SelectStepAdapter(null);
        float[] baseAfter = model.Linear(x, rows);

        // y = x W^T + s (x A^T) B^T, from the ORIGINAL A and B with nested loops.
        var expected = new float[rows * outDim];
        for (int r = 0; r < rows; r++)
        {
            var h = new float[rank];
            for (int j = 0; j < rank; j++)
                for (int i = 0; i < inDim; i++)
                    h[j] += x[r * inDim + i] * a[j * inDim + i];
            for (int o = 0; o < outDim; o++)
            {
                float acc = 0;
                for (int i = 0; i < inDim; i++)
                    acc += x[r * inDim + i] * w[o * inDim + i];
                for (int j = 0; j < rank; j++)
                    acc += scale * h[j] * b[o * rank + j];
                expected[r * outDim + o] = acc;
            }
        }
        for (int k = 0; k < expected.Length; k++)
            Assert.Equal(expected[k], adapted[k], 4);

        // Bound but not selected is exactly the base model, bit for bit.
        Assert.Equal(baseBefore, baseWhileBound);
        Assert.Equal(baseBefore, baseAfter);
        Assert.NotEqual(baseBefore, adapted);
    }

    // ------------------------------------------------------------------ binding lifecycle

    [Fact]
    public void Binding_IsPlannedAndMaterializedOnce_PerModelAndAdapter()
    {
        using var model = new LoraProbeModel(_gguf, new float[12], 4, 3);
        var adapter = Adapter("once", (LoraProbeModel.Target, new float[6], new float[8], 2, 3, 4, 1f));

        Assert.True(model.TryBindAdapter(adapter, out _));
        TextLoraBinding binding = model.PlanTextLora(adapter);
        Assert.True(binding.TryGetFactors(LoraProbeModel.Target, out var factors));
        // Bound is not selected: base requests keep every fused fast path.
        Assert.False(model.LoraActiveForStep);

        for (int i = 0; i < 10; i++)
        {
            Assert.True(model.TryBindAdapter(adapter, out _));
            model.SelectStepAdapter(adapter);
            Assert.Same(adapter, model.StepAdapter);
            Assert.True(model.LoraActiveForStep);
            model.SelectStepAdapter(null);
            Assert.Null(model.StepAdapter);
            Assert.False(model.LoraActiveForStep);
        }

        // Same plan (its log line printed once), same tensors (uploaded once).
        Assert.Same(binding, model.PlanTextLora(adapter));
        Assert.True(binding.TryGetFactors(LoraProbeModel.Target, out var again));
        Assert.Same(factors.AT, again.AT);
        Assert.Same(factors.BT, again.BT);
        Assert.Equal(new IRequestAdapter[] { adapter }, model.BoundAdapters);
    }

    [Fact]
    public void ReleaseAdapter_RefusesWhileSelected_ThenDisposesTheFactors()
    {
        using var model = new LoraProbeModel(_gguf, new float[12], 4, 3);
        var adapter = Adapter("release", (LoraProbeModel.Target, new float[6], new float[8], 2, 3, 4, 1f));
        Assert.True(model.TryBindAdapter(adapter, out _));
        var binding = model.PlanTextLora(adapter);

        model.SelectStepAdapter(adapter);
        Assert.Throws<InvalidOperationException>(() => model.ReleaseAdapter(adapter));
        model.SelectStepAdapter(null);

        model.ReleaseAdapter(adapter);
        Assert.False(binding.IsMaterialized);
        Assert.Empty(model.BoundAdapters);
        Assert.Throws<InvalidOperationException>(() => model.SelectStepAdapter(adapter));
    }

    [Fact]
    public void YieldGpuComputeLock_NeverLetsAnotherUserInWhileAnAdapterIsSelected()
    {
        using var model = new LoraProbeModel(_gguf, new float[12], 4, 3);
        var adapter = Adapter("yield", (LoraProbeModel.Target, new float[6], new float[8], 2, 3, 4, 1f));
        Assert.True(model.TryBindAdapter(adapter, out _));
        IRequestAdapter? seenByOther = adapter;
        Thread other;
        lock (model.GpuComputeLock)
        {
            model.SelectStepAdapter(adapter);
            other = new Thread(() => { lock (model.GpuComputeLock) seenByOther = model.StepAdapter; });
            other.Start();
            Thread.Sleep(50);                    // the other user is now waiting for the lock
            for (int i = 0; i < 20; i++)
                model.YieldGpuComputeLock();     // an encoder block boundary inside a step
            model.SelectStepAdapter(null);
        }
        Assert.True(other.Join(TimeSpan.FromSeconds(10)));
        Assert.Null(seenByOther);
    }

    [Fact]
    public void ModelDispose_ReleasesEveryBinding()
    {
        var model = new LoraProbeModel(_gguf, new float[12], 4, 3);
        var one = Adapter("one", (LoraProbeModel.Target, new float[6], new float[8], 2, 3, 4, 1f));
        var two = Adapter("two", (LoraProbeModel.Target, new float[6], new float[8], 2, 3, 4, 1f));
        Assert.True(model.TryBindAdapter(one, out _));
        Assert.True(model.TryBindAdapter(two, out _));
        var bindings = new[] { model.PlanTextLora(one), model.PlanTextLora(two) };

        model.Dispose();

        Assert.All(bindings, b => Assert.False(b.IsMaterialized));
        Assert.Empty(model.BoundAdapters);
    }

    // ------------------------------------------------------------------ refusal (never partial)

    [Fact]
    public void Plan_RefusesTheWholeAdapter_NamingEveryModuleItCannotApply()
    {
        using var model = new LoraProbeModel(_gguf, new float[12], 4, 3);
        var adapter = Adapter("partial",
            (LoraProbeModel.Target, new float[6], new float[8], 2, 3, 4, 1f),                  // applies
            ("blk.0.attn_q.weight", new float[6], new float[8], 2, 3, 4, 1f),                   // fused / missing
            ("token_embd.weight", new float[6], new float[8], 2, 3, 4, 1f),                     // not a decoder projection
            ("v.blk.0.attn_qkv.weight", new float[6], new float[8], 2, 3, 4, 1f),               // vision tower
            ("blk.0.ffn_up_exps.weight", new float[6], new float[8], 2, 3, 4, 1f));             // routed experts

        TextLoraBinding plan = model.PlanTextLora(adapter);

        Assert.False(plan.IsApplicable);
        Assert.Equal(new[] { LoraProbeModel.Target }, plan.AppliedTargets);
        Assert.Equal(4, plan.Rejected.Count);
        string message = plan.RefusalMessage!;
        foreach (string module in new[] { "blk.{0}.attn_q.weight", "token_embd.weight", "v.blk.0.attn_qkv.weight", "blk.{0}.ffn_up_exps.weight" })
            Assert.Contains(module, message);
        Assert.Contains("refused rather than applied in part", message);

        Assert.False(model.TryBindAdapter(adapter, out string? error));
        Assert.Equal(message, error);
        Assert.False(plan.IsMaterialized);
        Assert.Throws<InvalidOperationException>(() => model.SelectStepAdapter(adapter));
    }

    [Fact]
    public void Plan_RefusesAShapeMismatch()
    {
        using var model = new LoraProbeModel(_gguf, new float[12], 4, 3);
        var adapter = Adapter("shape", (LoraProbeModel.Target, new float[4], new float[10], 2, 2, 5, 1f));

        var plan = model.PlanTextLora(adapter);

        Assert.False(plan.IsApplicable);
        Assert.Contains("weight is [4, 3] but the delta is [5, 2]", plan.RefusalMessage);
    }

    [Fact]
    public void Plan_RefusesANormOrBiasTarget_RatherThanThrowing()
    {
        // A crafted adapter may name any loaded tensor; a 1-D one has no [out, in] shape.
        using var model = new LoraProbeModel(_gguf, new float[12], 4, 3);
        var adapter = Adapter("norm", (LoraProbeModel.Norm, new float[3], new float[3], 1, 3, 3, 1f));

        var plan = model.PlanTextLora(adapter);

        Assert.False(plan.IsApplicable);
        var rejection = Assert.Single(plan.Rejected);
        Assert.Equal(LoraProbeModel.Norm, rejection.Target);
        Assert.Equal("not a 2-D projection weight (norms and biases take no LoRA delta)", rejection.Reason);
        Assert.False(model.TryBindAdapter(adapter, out string? error));
        Assert.Contains("blk.{0}.attn_norm.weight", error);
    }

    [Fact]
    public void Plan_RefusesEveryAdapter_OnAFamilyWhoseProjectionsBypassTheHook()
    {
        using var model = new LoraProbeModel(_gguf, new float[12], 4, 3, supportsLora: false);
        var adapter = Adapter("family", (LoraProbeModel.Target, new float[6], new float[8], 2, 3, 4, 1f));

        var plan = model.PlanTextLora(adapter);

        Assert.False(plan.IsApplicable);
        Assert.NotNull(plan.ModelRefusal);
        Assert.Contains("nemotron_h, mistral3 and hunyuan-dense", plan.RefusalMessage);
        Assert.Empty(plan.AppliedTargets);
    }

    [Fact]
    public void DescribeRejections_CollapsesLayers()
    {
        var rejected = Enumerable.Range(0, 28)
            .Select(l => new TextLoraRejection($"blk.{l}.attn_q.weight", "fused"))
            .Append(new TextLoraRejection("blk.3.attn_k.weight", "fused"))
            .Append(new TextLoraRejection("blk.5.attn_k.weight", "fused"))
            .ToList();

        string text = TextLoraBinding.DescribeRejections(rejected);

        Assert.Equal("blk.{0-27}.attn_q.weight (fused); blk.{3,5}.attn_k.weight (fused)", text);
    }

    // ------------------------------------------------------------------ registry

    [Fact]
    public void Registry_ReloadAndUnload_RetireTheInstanceThatLeaves()
    {
        string path = WriteAdapter("w.safetensors", "blk.0.attn_output.weight", 1, 1, 1);
        using var registry = new TextLoraRegistry();

        var first = registry.Load("style", path);
        var second = registry.Load("style", path);

        Assert.True(first.IsRetired);
        Assert.False(second.IsRetired);
        Assert.True(second.Generation > first.Generation);
        Assert.NotEqual(first.Id, second.Id);
        Assert.Same(second, registry.Get("style"));

        Assert.True(registry.Unload("style"));
        Assert.True(second.IsRetired);
        Assert.Null(registry.Get("style"));
    }

    // ------------------------------------------------------------------ ModelService

    [Fact]
    public void Service_ResolvesNoneToTheBaseModel_AndRefusesWhatItCannotRun()
    {
        using var service = Service(supportsLora: true);
        foreach (string? none in new[] { null, "", "  ", "none", "NONE" })
        {
            Assert.True(service.TryResolveLora(none, out var adapter, out string? error));
            Assert.Null(adapter);
            Assert.Null(error);
        }

        Assert.False(service.TryResolveLora("missing", out var unknown, out string? unknownError));
        Assert.Null(unknown);
        Assert.Contains("No LoRA adapter named 'missing'", unknownError);

        using var unloaded = new ModelService();
        Assert.False(unloaded.TryResolveLora("missing", out _, out string? noModel));
        Assert.Equal("No model is loaded.", noModel);
    }

    [Fact]
    public void Service_LoadLora_BindsEagerly_AndRefusesAnAdapterTheModelCannotApply()
    {
        string good = WriteAdapter("good.safetensors", LoraProbeModel.Target, 1, 1, 1);
        string bad = WriteAdapter("bad.safetensors", "blk.0.attn_q.weight", 1, 1, 1);
        using var service = Service(supportsLora: true);

        var adapter = service.LoadLora("canary", good);
        Assert.Contains(adapter, ((IAdapterAwareModel)service.Model).BoundAdapters);
        Assert.True(service.Model.PlanTextLora(adapter).IsMaterialized);
        Assert.True(service.TryResolveLora("canary", out var resolved, out _));
        Assert.Same(adapter, resolved);

        var ex = Assert.Throws<TextLoraNotApplicableException>(() => service.LoadLora("broken", bad));
        Assert.Contains("blk.{0}.attn_q.weight", ex.Message);
        Assert.Equal(new[] { "blk.{0}.attn_q.weight" }, ex.UnsupportedModules);
        Assert.Null(service.LoraRegistry.Get("broken"));
        Assert.DoesNotContain(((IAdapterAwareModel)service.Model).BoundAdapters, a => a.Id.StartsWith("broken#", StringComparison.Ordinal));
    }

    [Fact]
    public void Service_LoadLora_ABindThatThrows_LeavesNoBindingBehind()
    {
        string path = WriteAdapter("oom.safetensors", LoraProbeModel.Target, 1, 1, 1);
        using var service = Service(supportsLora: true);
        var model = (LoraProbeModel)service.Model;
        model.BindFailure = new InvalidOperationException("out of device memory");

        var ex = Assert.Throws<InvalidOperationException>(() => service.LoadLora("oom", path));

        Assert.Same(model.BindFailure, ex);
        // The plan stored before the failure is gone: the adapter was never registered, so it
        // is never retired, and nothing else would release it.
        Assert.Empty(((IAdapterAwareModel)model).BoundAdapters);
        Assert.Null(service.LoraRegistry.Get("oom"));
    }

    /// <summary>
    /// A refused bind that fails while its unsupported modules are listed still releases the
    /// plan it stored. That listing ran outside the bind's cleanup, so a fault there (a layer
    /// number past int's range was one) left the refused adapter bound for good. The model
    /// refuses this adapter as a whole, and its modules fault when listed.
    /// </summary>
    [Fact]
    public void Service_RegisterLora_AFaultListingTheRefusedModules_LeavesNoBindingBehind()
    {
        using var service = Service(supportsLora: false);
        var fault = new InvalidOperationException("listing the modules failed");
        var adapter = new TextLoraAdapter("listed", 1, 1f, new FaultingDeltas(fault), "listed");

        var ex = Assert.Throws<InvalidOperationException>(() => service.RegisterLora(adapter));

        Assert.Same(fault, ex);
        Assert.Empty(((IAdapterAwareModel)service.Model).BoundAdapters);
        Assert.Null(service.LoraRegistry.Get("listed"));
    }

    /// <summary>One delta whose name throws the given fault whenever it is listed.</summary>
    private sealed class FaultingDeltas(Exception fault) : IReadOnlyDictionary<string, LoraDelta>
    {
        public IEnumerable<string> Keys => throw fault;
        public IEnumerable<LoraDelta> Values => throw fault;
        public int Count => 1;
        public LoraDelta this[string key] => throw new KeyNotFoundException(key);
        public bool ContainsKey(string key) => false;
        public bool TryGetValue(string key, out LoraDelta value)
        {
            value = null!;
            return false;
        }
        public IEnumerator<KeyValuePair<string, LoraDelta>> GetEnumerator() => throw fault;
        System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator() => GetEnumerator();
    }

    [Fact]
    public void Service_UnloadOrReload_WithNoEngine_ReleasesTheRetiredBinding()
    {
        string path = WriteAdapter("a.safetensors", LoraProbeModel.Target, 1, 1, 1);
        using var service = Service(supportsLora: true);
        var aware = (IAdapterAwareModel)service.Model;

        var first = service.LoadLora("style", path);
        var second = service.LoadLora("style", path);
        Assert.True(first.IsRetired);
        Assert.Equal(new IRequestAdapter[] { second }, aware.BoundAdapters);

        Assert.True(service.UnloadLora("style"));
        Assert.Empty(aware.BoundAdapters);
        Assert.False(service.UnloadLora("style"));
        Assert.False(service.TryResolveLora("style", out _, out _));
    }

    [Fact]
    public async Task Service_Unload_WhileTheEngineHostIsBusy_LeavesTheBindingToItsLastRequest()
    {
        string path = WriteAdapter("busy.safetensors", LoraProbeModel.Target, 1, 1, 1);
        using var service = Service(supportsLora: true);
        var aware = (IAdapterAwareModel)service.Model;
        var lora = service.LoadLora("style", path);

        // A request on the adapter, held before its first step.
        var gpu = new ComputeGate();
        gpu.Close();
        service.EngineHost.ComputeGate = gpu;
        var engine = service.EngineHost.TryGetEngine();
        var seq = new SequenceState("held", "user: hi".Select(c => (int)c).ToList(), 3, 16, SamplingConfig.Greedy, adapter: lora);
        var handle = engine.SubmitRequest(seq);

        // The unload lands while another thread (a status poll, a submission) holds the
        // host's gate: the engine still stands, so it must not be released here.
        object hostGate = typeof(InferenceEngineHost).GetField("_gate", BindingFlags.NonPublic | BindingFlags.Instance)!
            .GetValue(service.EngineHost)!;
        using var held = new ManualResetEventSlim();
        using var leave = new ManualResetEventSlim();
        var holder = new Thread(() => { lock (hostGate) { held.Set(); leave.Wait(); } });
        holder.Start();
        Assert.True(held.Wait(TimeSpan.FromSeconds(10)));
        try
        {
            Assert.True(service.EngineHost.ReleaseRetiredAdapters());
            Assert.True(service.UnloadLora("style"));
            Assert.Contains(lora, aware.BoundAdapters);
        }
        finally
        {
            leave.Set();
            holder.Join();
        }

        gpu.Open();
        await handle.Completion.WaitAsync(TimeSpan.FromSeconds(30));
        Assert.Equal("LLL", new string(seq.OutputTokens.Select(t => (char)t).ToArray()));
        var until = DateTime.UtcNow.AddSeconds(10);
        while (aware.BoundAdapters.Contains(lora) && DateTime.UtcNow < until)
            await Task.Delay(5);
        Assert.DoesNotContain(lora, aware.BoundAdapters);
    }

    [Fact]
    public async Task Service_ConcurrentLoraAndPlainRequests_EachGetTheirOwnModel()
    {
        string path = WriteAdapter("canary.safetensors", LoraProbeModel.Target, 1, 1, 1);
        using var service = Service(supportsLora: true, maxRunning: 4);
        var lora = service.LoadLora("canary", path);

        var tasks = Enumerable.Range(0, 8).Select(i => Generate(service, $"question {i}", i % 2 == 0 ? lora : null)).ToArray();
        string[] texts = await Task.WhenAll(tasks);

        for (int i = 0; i < texts.Length; i++)
            Assert.Equal(i % 2 == 0 ? "LLLLLL" : "BBBBBB", texts[i]);
        Assert.Null(((IAdapterAwareModel)service.Model).StepAdapter);
    }

    [Fact]
    public void SubAgentTurns_KeepTheRequestsAdapter()
    {
        var adapter = Adapter("agents", (LoraProbeModel.Target, new float[1], new float[1], 1, 1, 1, 1f));
        var root = new ChatTurnContext { Adapter = adapter, ThinkingTokenLimit = 64 };

        var child = ModelService.CreateChildTurnContext(root, publicProfiles: null);

        Assert.Same(adapter, child.Adapter);
        Assert.Null(ModelService.CreateChildTurnContext(new ChatTurnContext(), null).Adapter);
    }

    // ------------------------------------------------------------------ helpers

    private static async Task<string> Generate(ModelService service, string question, TextLoraAdapter? lora)
    {
        var text = new StringBuilder();
        await foreach (var update in service.ChatStreamWithSkillsAsync(
                           new List<ChatMessage> { new() { Role = "user", Content = question } }, 6,
                           CancellationToken.None, SamplingConfig.Greedy, tools: null, enableThinking: false,
                           skills: null, lora: lora))
        {
            if (!update.Done) text.Append(update.Piece);
        }
        return text.ToString();
    }

    internal ModelService Service(bool supportsLora, int maxRunning = 2)
    {
        var service = new ModelService(NullLogger<ModelService>.Instance,
            (path, _, _, _) => new LoraProbeModel(path, new float[1], 1, 1, supportsLora, LoraProbeModel.Template));
        service.EngineHost.SchedulerConfigOverride = new SchedulerConfig
        {
            BlockSize = 16, NumBlocks = 1024, MaxNumRunningSequences = maxRunning,
            MaxNumBatchedTokens = 1024, MaxPrefillChunkSize = 1024, SoloPrefillChunkSize = 1024,
            EnablePrefixCaching = false, Speculation = SpeculationOptions.Disabled,
        };
        service.LoadModel(_gguf, null, "cpu");
        return service;
    }

    private static TextLoraAdapter Adapter(string name,
        params (string Target, float[] A, float[] B, int Rank, int In, int Out, float Scale)[] targets)
    {
        var deltas = new Dictionary<string, LoraDelta>(StringComparer.Ordinal);
        foreach (var (target, a, b, rank, inDim, outDim, scale) in targets)
        {
            // AT = A^T [in, rank]; BT = scale * B^T [rank, out] (the loader's layout).
            var at = new float[inDim * rank];
            for (int j = 0; j < rank; j++)
                for (int i = 0; i < inDim; i++)
                    at[i * rank + j] = a[j * inDim + i];
            var bt = new float[rank * outDim];
            for (int o = 0; o < outDim; o++)
                for (int j = 0; j < rank; j++)
                    bt[j * outDim + o] = scale * b[o * rank + j];
            deltas[target] = new LoraDelta { In = inDim, Out = outDim, Rank = rank, AT = at, BT = bt };
        }
        return new TextLoraAdapter(name, targets[0].Rank, targets[0].Rank, deltas, name);
    }

    /// <summary>A rank-<paramref name="rank"/> adapter of ones on one target (scale 1).</summary>
    internal string WriteAdapter(string fileName, string target, int rank, int inDim, int outDim)
    {
        string path = Path.Combine(_dir, fileName);
        var header = new JsonObject();
        byte[] a = Ones(rank * inDim), b = Ones(outDim * rank);
        header[target + ".lora_a"] = new JsonObject
        {
            ["dtype"] = "F32", ["shape"] = new JsonArray(rank, inDim), ["data_offsets"] = new JsonArray(0, a.Length),
        };
        header[target + ".lora_b"] = new JsonObject
        {
            ["dtype"] = "F32", ["shape"] = new JsonArray(outDim, rank), ["data_offsets"] = new JsonArray(a.Length, a.Length + b.Length),
        };
        byte[] hb = Encoding.UTF8.GetBytes(header.ToJsonString());
        using var fs = File.Create(path);
        fs.Write(BitConverter.GetBytes((ulong)hb.Length));
        fs.Write(hb);
        fs.Write(a);
        fs.Write(b);
        return path;

        static byte[] Ones(int n)
        {
            var bytes = new byte[n * 4];
            Buffer.BlockCopy(Enumerable.Repeat(1f, n).ToArray(), 0, bytes, 0, bytes.Length);
            return bytes;
        }
    }
}

/// <summary>
/// A cpu-backend model with one F32 decoder projection (<see cref="Target"/>). Its forward
/// runs that projection on a vector of ones through <c>LinearForward</c> and writes 'L' when
/// the result moved off the base weight's output (an adapter changed it) and 'B' otherwise, so
/// a reply shows which model produced every token. Stateless, so engine swaps cannot disturb it.
/// </summary>
internal sealed class LoraProbeModel : ModelBase, IAdapterAwareModel
{
    public const string Target = "blk.0.attn_output.weight";
    /// <summary>A 1-D weight, loaded as a real model's norms and biases are.</summary>
    public const string Norm = "blk.0.attn_norm.weight";
    public const string Template = "{% for message in messages %}{{ message.role + ': ' + message.content + '\\n' }}{% endfor %}{{ 'assistant: ' }}";
    private readonly bool _supportsLora;
    private readonly int _in, _out;

    public LoraProbeModel(string path, float[] weight, int outDim, int inDim, bool supportsLora = true, string? template = null,
        int tpDegree = 1, ITensorParallelGroup? tpGroup = null, int layerSplitDegree = 1)
        : base(path, BackendType.Cpu, tpDegree, tpGroup, layerSplitDegree)
    {
        _supportsLora = supportsLora;
        _in = inDim;
        _out = outDim;
        Config = new ModelConfig { Architecture = "probe", VocabSize = 128, NumLayers = 1, ChatTemplate = template };
        Tokenizer = new ProbeCharTokenizer();
        _maxContextLength = 4096;
        _weights[Target] = CreateFloatTensor(weight, outDim, inDim);
        _weights[Norm] = CreateFloatTensor(Enumerable.Repeat(1f, inDim).ToArray(), inDim);
    }

    protected override bool SupportsTextLora => _supportsLora;

    /// <summary>Every weight of the probe runs through <c>LinearForward</c>.</summary>
    protected override string? TextLoraWeightRefusal(string weightName) => null;

    /// <summary>When set, binding through <see cref="IAdapterAwareModel"/> plans the adapter
    /// and then throws this, as a factor upload that runs out of device memory does.</summary>
    public Exception? BindFailure { get; set; }

    bool IAdapterAwareModel.TryBindAdapter(IRequestAdapter adapter, out string? error)
    {
        if (BindFailure == null)
            return TryBindAdapter(adapter, out error);
        PlanTextLora((TextLoraAdapter)adapter);
        throw BindFailure;
    }

    /// <summary>The fast-path predicate (<c>ModelBase.IsLoraActiveForStep</c>).</summary>
    public bool LoraActiveForStep => IsLoraActiveForStep;

    public override bool SupportsKVStateSnapshot => true;
    public override bool SupportsCrossSequenceKvReuse => false;
    public override string KVStateFingerprint => "lora-probe";
    public override long ComputeKVBlockByteSize(int tokenCount) => 4L * tokenCount;
    public override bool TryExtractKVBlock(int start, int count, Span<byte> destination) => true;
    public override bool TryInjectKVBlock(int start, int count, ReadOnlySpan<byte> source) => true;

    /// <summary><c>LinearForward(x, Target)</c> for <paramref name="rows"/> rows of <paramref name="x"/>.</summary>
    public float[] Linear(float[] x, int rows)
    {
        using var input = CreateFloatTensor(x, rows, _in);
        using var y = LinearForward(input, Target);
        return y.GetElementsAsFloat(rows * _out);
    }

    protected override float[] ForwardCore(int[] tokens)
    {
        float[] y = Linear(Enumerable.Repeat(1f, _in).ToArray(), 1);
        var logits = Enumerable.Repeat(-1000f, 128).ToArray();
        logits[y[0] > 0.5f ? 'L' : 'B'] = 1000f;
        return logits;
    }

    protected override void ResetKVCacheCore() { }

    private sealed class ProbeCharTokenizer : ITokenizer
    {
        public string[] Vocab => Enumerable.Range(0, 128).Select(i => ((char)i).ToString()).ToArray();
        public int VocabSize => 128;
        public int BosTokenId => -1;
        public int[] EosTokenIds => new[] { 1 };
        public bool IsEos(int id) => id == 1;
        public int LookupToken(string token) => -1;
        public List<int> Encode(string text, bool addSpecial = true) => text.Select(c => c < 128 ? (int)c : '?').ToList();
        public string Decode(List<int> tokens) => new(tokens.Select(id => (char)id).ToArray());
        public void AppendTokenBytes(int token, List<byte> bytes) => bytes.Add((byte)token);
    }
}
