// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// Qwen3.5/3.8 recurrent-state hand-off probe.
//
// Drives several sequences through the per-request holders the way the
// scheduler's PerSequenceFused path does - prompt chunks that alternate between
// holders, solo decode steps, batched arena steps, a later prompt chunk, a
// release and a new sequence - and records a SHA-256 of every step's logits.
// TS_QWEN35_DEVICE_STATE_HANDOFF only changes where the recurrent state travels
// between those steps (the sequences' device copies instead of the host), so a
// run with it on must reproduce a run with it off bit for bit. See README.md.
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Speculative;

if (args.Length == 3 && args[0] == "--compare")
    return Compare(args[1], args[2]);

string modelPath = Environment.GetEnvironmentVariable("TS_TEST_QWEN35_MODEL")
    ?? throw new ArgumentException("Set TS_TEST_QWEN35_MODEL.");
string output = args.Length == 2 && args[0] == "--out" ? args[1]
    : throw new ArgumentException("Usage: --out <steps.jsonl> | --compare <a.jsonl> <b.jsonl>");
string kv = Environment.GetEnvironmentVariable("TS_KV_CACHE_DTYPE") ?? "q8_0";
if (!KvCacheDtypeConfig.TryParse(kv, out KvCacheDtype kvDtype))
    throw new ArgumentException($"Unknown KV cache dtype '{kv}'.");
KvCacheDtypeConfig.Set(kvDtype);
const int Chunk = 1024;

using var model = ModelBase.Create(modelPath, BackendType.GgmlCuda);
if (model is not Qwen35Model || model is not IBatchedPagedModel holders)
    throw new InvalidOperationException("Requires a Qwen3.5/3.6/3.8 model.");
// A DFlash drafter enables the batched speculative verify phase (4b).
string? draftPath = Environment.GetEnvironmentVariable("TS_TEST_QWEN35_DRAFT");
if (!string.IsNullOrEmpty(draftPath))
{
    model.LoadDFlashDraftWeights(draftPath);
    if (!model.HasDFlash)
        throw new InvalidOperationException($"DFlash drafter '{draftPath}' did not attach.");
}

var prefix = (TensorSharp.Runtime.Scheduling.PrefixCache.IHolderPrefixCacheModel)model;
var steps = new List<Step>();
var last = new Dictionary<string, float[]>(StringComparer.Ordinal);
// A sequence's next token when a speculative verify drew it instead of the logits in last.
var next = new Dictionary<string, int>(StringComparer.Ordinal);
var length = new Dictionary<string, int>(StringComparer.Ordinal);
string[] sources =
{
    "The river flows through a green valley. Trees grow beside the water and birds fly overhead. ",
    "A computer stores numbers in memory. A program performs arithmetic and produces a result. ",
    "Paris is the capital of France. Tokyo is the capital of Japan. Rome is the capital of Italy. ",
    "A violin has four strings tuned in fifths. An orchestra follows the conductor through changes in rhythm. ",
};
int[] Tokens(int source, int count, int offset = 0)
{
    int[] pool = model.Tokenizer.Encode(sources[source], addSpecial: false).ToArray();
    return Enumerable.Range(offset, count).Select(i => pool[i % pool.Length]).ToArray();
}

void Record(string phase, string id, float[] logits)
{
    var copy = (float[])logits.Clone();
    last[id] = copy;
    steps.Add(new Step(steps.Count, phase, id, length[id], ArgMax(copy),
        Convert.ToHexString(SHA256.HashData(MemoryMarshal.AsBytes(copy.AsSpan())))));
}

void Prompt(string phase, string id, int[] tokens)
{
    holders.BindSequenceCache(id);
    model.PrepareForPrefill(length[id] + tokens.Length + 64);
    for (int at = 0; at < tokens.Length; at += Chunk)
    {
        if (at > 0) holders.BindSequenceCache(id);
        int[] chunk = tokens.Skip(at).Take(Chunk).ToArray();
        float[] logits = model.ForwardPrompt(chunk);
        length[id] += chunk.Length;
        Record(phase, id, logits);
    }
}

void SoloDecode(string phase, IEnumerable<string> ids, int count)
{
    for (int s = 0; s < count; s++)
        foreach (string id in ids)
        {
            holders.BindSequenceCache(id);
            int token = next.Remove(id, out int pendingToken) ? pendingToken : ArgMax(last[id]);
            float[] logits = model.Forward(new[] { token });
            length[id]++;
            Record(phase, id, logits);
        }
}

var watch = Stopwatch.StartNew();
string[] ids = { "A", "B", "C" };
int[] promptLengths = { 2600, 1900, 3100 };
foreach (string id in ids) length[id] = 0;

// 1. Prompt chunks that alternate between three holders: every chunk after a
//    sequence's first one resumes a state another sequence displaced.
var prompts = ids.Select((id, row) => Tokens(row, promptLengths[row])).ToArray();
for (int at = 0; prompts.Any(p => at < p.Length); at += Chunk)
    for (int row = 0; row < ids.Length; row++)
        if (at < prompts[row].Length)
        {
            holders.BindSequenceCache(ids[row]);
            if (at == 0) model.PrepareForPrefill(prompts[row].Length + 512);
            int[] chunk = prompts[row].Skip(at).Take(Chunk).ToArray();
            float[] logits = model.ForwardPrompt(chunk);
            length[ids[row]] += chunk.Length;
            Record("interleaved-prefill", ids[row], logits);
        }

// 2. A capture and clone before any solo decode has run: A's state sits in its own
//    device copies (handed off when C was bound) before the decode path ever built
//    its descriptors, and the capture must still read it.
Capture("A", "ckpt-early");
Clone("ckpt-early", "ckpt-early", "G");
SoloDecode("early-clone-decode", new[] { "G" }, 2);
holders.OnSequenceReleased("G");

// 3. Solo fused decode, round robin.
SoloDecode("solo-decode", ids, 6);

// 4. Batched arena decode over every holder.
for (int s = 0; s < 4; s++)
{
    int[] tokens = ids.Select(id => ArgMax(last[id])).ToArray();
    int[] positions = ids.Select(id => length[id]).ToArray();
    var batch = new float[ids.Length][];
    if (!holders.TryForwardBatchedFusedDecode(ids, tokens, positions, batch))
        throw new InvalidOperationException("Batched decode declined: " + holders.BatchedFusedDecodeDeclineReason);
    for (int i = 0; i < ids.Length; i++)
    {
        length[ids[i]]++;
        Record("batched-decode", ids[i], batch[i]);
    }
}

// 4b. Batched speculative verify and commit over every holder (with a drafter). Each
//     round verifies every sequence's next token followed by fixed draft tokens and
//     keeps a different prefix per sequence, so the rounds commit every acceptance
//     count; the decode that follows reads the committed recurrent state.
if (model.HasDFlash)
{
    var spec = (IBatchedSpeculativeTarget)model;
    int window = spec.BatchedSpecWindow(ids.Length);
    if (window < 2)
        throw new InvalidOperationException("Batched speculative verify is unavailable.");
    int feature = ((ISpeculativeTarget)model).SpecFeatureSize;
    for (int r = 0; r < 2 * window; r++)
    {
        var tokens = new int[ids.Length * window];
        var positions = new int[ids.Length];
        for (int i = 0; i < ids.Length; i++)
        {
            tokens[i * window] = next.TryGetValue(ids[i], out int pending) ? pending : ArgMax(last[ids[i]]);
            Array.Copy(Tokens(i, window - 1, offset: r), 0, tokens, i * window + 1, window - 1);
            positions[i] = length[ids[i]];
        }
        var drawn = new int[tokens.Length];
        var features = new float[tokens.Length * feature];
        if (!spec.TryBatchedSpecVerify(ids, positions, window, tokens, drawn, features))
            throw new InvalidOperationException("Batched speculative verify declined.");
        int[] kept = ids.Select((_, i) => 1 + (r + i) % window).ToArray();
        spec.CommitBatchedSpecVerify(ids, positions, kept);
        for (int i = 0; i < ids.Length; i++)
        {
            string id = ids[i];
            length[id] += kept[i];
            next[id] = drawn[i * window + kept[i] - 1];
            using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
            hash.AppendData(MemoryMarshal.AsBytes(drawn.AsSpan(i * window, window)));
            hash.AppendData(MemoryMarshal.AsBytes(features.AsSpan(i * window * feature, window * feature)));
            steps.Add(new Step(steps.Count, "batched-spec", id, length[id], next[id], Convert.ToHexString(hash.GetHashAndReset())));
        }
    }
}

// 5. Back to solo decode, then a second-turn prompt chunk per holder and more decode.
SoloDecode("solo-after-batch", ids, 2);
for (int row = 0; row < ids.Length; row++)
    Prompt("second-turn-prefill", ids[row], Tokens(row, 300, offset: 7));
SoloDecode("solo-after-turn", ids, 2);

// 6. Release B while it is bound, keep decoding the others, then start D at position 0.
holders.BindSequenceCache("B");
holders.OnSequenceReleased("B");
SoloDecode("after-release", new[] { "A", "C" }, 2);
length["D"] = 0;
Prompt("new-sequence-prefill", "D", Tokens(3, 1500));
SoloDecode("new-sequence-decode", new[] { "A", "C", "D" }, 2);

// 7. Checkpoint chain, the prefix cache's route: capture A, clone it into E (a clone
//    copies the unbound auxiliary layer on the host), extend and capture E, clone that
//    into F and decode. Capturing a clone exercises the background capture route.
void Clone(string key, string from, string id)
{
    if (!prefix.SettleForCopy(key) || !holders.TryCloneRetainedCache(key, id))
        throw new InvalidOperationException($"Clone of {key} declined.");
    length[id] = length[from];
    last[id] = last[from];
}
void Capture(string id, string key)
{
    holders.BindSequenceCache(id);
    if (!holders.TryCheckpointActiveCache(key))
        throw new InvalidOperationException($"Capture of {id} declined.");
    length[key] = length[id];
    last[key] = last[id];
    // A checkpoint is written to disk from its host bytes; every capture route must keep
    // them current.
    using var file = new MemoryStream();
    if (!holders.TryExportRetainedCache(key, file) || file.Length == 0)
        throw new InvalidOperationException($"Export of checkpoint {key} declined.");
}
Capture("A", "ckpt-A");
SoloDecode("after-capture", new[] { "A" }, 2);
Clone("ckpt-A", "ckpt-A", "E");
Prompt("clone-prefill", "E", Tokens(0, 600, offset: 11));
Capture("E", "ckpt-E");
SoloDecode("clone-decode", new[] { "E" }, 2);
Clone("ckpt-E", "ckpt-E", "F");
SoloDecode("second-clone-decode", new[] { "F", "E", "A" }, 3);

holders.RestorePrimaryCache();
foreach (string id in new[] { "A", "C", "D", "E", "F" }) holders.OnSequenceReleased(id);
foreach (string key in new[] { "ckpt-early", "ckpt-A", "ckpt-E" }) holders.DiscardRetainedCache(key);

Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(output))!);
using (var writer = new StreamWriter(output))
    foreach (Step step in steps)
        writer.WriteLine(JsonSerializer.Serialize(step));
Console.WriteLine($"[handoff-probe] handoff={Environment.GetEnvironmentVariable("TS_QWEN35_DEVICE_STATE_HANDOFF") ?? "default"} " +
    $"steps={steps.Count} kv={kvDtype} elapsed={watch.Elapsed.TotalSeconds:F1}s -> {output}");
return 0;

static int Compare(string a, string b)
{
    Step[] Read(string path) => File.ReadLines(path).Select(l => JsonSerializer.Deserialize<Step>(l)!).ToArray();
    Step[] left = Read(a), right = Read(b);
    int mismatches = 0;
    for (int i = 0; i < Math.Max(left.Length, right.Length); i++)
    {
        Step? x = i < left.Length ? left[i] : null, y = i < right.Length ? right[i] : null;
        if (x != null && y != null && x.Phase == y.Phase && x.Sequence == y.Sequence
            && x.Position == y.Position && x.Sha256 == y.Sha256)
            continue;
        mismatches++;
        Console.WriteLine($"MISMATCH step {i}: {x} | {y}");
    }
    Console.WriteLine($"[handoff-probe] {left.Length} vs {right.Length} steps, {mismatches} mismatched");
    return mismatches == 0 && left.Length > 0 ? 0 : 1;
}

static int ArgMax(float[] values)
{
    int best = 0;
    for (int i = 1; i < values.Length; i++)
        if (values[i] > values[best]) best = i;
    return best;
}

sealed record Step(int Index, string Phase, string Sequence, int Position, int ArgMax, string Sha256);
