// Concurrent-decode aggregate-throughput bench for ANY model via InferenceEngine.
// Reproduces the engine_comparison "parallel-request scaling" collapse: submits N
// identical decode requests at once and reports per-request and aggregate decode
// tok/s (decode window = first-token .. last-token, prefill excluded). The whole
// point is to compare N=1 vs N>=2 on the fused per-seq decode path.
//
// Env:
//   TS_DECODE_MODEL   path to .gguf (required-ish; default Gemma 12B QAT)
//   TS_DECODE_BACKEND ggml_cuda (default) | ggml_metal | ggml_vulkan | ggml_cpu | cpu | cuda
//   TS_CONC           comma list of concurrency levels (default 1,2)
//   TS_CONC_REPEATS   timed/correctness runs per concurrency level (default 1)
//   TS_NEW            decode tokens per request (default 96)
//   TS_PREFILL        prompt tokens (default 48)
using System.Diagnostics;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling;

static int EnvInt(string n, int d){ var s=Environment.GetEnvironmentVariable(n); return !string.IsNullOrEmpty(s)&&int.TryParse(s,out int v)&&v>0?v:d; }
static int[] EnvList(string n, int[] d){ var s=Environment.GetEnvironmentVariable(n); if(string.IsNullOrEmpty(s))return d; var l=new List<int>(); foreach(var p in s.Split(',',StringSplitOptions.RemoveEmptyEntries|StringSplitOptions.TrimEntries)) if(int.TryParse(p,out int v)&&v>0)l.Add(v); return l.Count>0?l.ToArray():d; }

string modelPath = Environment.GetEnvironmentVariable("TS_DECODE_MODEL") ?? @"C:\Works\models\gemma_mtp\qat\gemma-4-12B-it-qat-UD-Q4_K_XL.gguf";
BackendType backend = (Environment.GetEnvironmentVariable("TS_DECODE_BACKEND") ?? "ggml_cuda").ToLowerInvariant() switch {
    "ggml_cuda"=>BackendType.GgmlCuda, "ggml_metal"=>BackendType.GgmlMetal,
    "ggml_vulkan"=>BackendType.GgmlVulkan, "ggml_cpu"=>BackendType.GgmlCpu,
    "cpu"=>BackendType.Cpu, "cuda"=>BackendType.Cuda, _=>BackendType.GgmlCuda };
int[] concs = EnvList("TS_CONC", new[]{1,2});
int repeats = EnvInt("TS_CONC_REPEATS", 1);
int newTok = EnvInt("TS_NEW", 96);
int prefill = EnvInt("TS_PREFILL", 48);
int blockSize = 256;

Console.WriteLine($"[conc-bench] loading {Path.GetFileName(modelPath)} backend={backend} new={newTok} prefill={prefill} conc=[{string.Join(",",concs)}] repeats={repeats}");
using var model = ModelBase.Create(modelPath, backend);

string baseText = "The history of computing spans many centuries, beginning with simple counting tools and culminating in the modern electronic computer, transforming how humanity stores and reasons about information. ";
int[] pool = model.Tokenizer.Encode(baseText, addSpecial:false).ToArray();
int[] MakePrompt(int n, int salt){ var t=new int[n]; for(int i=0;i<n;i++)t[i]=pool[(i+salt)%pool.Length]; return t; }

var cfg = new SchedulerConfig {
    MaxNumBatchedTokens = 8192, MaxNumRunningSequences = 16, MaxPrefillChunkSize = 2048,
    SoloPrefillChunkSize = 8192, NumBlocks = EnvInt("TS_SCHED_NUM_BLOCKS", 1024), BlockSize = blockSize,
    EnablePrefixCaching = false, DecodeQuantumTokens = blockSize,
};
using var engine = new InferenceEngine(model, cfg, NullLogger.Instance);

// TS_CONC_SAME_PROMPT=1 makes every sequence use the SAME prompt (isolates
// column-index bugs in batched decode: all columns must then be identical).
int Salt(int id, int n) => Environment.GetEnvironmentVariable("TS_CONC_SAME_PROMPT") == "1" ? 1000 : 1000 + id*131;

int requestSerial = 0;
async Task<(int count, double firstMs, double lastMs, List<int> toks)> RunOne(int id, int salt, Stopwatch wall)
{
    // Every invocation must get a fresh engine id. Reusing r0/r1 across the
    // solo-reference and concurrent phases can re-adopt a retained fused cache
    // and turns this into a cache-lifecycle test instead of a batching test.
    string requestId = $"r{id}-run{Interlocked.Increment(ref requestSerial)}";
    var seq = new SequenceState(requestId, MakePrompt(prefill, salt), newTok, blockSize, SamplingConfig.Greedy);
    var handle = engine.SubmitRequest(seq);
    int count=0; double first=-1, last=0; var toks=new List<int>();
    await foreach (var tk in handle.Tokens.ReadAllAsync())
    {
        double t = wall.Elapsed.TotalMilliseconds;
        if (first<0) first=t;
        last=t; count++; toks.Add(tk);
    }
    try { await handle.Completion; } catch {}
    return (count, first, last, toks);
}

async Task<(double perReq, double agg, List<List<int>> outs)> RunConc(int n, string label = null)
{
    var wall = Stopwatch.StartNew();
    var tasks = new List<Task<(int,double,double,List<int>)>>();
    for (int i=0;i<n;i++) tasks.Add(RunOne(i, Salt(i,n), wall));
    var res = await Task.WhenAll(tasks);
    wall.Stop();
    int total=0; double minFirst=double.MaxValue, maxLast=0; double perReqSum=0;
    var outs=new List<List<int>>();
    foreach (var (c,f,l,tk) in res){ total+=c; outs.Add(tk); if(f>=0)minFirst=Math.Min(minFirst,f); maxLast=Math.Max(maxLast,l);
        if (c>1 && l>f) perReqSum += (c-1)/((l-f)/1000.0); }
    double agg = (maxLast>minFirst) ? (total - n) / ((maxLast-minFirst)/1000.0) : 0; // exclude each seq's first token
    double perReq = perReqSum / n;
    string run = string.IsNullOrEmpty(label) ? "" : $" {label}";
    Console.WriteLine($"  N={n,2}{run}: per-req decode={perReq,7:F2} tok/s   aggregate={agg,7:F2} tok/s   (total {total} toks, window {(maxLast-minFirst):F0} ms)");
    return (perReq, agg, outs);
}

static double Median(List<double> values)
{
    var ordered = values.OrderBy(v => v).ToArray();
    int middle = ordered.Length / 2;
    return ordered.Length % 2 == 0
        ? (ordered[middle - 1] + ordered[middle]) / 2.0
        : ordered[middle];
}

// warm (excluded from TS_DUMP_LOGITS: the dump must hold a measured request's first token)
using (FirstTokenLogitsDump.Suppress())
    await RunConc(1, "warmup");

// Correctness: greedy solo token streams (the ground truth: each prompt run alone),
// then assert the concurrent run reproduces them EXACTLY for the same prompts.
Console.WriteLine("[conc-bench] correctness (concurrent greedy output == solo greedy output):");
int maxN = concs.Max();
var solo = new Dictionary<int,List<int>>();
for (int i=0;i<maxN;i++) { var sw=Stopwatch.StartNew(); var (_,_,_,t)=await RunOne(i, Salt(i,maxN), sw); solo[i]=t; }

Console.WriteLine("[conc-bench] results:");
bool anyMismatch = false;
foreach (int n in concs)
{
    var perReqRates = new List<double>(repeats);
    var aggregateRates = new List<double>(repeats);
    int totalMismatches = 0;
    for (int repeat = 0; repeat < repeats; repeat++)
    {
        string label = repeats > 1 ? $"run {repeat + 1}/{repeats}" : null;
        var (perReq, aggregate, outs) = await RunConc(n, label);
        perReqRates.Add(perReq);
        aggregateRates.Add(aggregate);

        int mismatches = 0;
        for (int i = 0; i < n; i++)
        {
            var expect = solo[i]; var got = outs[i];
            int cmp = Math.Min(expect.Count, got.Count);
            int firstDiff = -1;
            for (int k=0;k<cmp;k++) if (expect[k]!=got[k]) { firstDiff=k; break; }
            bool ok = expect.Count==got.Count && firstDiff<0;
            if (!ok) { mismatches++; string d = firstDiff<0?"len-only":$"@{firstDiff} solo={expect[firstDiff]} conc={got[firstDiff]} (of {cmp})"; Console.WriteLine($"      N={n} run {repeat + 1} req {i}: MISMATCH (solo {expect.Count} vs conc {got.Count}) firstDiff {d}"); }
        }
        totalMismatches += mismatches;
        anyMismatch |= mismatches > 0;
        Console.WriteLine($"      N={n} run {repeat + 1}: correctness {n-mismatches}/{n} requests match solo greedy output {(mismatches==0?"OK":"*** FAIL ***")}");
        if (mismatches > 0 && Environment.GetEnvironmentVariable("TS_CONC_DUMP_TEXT") == "1")
            for (int i = 0; i < n; i++)
            {
                string soloTxt = model.Tokenizer.Decode(solo[i]).Replace("\n", "\\n");
                string concTxt = model.Tokenizer.Decode(outs[i]).Replace("\n", "\\n");
                Console.WriteLine($"      [run {repeat + 1} req {i}] solo: {soloTxt.Substring(0, Math.Min(160, soloTxt.Length))}");
                Console.WriteLine($"      [run {repeat + 1} req {i}] conc: {concTxt.Substring(0, Math.Min(160, concTxt.Length))}");
            }
    }
    if (repeats > 1)
    {
        Console.WriteLine($"      N={n}: {repeats}-run per-req median={Median(perReqRates):F2} min={perReqRates.Min():F2} max={perReqRates.Max():F2} tok/s; aggregate median={Median(aggregateRates):F2} min={aggregateRates.Min():F2} max={aggregateRates.Max():F2} tok/s");
        Console.WriteLine($"      N={n}: repeated correctness {n * repeats - totalMismatches}/{n * repeats} request-runs match solo {(totalMismatches==0?"OK":"*** FAIL ***")}");
    }
}
Console.WriteLine("[conc-bench] done.");
if (anyMismatch)
    Environment.ExitCode = 2;
