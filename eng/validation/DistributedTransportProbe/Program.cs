using System.Diagnostics;
using System.Reflection;
using System.Security.Cryptography;
using System.Text.Json;
using TensorSharp.Distributed;

// Both nodes use identical arguments except --node-id and --output. Timings
// include synchronization inherent to the actual transport implementation.
var options = new Dictionary<string, string>(StringComparer.Ordinal);
for (int i = 0; i < args.Length; i += 2)
{
    if (i + 1 == args.Length || !options.TryAdd(args[i], args[i + 1]))
        throw new ArgumentException("Every option requires exactly one value.");
}
var allowed = new HashSet<string> { "--node-id", "--peers", "--output", "--route", "--iterations", "--warmup" };
if (options.Keys.Any(key => !allowed.Contains(key))) throw new ArgumentException("Unknown option.");
int node = int.Parse(options["--node-id"]);
int iterations = int.Parse(options.GetValueOrDefault("--iterations", "100"));
int warmup = int.Parse(options.GetValueOrDefault("--warmup", "10"));
if (iterations < 3 || warmup < 1) throw new ArgumentException("Use at least three iterations and one warmup.");
string output = Path.GetFullPath(options["--output"]);
if (File.Exists(output)) throw new ArgumentException("Output must be fresh: " + output);
Directory.CreateDirectory(Path.GetDirectoryName(output)!);
var cases = new List<object>();
var report = new Dictionary<string, object?>
{
    ["schema_version"] = 1, ["status"] = "running", ["node_id"] = node,
    ["arguments"] = args, ["route"] = options.GetValueOrDefault("--route", "unspecified"),
    ["started_utc"] = DateTime.UtcNow, ["cases"] = cases,
    ["transport_sha256"] = Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(typeof(TcpCommunicator).Assembly.Location))),
    ["probe_sha256"] = Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(Assembly.GetExecutingAssembly().Location))),
    ["limitations"] = new[] { "Host F32 buffers; no GPU synchronization or GPU staging cost is measured.",
        "This is a transport microbenchmark, not model quality or end-to-end inference throughput.",
        "Compare matching routes, payloads and competing workloads; a tunnel does not qualify a direct fabric." }
};
void Save() => File.WriteAllText(output, JsonSerializer.Serialize(report, new JsonSerializerOptions { WriteIndented = true }) + "\n");
Save();
try
{
    var peers = DistributedTpConfig.ParsePeers(options["--peers"]);
    using var communicator = new TcpCommunicator(node, peers);
    foreach (int length in new[] { 1, 1024, 16384, 262144, 1048576 })
    {
        float[] buffer = new float[length];
        foreach (string operation in new[] { "all_reduce", "broadcast_from_last" })
        {
            var samples = new List<double>();
            for (int iteration = -warmup; iteration < iterations; iteration++)
            {
                for (int i = 0; i < length; i++) buffer[i] = operation == "all_reduce" ? node + 1 + i % 17 : node == peers.Length - 1 ? i % 251 : -1;
                communicator.Barrier();
                long start = Stopwatch.GetTimestamp();
                if (operation == "all_reduce") communicator.AllReduce(buffer, length);
                else communicator.Broadcast(buffer, peers.Length - 1);
                double milliseconds = Stopwatch.GetElapsedTime(start).TotalMilliseconds;
                for (int i = 0; i < length; i++)
                {
                    float expected = operation == "all_reduce" ? peers.Length * (peers.Length + 1) / 2 + peers.Length * (i % 17) : i % 251;
                    if (buffer[i] != expected) throw new InvalidOperationException($"{operation} payload mismatch at {i}: {buffer[i]} != {expected}");
                }
                if (iteration >= 0) samples.Add(milliseconds);
            }
            double[] sorted = samples.Order().ToArray();
            double median = (sorted[(sorted.Length - 1) / 2] + sorted[sorted.Length / 2]) / 2;
            cases.Add(new { operation, bytes = length * sizeof(float), warmup, iterations,
                median_ms = median, p95_ms = sorted[(int)Math.Ceiling(.95 * sorted.Length) - 1],
                payload_mib_per_second = length * sizeof(float) / 1048576d / (median / 1000), samples_ms = samples });
            Save();
        }
    }
    communicator.Barrier();
    report["status"] = "passed";
    return 0;
}
catch (Exception error)
{
    report["status"] = "failed";
    report["error"] = error.ToString();
    return 1;
}
finally
{
    report["finished_utc"] = DateTime.UtcNow;
    Save();
}
