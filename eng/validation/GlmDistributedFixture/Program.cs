// Generate a small deterministic fixture for DeepSeek41DistributedProbe.
// It preserves KDA, pooled MLA, mHC and expert/head partition geometry.
using System.Security.Cryptography;
using System.Text.Json;
using InferenceWeb.Tests;

if (args.Length is < 1 or > 5)
{
    Console.Error.WriteLine("Usage: GlmDistributedFixture FRESH_DIRECTORY [all|attention|experts|q8-control|iq3-iq4|mixed|mixed-checkpoint][-dense|-kda|-kda-dense] [LAYERS] [PREFILL_TOKENS] [EXPERT_FFN]");
    return 2;
}
string directory = Path.GetFullPath(args[0]);
if (Directory.Exists(directory)) throw new ArgumentException("Fixture directory must be fresh.");
Directory.CreateDirectory(directory);
string profile = args.Length > 1 ? args[1] : null;
bool denseOnly = profile?.EndsWith("-dense", StringComparison.Ordinal) == true;
bool kdaOnly = profile?.EndsWith("-kda-dense", StringComparison.Ordinal) == true ||
    profile?.EndsWith("-kda", StringComparison.Ordinal) == true;
string quantizationProfile = kdaOnly ? (denseOnly ? profile[..^10] : profile[..^4]) : denseOnly ? profile[..^6] : profile;
int layers = profile == null ? 6 : args.Length > 2 ? int.Parse(args[2]) : 4;
int prefix = profile == null ? 52 : args.Length > 3 ? int.Parse(args[3]) : 129;
int expertFfn = profile == null ? 128 : args.Length > 4 ? int.Parse(args[4]) : 2048;
if (prefix is < 2 or > 1016) throw new ArgumentOutOfRangeException(nameof(prefix));
string path = profile == null ? GlmDsaSyntheticModelBuilder.WriteGlm5NextTpFixture(
    Path.Combine(directory, "glm53-mixed-q8.gguf"), numHeads: 8, quantizeAttentionOutput: true,
    numLayers: 6, mixedAttention: true, routedExperts: true, quantizeExperts: true,
    expertFfn: 128, architecture: "glm5-next") :
    GlmDsaSyntheticModelBuilder.WriteGlm5NextCudaFixture(
        Path.Combine(directory, $"glm53-{profile}.gguf"), contextLength: 1024, indexerTopK: 2048,
        distributedQuantizationProfile: quantizationProfile, distributedLayers: layers,
        distributedDenseOnly: denseOnly, distributedKdaOnly: kdaOnly, distributedExpertFfn: expertFfn);
int[] tokens = Enumerable.Range(0, prefix + 8).Select(i => 40 + i * 7 % 70).ToArray();
File.WriteAllText(Path.Combine(directory, "tokens.json"), JsonSerializer.Serialize(tokens));
File.WriteAllText(Path.Combine(directory, "fixture.json"), JsonSerializer.Serialize(new
{
    synthetic = true, architecture = profile == null ? "glm5-next" : "glm5next", layers, tokens = tokens.Length,
    quantization_profile = profile ?? "original-q8", native_quantizer = quantizationProfile is "all" or "attention" or "experts" or "iq3-iq4" or "mixed" or "mixed-checkpoint",
    dense_only = denseOnly, kda_only = kdaOnly,
    source_layer_roles = quantizationProfile is "mixed" or "mixed-checkpoint" ? Enumerable.Range(0, layers)
        .Select(layer => layer == 0 ? 0 : new[] { 11, 12, 44 }[(layer - 1) % 3]).ToArray() : null,
    head_dim = profile == null ? 16 : 128, heads = 8, hidden = profile == null ? 64 : 256,
    expert_ffn = expertFfn, prefill_tokens_for_decode8 = prefix,
    sha256 = Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(path))),
    coverage = "Numerical state and distributed partition fixture; not evidence of production checkpoint quality or throughput."
}, new JsonSerializerOptions { WriteIndented = true }));
Console.WriteLine(path);
return 0;
