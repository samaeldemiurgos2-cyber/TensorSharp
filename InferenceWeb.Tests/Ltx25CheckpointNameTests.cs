using TensorSharp.Models.Ltx25;

namespace InferenceWeb.Tests;

/// <summary>
/// unsloth/LTX-2.5-GGUF keeps the dev checkpoint (30 steps, CFG 3) at its root and the distilled
/// one in distilled/. The two GGUFs have the same tensors and config, so a dev file loaded as if
/// it were distilled samples noise without an error; the model refuses it by its name instead.
/// </summary>
public sealed class Ltx25CheckpointNameTests
{
    [Theory]
    [InlineData("ltx-2.5-22b-dev-UD-Q4_K_XL.gguf")]
    [InlineData("ltx-2.5-22b-dev-Q8_0.gguf")]
    [InlineData("ltx-2.5-22b-dev-BF16.gguf")]
    [InlineData("/models/ltx25/LTX-2.5-Dev-Q4_K_M.gguf")]
    [InlineData("ltx-2.5-22b-dev-transformer-bf16.gguf")]
    [InlineData("ltx_2.5_dev_q4_k_m.gguf")]
    public void TheDevCheckpointIsRecognizedByItsName(string file) =>
        Assert.True(Ltx25Model.IsDevCheckpointName(file));

    [Theory]
    [InlineData("ltx-2.5-22b-distilled-UD-Q4_K_XL.gguf")]
    [InlineData("/models/ltx25/distilled/ltx-2.5-22b-distilled-Q8_0.gguf")]
    [InlineData("LTX-2.5-Distilled-Q4_K_M.gguf")]
    // "dev" only as a whole word: device, developer and the like are not the dev checkpoint.
    [InlineData("ltx-2.5-device-test.gguf")]
    [InlineData("ltx-2.5-devel-Q4_K_M.gguf")]
    // A distilled file is distilled whatever else its name says.
    [InlineData("ltx-2.5-22b-dev-distilled-merge-Q4_K_M.gguf")]
    public void DistilledFilesAreNot(string file) =>
        Assert.False(Ltx25Model.IsDevCheckpointName(file));
}
