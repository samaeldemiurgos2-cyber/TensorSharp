// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp.Models;

namespace InferenceWeb.Tests;

public sealed class MuseSparkVisionProjectorTests
{
    [Theory]
    [InlineData("muse-glimmer", null, 8192, true)]
    [InlineData(null, "muse-glimmer", 8192, true)]
    [InlineData("muse-glimmer", "other", 8192, true)]
    [InlineData("other", "muse-glimmer", 8192, false)]
    [InlineData("muse-glimmer", null, 6656, false)]
    [InlineData(null, null, 8192, false)]
    public void MixedModalityProjectorUsesVisionTypeAndExactTextWidth(
        string visionType, string legacyType, int projectionWidth, bool valid)
    {
        string path = Path.Combine(Path.GetTempPath(), "spark-projector-" + Guid.NewGuid() + ".gguf");
        try
        {
            var metadata = new List<SyntheticGguf.Kv>
            {
                new SyntheticGguf.U32 { Key = "clip.vision.projection_dim", V = (uint)projectionWidth },
                new SyntheticGguf.Str { Key = "clip.audio.projector_type", V = "muse-spark-audio" },
                new SyntheticGguf.Bool { Key = "clip.has_vision_encoder", V = true },
                new SyntheticGguf.Bool { Key = "clip.has_audio_encoder", V = true },
            };
            if (visionType != null) metadata.Add(new SyntheticGguf.Str { Key = "clip.vision.projector_type", V = visionType });
            if (legacyType != null) metadata.Add(new SyntheticGguf.Str { Key = "clip.projector_type", V = legacyType });
            SyntheticGguf.Write(path, metadata, []);
            using var file = new GgufFile(path);
            if (valid) MuseSparkModel.ValidateVisionProjector(file, 8192);
            else Assert.Throws<ArgumentException>(() => MuseSparkModel.ValidateVisionProjector(file, 8192));
        }
        finally { File.Delete(path); }
    }
}
