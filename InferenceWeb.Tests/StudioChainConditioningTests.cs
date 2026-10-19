using System.Reflection;
using System.Runtime.CompilerServices;
using TensorSharp.Chat.Studio;
using TensorSharp.Models;
using TensorSharp.Models.Ltx25;
using TensorSharp.Models.MiniMaxH3;
using TensorSharp.Models.Video;
using TensorSharp.Models.WanVideo;
using TensorSharp.Studio;

namespace InferenceWeb.Tests;

/// <summary>Chain segments that keep a character: a ref request that also starts on a first frame. The LTX 2.5 pipeline
/// composes it as upstream ICLoraPipeline does (a frame-0 image guide and the Ingredients sheet in one pass), but no
/// checkpoint offers it in Studio, and each says why. Metadata-only models; no weights or inference.</summary>
[Collection(MediaCodecSwaps.Name)]
public sealed class StudioChainConditioningTests
{
    static T WithProperty<T>(T model, string property, object value)
    {
        typeof(T).GetField($"<{property}>k__BackingField", BindingFlags.Instance | BindingFlags.NonPublic)!.SetValue(model, value);
        return model;
    }
    static Ltx25Model Ltx(bool ingredients)
    {
        var model = (Ltx25Model)RuntimeHelpers.GetUninitializedObject(typeof(Ltx25Model));
        return ingredients ? WithProperty(model, "ReferenceLoraPath", "/models/ltx25/loras/ltx-2.5-22b-ic-lora-ingredients-0.9.safetensors") : model;
    }
    static MiniMaxH3Model H3(MiniMaxH3Partition partition) =>
        WithProperty((MiniMaxH3Model)RuntimeHelpers.GetUninitializedObject(typeof(MiniMaxH3Model)), nameof(MiniMaxH3Model.Partition), partition);
    static WanVideoModel Wan() => WithProperty(WithProperty((WanVideoModel)RuntimeHelpers.GetUninitializedObject(typeof(WanVideoModel)),
        nameof(WanVideoModel.Variant), WanVariant.TI2V), nameof(WanVideoModel.InputChannels), 48);

    static StudioGenerationRequest ChainWithCharacter() => new()
    {
        ModelId = "model", Prompt = "She keeps walking through the market.", Mode = StudioGenerationModes.Reference,
        Width = 768, Height = 448, FrameCount = 121, Fps = 24, Steps = 8, CfgScale = 1, Seed = 42,
        ConditioningAssetId = "chain", ReferenceAssetIds = ["face", "coat"],
    };
    static readonly Dictionary<string, string> Assets = new() { ["chain"] = "/p/chain.png", ["face"] = "/p/face.png", ["coat"] = "/p/coat.png" };

    [Fact]
    public void ARefRequestThatStartsOnAFrameReachesLtxAsAFrameZeroGuideBesideTheSheet()
    {
        var parameters = NativeStudioGenerationAdapter.Parameters(ChainWithCharacter(), Assets, null, default);
        // The first frame is a guide, never a third identity image.
        Assert.Null(parameters.ImagePath);
        var guide = Assert.Single(parameters.ImageConditions);
        Assert.Equal(("/p/chain.png", 0, 1f), (guide.ImagePath, guide.FrameIndex, guide.Strength));
        Assert.Equal(["/p/face.png", "/p/coat.png"], parameters.ReferenceImagePaths);
        var settings = Ltx25GenerationSettings.Resolve(parameters, supportsIdentityReferences: true);
        Assert.False(settings.TwoStage); // Ingredients keeps its single full-size pass
        Assert.Equal(["/p/face.png", "/p/coat.png"], Ltx25ReferenceConditioning.ResolveSources(parameters).Select(s => s.ImagePath));
        Assert.Equal(0, Ltx25ReferenceConditioning.ResolveFirstFrameGuide(parameters)!.FrameIndex);
        // Without a first frame the same request is a plain ref take, as before.
        var plain = ChainWithCharacter(); plain.ConditioningAssetId = null;
        var plainParameters = NativeStudioGenerationAdapter.Parameters(plain, Assets, null, default);
        Assert.Empty(plainParameters.ImageConditions); Assert.Null(Ltx25ReferenceConditioning.ResolveFirstFrameGuide(plainParameters));
    }

    [Fact]
    public void IdentityReferencesTakeOneGuideAndOnlyAtFrameZero()
    {
        VideoGenerationParams Ref(params VideoImageCondition[] guides) => new() { Mode = "ref", ReferenceImagePaths = ["face.png"], ImageConditions = guides, Frames = 121 };
        Assert.Equal(0, Ltx25ReferenceConditioning.ResolveFirstFrameGuide(Ref(new VideoImageCondition { ImagePath = "chain.png", FrameIndex = 0 }))!.FrameIndex);
        // A single identity image (the shorthand form) keeps working beside the guide.
        Assert.Equal("identity.png", Assert.Single(Ltx25ReferenceConditioning.ResolveSources(new() { Mode = "ref", ImagePath = "identity.png",
            ImageConditions = [new() { ImagePath = "chain.png", FrameIndex = 0 }] })).ImagePath);
        foreach (var refused in new[]
        {
            Ref(new VideoImageCondition { ImagePath = "timed.png", FrameIndex = 8 }),
            Ref(new VideoImageCondition { ImagePath = "last.png", FrameIndex = -1 }),
            Ref(new VideoImageCondition { ImagePath = "a.png", FrameIndex = 0 }, new VideoImageCondition { ImagePath = "b.png", FrameIndex = 0 }),
            Ref(new VideoImageCondition { FrameIndex = 0 }),
            Ref(new VideoImageCondition { ImagePath = "chain.png", FrameIndex = 0, Strength = 1.5f }),
        })
        {
            Assert.Throws<ArgumentException>(() => Ltx25ReferenceConditioning.ResolveSources(refused));
            Assert.Throws<ArgumentException>(() => Ltx25GenerationSettings.Resolve(refused, true));
        }
        Assert.Throws<ArgumentException>(() => Ltx25ReferenceConditioning.ResolveSources(new() { Mode = "ref", ReferenceImagePaths = ["face.png"], EndImagePath = "last.png" }));
    }

    static float[] Random(int length, int seed) { var r = new Random(seed); var values = new float[length]; for (int i = 0; i < length; ++i) values[i] = (float)(r.NextDouble() * 2 - 1); return values; }

    [Theory]
    [InlineData(1f)]
    [InlineData(.6f)]
    public void AFirstFrameGuideIsPinnedOnTheCanvasAndLeavesTheReferenceTailAlone(float strength)
    {
        var settings = Ltx25GenerationSettings.Resolve(new() { Mode = "ref", ReferenceImagePaths = ["identity.png"], Width = 64, Height = 32, Frames = 121, Seed = 42 }, true);
        int spatial = settings.LatentHeight * settings.LatentWidth, tokens = settings.VideoTokens * 2;
        float[] noise = Random(tokens * 128, 1), reference = Random(settings.VideoTokens * 128, 2), image = Random(spatial * 128, 3);
        var input = new Ltx25DenoiserInput
        {
            Video = noise, VideoTokens = tokens, Audio = [0], AudioTokens = 1, VideoContext = [0], VideoContextTokens = 1, AudioContext = [0], AudioContextTokens = 1,
            VideoPositions = Ltx25Geometry.VideoPositions(settings.LatentFrames, settings.LatentHeight, settings.LatentWidth, 24),
            AudioPositions = Ltx25Geometry.AudioPositions(1), VideoSigma = 1, AudioSigma = 1, FrequencyPrecision = Ltx25FrequencyPrecision.Float64,
        };
        var referenced = Ltx25ReferenceConditioning.Apply(input, settings, reference, settings.LatentFrames, settings.LatentHeight, settings.LatentWidth);
        var pinned = Ltx25ImageConditioning.ApplyFirstFrame(referenced, settings, new Ltx25EncodedImage(image, 0, strength));
        // Canvas frame 0: the image's clean latent with mask 1 - strength, its noise blended toward it.
        float[] packed = Ltx25Geometry.PackVideo(image, spatial, 128);
        for (int i = 0; i < spatial * 128; ++i)
        {
            Assert.Equal(packed[i], pinned.CleanVideo[i]);
            Assert.Equal(packed[i] + (noise[i] - packed[i]) * (1 - strength), pinned.Video[i], 6);
        }
        Assert.All(pinned.VideoDenoiseMask.Take(spatial), m => Assert.Equal(1 - strength, m, 6));
        // Everything else is exactly the reference conditioning: the rest of the canvas, the tail, positions and markers.
        Assert.Equal(referenced.Video[(spatial * 128)..], pinned.Video[(spatial * 128)..]);
        Assert.Equal(referenced.CleanVideo[(spatial * 128)..], pinned.CleanVideo[(spatial * 128)..]);
        Assert.Equal(referenced.VideoDenoiseMask[spatial..], pinned.VideoDenoiseMask[spatial..]);
        Assert.Equal(referenced.VideoPositions.Boundaries, pinned.VideoPositions.Boundaries);
        Assert.Equal(referenced.KeyframeMask, pinned.KeyframeMask);
        Assert.True(pinned.UseReferenceLora); Assert.Equal(tokens, pinned.VideoTokens);
        // Pinning frame 0 on the canvas alone (the i2v path) gives the same frame-0 tokens: the order is immaterial.
        var canvas = new Ltx25DenoiserInput
        {
            Video = noise[..(settings.VideoTokens * 128)], VideoTokens = settings.VideoTokens, Audio = [0], AudioTokens = 1, VideoContext = [0], VideoContextTokens = 1,
            AudioContext = [0], AudioContextTokens = 1, VideoPositions = input.VideoPositions, AudioPositions = input.AudioPositions,
            VideoSigma = 1, AudioSigma = 1, FrequencyPrecision = Ltx25FrequencyPrecision.Float64,
        };
        var i2v = Ltx25ImageConditioning.Apply(canvas, settings, [new Ltx25EncodedImage(image, 0, strength)]);
        Assert.Equal(i2v.Video[..(spatial * 128)], pinned.Video[..(spatial * 128)]);
        Assert.Equal(i2v.VideoDenoiseMask[..spatial], pinned.VideoDenoiseMask[..spatial]);
        Assert.Throws<ArgumentException>(() => Ltx25ImageConditioning.ApplyFirstFrame(input, settings, new Ltx25EncodedImage(image, 0)));      // no clean latents yet
        Assert.Throws<ArgumentException>(() => Ltx25ImageConditioning.ApplyFirstFrame(referenced, settings, new Ltx25EncodedImage(image, 8))); // only frame 0
    }

    sealed class ConditioningCodec(bool supported) : TensorSharp.Models.Media.IVideoEncoder
    {
        public bool SupportsH264ImageConditioning => supported;
        public string SaveMp4(string path, TensorSharp.Models.QwenImage.RgbImage[] frames, int fps) => throw new InvalidOperationException("Capabilities never encode.");
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]  // no desktop FFmpeg/libx264 image transform
    public void NoLoadedCheckpointOffersACharacterWhileStartingOnAFrameAndEachSaysWhy(bool imageTransform)
    {
        var previous = TensorSharp.Models.Media.MediaCodecs.VideoEncoder;
        StudioModelCapability ltx, bare;
        try
        {
            TensorSharp.Models.Media.MediaCodecs.VideoEncoder = new ConditioningCodec(imageTransform);
            ltx = NativeStudioGenerationAdapter.CreateCapability(Ltx(ingredients: true), "ltx", new());
            bare = NativeStudioGenerationAdapter.CreateCapability(Ltx(ingredients: false), "ltx", new());
        }
        finally { TensorSharp.Models.Media.MediaCodecs.VideoEncoder = previous; }
        // LTX Ingredients composes the two (as upstream does) but cuts to the reference sheet: Studio does not offer it.
        Assert.False(ltx.ReferencesWithFirstFrame); Assert.Contains("cuts to the reference photos", ltx.ReferencesWithFirstFrameUnavailableReason);
        Assert.Contains("ref", ltx.Modes);
        Assert.Equal((768, 448), (ltx.ReferenceDefaultWidth, ltx.ReferenceDefaultHeight));
        // A host that pins one size for every mode keeps it for references too.
        Assert.Null(NativeStudioGenerationAdapter.CreateCapability(Ltx(ingredients: true), "ltx", new() { Width = 1024, Height = 1024 }).ReferenceDefaultWidth);
        // The family limit comes first whatever is installed, so the reason never sends anyone to install what would not help.
        Assert.False(bare.ReferencesWithFirstFrame); Assert.Contains("cuts to the reference photos", bare.ReferencesWithFirstFrameUnavailableReason);
        Assert.Contains("needs the Ingredients IC-LoRA", bare.ReferencesWithFirstFrameUnavailableReason);
        Assert.Equal(imageTransform, ltx.ReferencesWithFirstFrameUnavailableReason.Contains("Continue from the frame alone"));
        Assert.Null(bare.ReferenceDefaultWidth);
        foreach (var (model, why) in new (IVideoGenerationModel, string)[]
        {
            (H3(MiniMaxH3Partition.FirstLastFrame), "stable-diffusion.cpp"), (H3(MiniMaxH3Partition.Reference), "stable-diffusion.cpp"), (Wan(), "identity-reference"),
        })
        {
            var capability = NativeStudioGenerationAdapter.CreateCapability(model, "model", new());
            Assert.False(capability.ReferencesWithFirstFrame);
            Assert.Contains(why, capability.ReferencesWithFirstFrameUnavailableReason);
            Assert.Null(capability.ReferenceDefaultWidth);
        }
    }

    [Fact]
    public void AdmissionRefusesAChainWithACharacterOnACheckpointThatCannotKeepBoth()
    {
        var request = ChainWithCharacter();
        var h3 = NativeStudioGenerationAdapter.CreateCapability(H3(MiniMaxH3Partition.Reference), "h3", new());
        var refused = Assert.Throws<StudioException>(() => h3.RequireReferencesWithFirstFrame(request, StudioGenerationModes.Reference));
        Assert.Equal(("unsupported_conditioning", h3.ReferencesWithFirstFrameUnavailableReason), (refused.Code, refused.Message));
        // The same references without the first frame are an ordinary ref take there.
        request.ConditioningAssetId = null;
        h3.RequireReferencesWithFirstFrame(request, StudioGenerationModes.Reference);
    }
}

/// <summary>Tests that swap the process-wide <see cref="TensorSharp.Models.Media.MediaCodecs.VideoEncoder"/> run one at a time,
/// so no test reads the encoder another has replaced.</summary>
[CollectionDefinition(Name, DisableParallelization = true)]
public sealed class MediaCodecSwaps { public const string Name = "Media codec swaps"; }
