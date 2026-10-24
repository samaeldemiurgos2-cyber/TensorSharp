// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using TensorAgent.Core.Catalog;
using TensorAgent.Core.Hosting;
using TensorAgent.Core.Settings;
using TensorSharp.Models.MiniMaxH3;

namespace TensorAgent.Tests;

/// <summary>
/// The single MiniMax-H3 entry contains both denoisers and the shared files needed for
/// Video Studio to switch conditioning modes without reloading the host model.
/// </summary>
[Collection(ProcessEnvironmentCollection.Name)]
public sealed class MiniMaxH3CatalogTests
{
    private static CatalogModel Model => ModelCatalog.Find("minimax-h3-fl2va-q4k")!;

    private static readonly string[] VideoVariables =
    [
        "TS_VIDEO_TEXT_ENCODER", "TS_VIDEO_VAE", "TS_VIDEO_AUDIO_VAE", "TS_VIDEO_TOKENIZER",
    ];

    private static readonly string[] ImageVariables =
    [
        "TS_QWEN_IMAGE_VAE", "TS_QWEN_IMAGE_TE", "TS_QWEN_IMAGE_MMPROJ",
    ];

    [Fact]
    public void OneEntryContainsBothCheckpointsAndAllInputModalities()
    {
        Assert.Single(ModelCatalog.BuiltIn.Where(model => model.Family == CatalogFamily.MiniMaxH3));
        Assert.Equal(CatalogFamily.MiniMaxH3, Model.Family);
        Assert.Equal(CatalogArchitectureKind.Diffusion, Model.Kind);
        Assert.True(Model.IsVideoGenerator);
        Assert.False(Model.IsImageGenerator);
        Assert.True(Model.Modalities.HasFlag(CatalogModalities.VideoOutput | CatalogModalities.AudioOutput));
        Assert.Equal(0, Model.ContextLength);
        Assert.All(Model.Files, f => Assert.False(f.Optional, $"{Model.Id}/{f.FileName} is optional"));
        Assert.Equal("minimax_h3_fl2va_pruned-Q4_K.gguf", Model.Weights.FileName);
        Assert.Equal("minimax_h3_ref2va_pruned-Q4_K.gguf",
            Model.Files.Single(f => f.Role == CatalogFileRole.SecondaryWeights).FileName);
        Assert.Equal(CatalogModalities.Image | CatalogModalities.Video | CatalogModalities.Audio,
            Model.Modalities & (CatalogModalities.Image | CatalogModalities.Video | CatalogModalities.Audio));
    }

    [Fact]
    public void TheEntryCarriesSharedCompanionsForBothDenoisers()
    {
        Assert.Single(Model.Files, f => f.Role == CatalogFileRole.TextEncoder);
        Assert.Single(Model.Files, f => f.Role == CatalogFileRole.Vae);
        Assert.Single(Model.Files, f => f.Role == CatalogFileRole.AudioVae);
        Assert.Equal(new[] { "merges.txt", "tokenizer_config.json", "vocab.json" },
            Model.Files.Where(f => f.Role == CatalogFileRole.Tokenizer).Select(f => f.FileName).Order(StringComparer.Ordinal));
    }

    [Fact]
    public void TheCheckpointIsTheOneItsFileNameSays()
    {
        // The engine reads the partition off the file name (the GGUFs carry no metadata),
        // and the app stores the file under the catalog's name.
        Assert.Equal(MiniMaxH3Partition.FirstLastFrame, MiniMaxH3Config.PartitionFromFileName(Model.Weights.FileName));
        Assert.Equal(MiniMaxH3Partition.Reference, MiniMaxH3Config.PartitionFromFileName(
            Model.Files.Single(f => f.Role == CatalogFileRole.SecondaryWeights).FileName));
    }

    /// <summary>
    /// The engine resolves its companions from these variables and, failing that, from
    /// preferred file names - stated here, because they are private to the model, and
    /// kept honest by reading its source.
    /// </summary>
    [Fact]
    public void TheNamesAndVariablesThisCatalogReliesOnAreTheOnesTheEngineUses()
    {
        string model = ReadSource("TensorSharp.Models/Models/MiniMaxH3/MiniMaxH3Model.cs");
        foreach (string literal in new[]
                 {
                     "\"TS_VIDEO_TEXT_ENCODER\"", "\"TS_VIDEO_VAE\"", "\"TS_VIDEO_AUDIO_VAE\"",
                     $"\"{Model.Files.Single(f => f.Role == CatalogFileRole.TextEncoder).FileName}\"",
                     $"\"{Model.Files.Single(f => f.Role == CatalogFileRole.Vae).FileName}\"",
                     $"\"{Model.Files.Single(f => f.Role == CatalogFileRole.AudioVae).FileName}\"",
                 })
            Assert.Contains(literal, model, StringComparison.Ordinal);

        string encoder = ReadSource("TensorSharp.Models/Models/MiniMaxH3/MiniMaxH3TextEncoder.cs");
        foreach (string literal in new[] { "\"TS_VIDEO_TOKENIZER\"", "\"vocab.json\"", "\"merges.txt\"", "\"tokenizer_config.json\"" })
            Assert.Contains(literal, encoder, StringComparison.Ordinal);
    }

    [Fact]
    public void EveryInstalledCompanionIsPublishedAndTheTokenizerAsItsFolder()
    {
        using var installation = new FakeInstall(Model, install: m => m.Files);

        IReadOnlyDictionary<string, string> published = DiffusionCompanions.Publish(Model, installation.Store);

        Assert.Equal(installation.PathOf(CatalogFileRole.TextEncoder), published["TS_VIDEO_TEXT_ENCODER"]);
        Assert.Equal(installation.PathOf(CatalogFileRole.Vae), published["TS_VIDEO_VAE"]);
        Assert.Equal(installation.PathOf(CatalogFileRole.AudioVae), published["TS_VIDEO_AUDIO_VAE"]);
        Assert.Equal(installation.Store.DirectoryFor(Model), published["TS_VIDEO_TOKENIZER"]);
        Assert.Equal(VideoVariables.Order(StringComparer.Ordinal), published.Keys.Order(StringComparer.Ordinal));
        foreach (string variable in VideoVariables)
            Assert.Equal(published[variable], Environment.GetEnvironmentVariable(variable));
    }

    [Fact]
    public void ATokenizerFolderMissingAFileIsNotPublished()
    {
        Environment.SetEnvironmentVariable("TS_VIDEO_TOKENIZER", "/somewhere/from/before");
        using var installation = new FakeInstall(Model,
            install: m => m.Files.Where(f => f.FileName != "tokenizer_config.json"));

        IReadOnlyDictionary<string, string> published = DiffusionCompanions.Publish(Model, installation.Store);

        Assert.False(published.ContainsKey("TS_VIDEO_TOKENIZER"));
        Assert.Null(Environment.GetEnvironmentVariable("TS_VIDEO_TOKENIZER"));
        Assert.True(published.ContainsKey("TS_VIDEO_TEXT_ENCODER"));
    }

    [Fact]
    public void SwitchingFamiliesLeavesNoOtherModelsPathsBehind()
    {
        using var video = new FakeInstall(Model, install: m => m.Files);
        using var image = new FakeInstall(DiffusionModelFixture.QwenImage21, install: m => m.Files);

        DiffusionCompanions.Publish(DiffusionModelFixture.QwenImage21, image.Store);
        DiffusionCompanions.Publish(Model, video.Store);
        foreach (string variable in ImageVariables)
            Assert.Null(Environment.GetEnvironmentVariable(variable));
        Assert.NotNull(Environment.GetEnvironmentVariable("TS_VIDEO_TEXT_ENCODER"));

        DiffusionCompanions.Publish(DiffusionModelFixture.QwenImage21, image.Store);
        foreach (string variable in VideoVariables)
            Assert.Null(Environment.GetEnvironmentVariable(variable));
        Assert.NotNull(Environment.GetEnvironmentVariable("TS_QWEN_IMAGE_TE"));
    }

    [Fact]
    public void BothDenoisersAreStoredBesideTheSharedCompanions()
    {
        using var installation = new FakeInstall(Model, install: m => m.Files);
        Assert.True(File.Exists(installation.PathOf(CatalogFileRole.Weights)));
        Assert.True(File.Exists(installation.PathOf(CatalogFileRole.SecondaryWeights)));
        Assert.Equal(installation.PathOf(CatalogFileRole.TextEncoder),
            DiffusionCompanions.Publish(Model, installation.Store)["TS_VIDEO_TEXT_ENCODER"]);
    }

    /// <summary>
    /// A half-downloaded set is refused before the engine sees it. The pipeline would find
    /// a missing companion by its name somewhere in the model store - for the text
    /// encoder, Qwen-Image's - and the weights file existing is all the old check asked.
    /// </summary>
    [Fact]
    public void AnIncompleteSetIsNotLoaded()
    {
        string root = Path.Combine(Path.GetTempPath(), "tensoragent-h3-" + Guid.NewGuid().ToString("N"));
        var paths = new AgentPaths(Path.Combine(root, "data"), Path.Combine(root, "cache")) { DeviceMemoryGB = 48 };
        paths.EnsureCreated();
        try
        {
            using var host = new AgentAppHost(paths);
            var store = new ModelStore(paths.ModelsDirectory);
            Directory.CreateDirectory(store.DirectoryFor(Model));
            // The weights exist (empty, as a .part rename would leave a short file) and
            // nothing else does.
            File.WriteAllBytes(store.PathFor(Model, Model.Weights), []);

            var ex = Assert.Throws<FileNotFoundException>(() => host.UseModel(Model, warmAfterwards: false));
            Assert.Equal("MiniMax-H3 is not completely downloaded yet.", ex.Message);
            Assert.Equal(AgentAppHost.ModelLoadState.Failed, host.ModelLoad);
        }
        finally
        {
            DiffusionCompanions.Publish(null, new ModelStore(paths.ModelsDirectory));
            try { Directory.Delete(root, true); } catch (Exception) { /* scratch */ }
        }
    }

    /// <summary>The repo's own copy of a file, so a test can read the source it mirrors.</summary>
    private static string ReadSource(string relativePath)
    {
        var directory = new DirectoryInfo(AppContext.BaseDirectory);
        while (directory is not null && !File.Exists(Path.Combine(directory.FullName, "TensorSharp.slnx")))
            directory = directory.Parent;
        Assert.True(directory is not null, $"no repository root above {AppContext.BaseDirectory}");
        string full = Path.Combine(directory!.FullName, relativePath);
        Assert.True(File.Exists(full), $"{full} is missing");
        return File.ReadAllText(full);
    }

    /// <summary>An entry's chosen files, empty, in a scratch store: which paths exist is the point.</summary>
    private sealed class FakeInstall : IDisposable
    {
        private readonly string _root = Path.Combine(Path.GetTempPath(), "tensoragent-h3-install-" + Guid.NewGuid().ToString("N"));
        private readonly CatalogModel _model;

        public FakeInstall(CatalogModel model, Func<CatalogModel, IEnumerable<CatalogFile>> install)
        {
            _model = model;
            Store = new ModelStore(_root);
            Directory.CreateDirectory(Store.DirectoryFor(model));
            foreach (CatalogFile file in install(model))
                File.WriteAllBytes(Store.PathFor(model, file), []);
        }

        public ModelStore Store { get; }

        public string PathOf(CatalogFileRole role) =>
            Store.PathFor(_model, _model.Files.Single(f => f.Role == role));

        public void Dispose()
        {
            DiffusionCompanions.Publish(null, Store);
            try { Directory.Delete(_root, true); } catch (Exception) { /* scratch */ }
        }
    }
}
