using System.Net;
using System.Collections;
using System.Security.Cryptography;
using TensorSharp.Runtime;
using TensorSharp;
using TensorSharp.Models;
using System.Net.Http.Json;
using System.Text.Json;
using Microsoft.Extensions.Logging.Abstractions;
using TensorAgent.Core.Catalog;
using TensorAgent.Core.Downloads;
using TensorAgent.Core.Hosting;
using TensorAgent.Core.Localization;
using TensorAgent.Core.Sessions;
using TensorAgent.Core.Settings;
using TensorAgent.Sharing.Localization;
using TensorSharp.Server;

namespace TensorAgent.Tests;

[Collection(ProcessEnvironmentCollection.Name)]
public sealed class Ltx25CatalogTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "tensoragent-ltx25-" + Guid.NewGuid().ToString("N"));
    private readonly Dictionary<string,string?> _environment = Environment.GetEnvironmentVariables().Cast<DictionaryEntry>().ToDictionary(e=>(string)e.Key,e=>(string?)e.Value,StringComparer.Ordinal);
    private static CatalogModel Entry => ModelCatalog.Find("ltx-2.5-distilled-q4kxl")!;

    // A synthetic unavailable card uses an existing localized catalog value to
    // exercise the gate without retaining an obsolete LTX-unavailable message.
    private static CatalogModel Unavailable => Entry with { UnavailableReason = "catalog.model.ltx25.notes" };
    private static CatalogModel Available => Entry with { UnavailableReason = null };

    public void Dispose()
    {
        foreach(string key in Environment.GetEnvironmentVariables().Keys.Cast<string>().Except(_environment.Keys)) Environment.SetEnvironmentVariable(key,null);
        foreach(var pair in _environment) Environment.SetEnvironmentVariable(pair.Key,pair.Value);
        KvCacheDtypeConfig.ConfigureFromEnvironment();
        Loc.Use(UiLanguages.English);
        try { if (Directory.Exists(_root)) Directory.Delete(_root, true); } catch (IOException) { }
    }

    [Fact]
    public void ExperimentalEntryPinsTheAuthenticFourFileSet()
    {
        CatalogModel model = Assert.Single(ModelCatalog.BuiltIn, m => m.Family == CatalogFamily.Ltx25);
        Assert.Equal("LTX 2.5 Distilled", model.DisplayName);
        Assert.Equal(CatalogArchitectureKind.Diffusion, model.Kind);
        Assert.True(model.Experimental);
        Assert.True(model.IsAvailable);
        Assert.True(model.IsVideoGenerator);
        Assert.False(model.IsImageGenerator);
        Assert.Equal(48, model.MinDeviceMemoryGB);
        Assert.NotEmpty(model.Notes!);
        Assert.Equal(43_077_008_276, model.TotalBytes);
        Assert.Equal(5, model.Files.Count);
        Assert.All(model.Files, file => Assert.False(file.Optional));
        Assert.Equal("55442750eb0013d94fffaa4dadb577c5f4dc07107ec04389d41444f6e517d980", model.Weights.Sha256);
        // Unsloth's ungated repository, the distilled checkpoint from its distilled/ folder (its root
        // holds the dev checkpoint the engine refuses), and every companion from the same revision.
        Assert.StartsWith("https://huggingface.co/unsloth/LTX-2.5-GGUF/resolve/ac3c540c6025903784336e99b2dc179385c4c3b7/distilled/", model.Weights.Url);
        Assert.Contains("distilled", model.Weights.FileName);
        Assert.Equal(new[] { CatalogFileRole.TextEncoder, CatalogFileRole.Vae, CatalogFileRole.AudioVae, CatalogFileRole.LatentUpscaler },
            model.Files.Skip(1).Select(file => file.Role));
        Assert.All(model.Files.Skip(1), file => Assert.StartsWith("https://huggingface.co/unsloth/LTX-2.5-GGUF/resolve/ac3c540c6025903784336e99b2dc179385c4c3b7/", file.Url));
        Assert.Contains("with-proj", model.Files.Single(file => file.Role == CatalogFileRole.TextEncoder).FileName);
        Assert.Contains("video-vae-conv", model.Files.Single(file => file.Role == CatalogFileRole.Vae).FileName);
        // The two-stage recipe's x2 spatial (not temporal) latent upscaler.
        Assert.Contains("spatial-upscaler-x2", model.Files.Single(file => file.Role == CatalogFileRole.LatentUpscaler).FileName);
        // The authentic custom Gemma file embeds its tokenizer assets.
        Assert.DoesNotContain(model.Files, file => file.Role == CatalogFileRole.Tokenizer);
    }

    [Fact]
    public void CatalogPinsMatchTheShippedLtxPreset()
    {
        var directory = new DirectoryInfo(AppContext.BaseDirectory);
        while (directory is not null && !File.Exists(Path.Combine(directory.FullName, "TensorSharp.slnx"))) directory = directory.Parent;
        Assert.NotNull(directory);
        using JsonDocument preset = JsonDocument.Parse(File.ReadAllText(Path.Combine(directory!.FullName, "config", "ltx-2.5-distilled.json")),
            new JsonDocumentOptions { CommentHandling = JsonCommentHandling.Skip });

        foreach ((CatalogFileRole role, string key) in new[]
        {
            (CatalogFileRole.Weights, "model"), (CatalogFileRole.TextEncoder, "video-text-encoder"),
            (CatalogFileRole.Vae, "video-vae"), (CatalogFileRole.AudioVae, "audio-vae"),
            (CatalogFileRole.LatentUpscaler, "video-upscaler"),
        })
        {
            CatalogFile file = Entry.Files.Single(file => file.Role == role);
            JsonElement configured = preset.RootElement.GetProperty(key);
            Assert.Equal(file.FileName, Path.GetFileName(configured.GetProperty("path").GetString()));
            Assert.Equal(file.Sha256, configured.GetProperty("sha256").GetString());
            Assert.Equal(file.Url, configured.GetProperty("urls")[0].GetString()!.Replace("${hf}", "https://huggingface.co", StringComparison.Ordinal));
        }
    }

    [Fact]
    public void UnavailableCloneIsVisibleButExcludedFromCompatibleModels()
    {
        CatalogModel model = Unavailable;
        CatalogModel[] catalog = ModelCatalog.BuiltIn.Select(m => m.Id == model.Id ? model : m).ToArray();
        var installed = new HashSet<string>([model.Id]);
        var all = ModelBrowser.Browse(catalog, "LTX 2.5", ModelBrowseFilter.All, 1024, installed, model.Id);
        ModelFamilyGroup group = Assert.Single(all);
        Assert.Equal("ltx", group.Id);
        Assert.Same(model, Assert.Single(group.Models));
        Assert.Empty(ModelBrowser.Search(catalog, "LTX", ModelBrowseFilter.Compatible, 1024, installed, model.Id));
        Assert.Contains(model, ModelBrowser.Search(catalog, "LTX", ModelBrowseFilter.Downloaded, 8, installed, model.Id));
        Assert.All(ModelCatalog.BuiltIn.Where(m => m.Id != model.Id), m => Assert.True(m.IsAvailable));
    }

    [Fact]
    public void AvailabilityReasonFollowsTheInterfaceLanguage()
    {
        string? english = Unavailable.UnavailableReason;
        foreach (UiLanguage language in UiLanguages.Supported)
        {
            Loc.Use(language);
            Assert.False(Unavailable.IsAvailable);
            Assert.NotEmpty(Unavailable.UnavailableReason!);
            Assert.DoesNotContain("catalog.", Unavailable.UnavailableReason!);
            Assert.DoesNotContain("catalog.", Entry.Notes!);
        }
        Loc.Use(UiLanguages.Find("zh-Hans")!);
        Assert.NotEqual(english, Unavailable.UnavailableReason);
    }

    [Fact]
    public void SelectionIsRefusedBeforeChangingSettingsOrCallingTheEngine()
    {
        var paths = new AgentPaths(Path.Combine(_root, "data"), Path.Combine(_root, "cache")) { DeviceMemoryGB = 128 };
        int loads = 0;
        var service = new ModelService(NullLogger<ModelService>.Instance, (_, _, _, _) =>
        {
            ++loads;
            throw new Xunit.Sdk.XunitException("Unavailable selection must not reach the model loader.");
        });
        using var host = new AgentAppHost(paths, modelService: service, backends: [new BackendOption("cpu", "CPU")]);
        host.Settings.Update(settings => { settings.SelectedModelId = "qwen3.5-9b-iq4xs"; return settings; });
        AgentAppHost.ModelLoadState state = host.ModelLoad;
        var error = Assert.Throws<NotSupportedException>(() => host.UseModel(Unavailable, warmAfterwards: false));
        Assert.Equal(Unavailable.UnavailableReason, error.Message);
        Assert.Equal("qwen3.5-9b-iq4xs", host.Settings.Load().SelectedModelId);
        Assert.Equal(state, host.ModelLoad);
        Assert.Equal(0, loads);
        Assert.False(Directory.Exists(host.Models.DirectoryFor(Entry)));
    }

    [Fact]
    public void AvailableEntryAppearsInCompatibleVideoCatalogAtItsMemoryTier()
    {
        CatalogModel model=Available;
        Assert.True(model.IsAvailable);
        Assert.True(model.IsVideoGenerator);
        Assert.True(model.Modalities.HasFlag(CatalogModalities.AudioOutput));
        Assert.Contains(model,ModelBrowser.Search([model],"LTX",ModelBrowseFilter.Compatible,48,new HashSet<string>(),null));
        Assert.Empty(ModelBrowser.Search([model],"LTX",ModelBrowseFilter.Compatible,12,new HashSet<string>(),null));
    }

    [Fact]
    public async Task AvailableFixtureDownloadsAllFiveFilesAndPublishesThemBeforeActivation()
    {
        byte[] bytes=new byte[32];
        BitConverter.GetBytes(0x46554747u).CopyTo(bytes,0);
        BitConverter.GetBytes(3u).CopyTo(bytes,4);
        using var files=new RangeServer(bytes);
        string hash=Convert.ToHexStringLower(SHA256.HashData(bytes));
        CatalogModel model=Available with {Files=Entry.Files.Select(f=>f with{Url=files.Url,Bytes=bytes.Length,Sha256=hash}).ToArray()};
        int loads=0;string? loadedPath=null;
        var published=new Dictionary<string,string?>();
        var service=new ModelService(NullLogger<ModelService>.Instance,(path,_,_,_)=>
        {
            loads++;loadedPath=path;
            foreach(string variable in new[]{"TS_VIDEO_TEXT_ENCODER","TS_VIDEO_VAE","TS_VIDEO_AUDIO_VAE","TS_VIDEO_UPSCALER"})published[variable]=Environment.GetEnvironmentVariable(variable);
            throw new InvalidDataException("Fixture reached the model loader; production inference is covered by the real E2E probe.");
        });
        var paths=new AgentPaths(Path.Combine(_root,"data"),Path.Combine(_root,"cache")){DeviceMemoryGB=128};
        using var host=new AgentAppHost(paths,modelService:service,backends:[new BackendOption("cpu","CPU")]);
        await host.Models.DownloadAsync(model,null,CancellationToken.None);
        Assert.Equal(5,files.Requests);
        Assert.Equal(InstallState.Installed,host.Models.StateOf(model));
        foreach(var file in model.Files)Assert.Equal(bytes,File.ReadAllBytes(host.Models.PathFor(model,file)));
        var error=Assert.Throws<InvalidOperationException>(()=>host.UseModel(model,warmAfterwards:false));
        Assert.Contains("Fixture reached the model loader",error.Message);
        Assert.Equal(1,loads);
        Assert.Equal(host.Models.PathFor(model,model.Weights),loadedPath);
        Assert.Equal(host.Models.PathFor(model,model.Files.Single(f=>f.Role==CatalogFileRole.TextEncoder)),published["TS_VIDEO_TEXT_ENCODER"]);
        Assert.Equal(host.Models.PathFor(model,model.Files.Single(f=>f.Role==CatalogFileRole.Vae)),published["TS_VIDEO_VAE"]);
        Assert.Equal(host.Models.PathFor(model,model.Files.Single(f=>f.Role==CatalogFileRole.AudioVae)),published["TS_VIDEO_AUDIO_VAE"]);
        Assert.Equal(host.Models.PathFor(model,model.Files.Single(f=>f.Role==CatalogFileRole.LatentUpscaler)),published["TS_VIDEO_UPSCALER"]);
        Assert.Null(Environment.GetEnvironmentVariable("TS_VIDEO_TOKENIZER"));
        DiffusionCompanions.Publish(null,host.Models);
        Assert.All(published.Keys,k=>Assert.Null(Environment.GetEnvironmentVariable(k)));
    }

    [Fact]
    public void IncompleteAvailableInstallationIsRefusedBeforeCallingTheLoader()
    {
        int loads=0;
        var service=new ModelService(NullLogger<ModelService>.Instance,(_,_,_,_)=>{loads++;throw new InvalidOperationException("must not load");});
        var paths=new AgentPaths(Path.Combine(_root,"data"),Path.Combine(_root,"cache")){DeviceMemoryGB=128};
        using var host=new AgentAppHost(paths,modelService:service,backends:[new BackendOption("cpu","CPU")]);
        Directory.CreateDirectory(host.Models.DirectoryFor(Available));
        // A truncated primary file does not make the complete companion set ready.
        File.WriteAllBytes(host.Models.PathFor(Available,Available.Weights),new byte[32]);
        Assert.Throws<FileNotFoundException>(()=>host.UseModel(Available,warmAfterwards:false));
        Assert.Equal(0,loads);
        Assert.False(host.ModelService.IsLoaded);
    }

    [Fact]
    public void AvailablePinnedInstallationCanUseIsolatedSymbolicLinks()
    {
        string source=Path.Combine(_root,"source.bin");Directory.CreateDirectory(_root);File.WriteAllBytes(source,new byte[32]);
        CatalogModel model=Available with {Files=Entry.Files.Select(f=>f with{Bytes=32}).ToArray()};
        var store=new ModelStore(Path.Combine(_root,"models"));Directory.CreateDirectory(store.DirectoryFor(model));
        foreach(var file in model.Files)File.CreateSymbolicLink(store.PathFor(model,file),source);
        Assert.Equal(InstallState.Installed,store.StateOf(model));
        Assert.Equal(4,DiffusionCompanions.Publish(model,store).Count);
    }

    [Fact]
    public async Task UnavailableDownloadOrImportCannotCreateAJobFileOrNetworkRequest()
    {
        using var files = new RangeServer(new byte[4096]);
        CatalogModel model = Unavailable with { Files = [Entry.Weights with { Url = files.Url, Bytes = 4096 }] };
        var store = new ModelStore(Path.Combine(_root, "weights"));
        using var manager = new ModelDownloadManager(store);
        Assert.Throws<NotSupportedException>(() => manager.Start(model));
        await Assert.ThrowsAsync<NotSupportedException>(() => store.DownloadAsync(model, null, CancellationToken.None));
        using var source = new MemoryStream(new byte[4096]);
        await Assert.ThrowsAsync<NotSupportedException>(() => store.ImportAsync(model with { SideloadOnly = true }, source));
        Assert.Empty(manager.All);
        Assert.False(Directory.Exists(store.DirectoryFor(model)));
        Assert.Equal(0, files.Requests);
    }

    [Fact]
    public async Task CatalogApiStatesUnavailabilityAndRejectsDownloadBeforeAnyTransfer()
    {
        using var files = new RangeServer(new byte[4096]);
        CatalogModel model = Unavailable with { Files = [Entry.Weights with { Url = files.Url, Bytes = 4096 }] };
        var store = new ModelStore(Path.Combine(_root, "weights"));
        using var manager = new ModelDownloadManager(store);
        using var server = new LoopbackServer(NullLogger.Instance);
        server.MapAgent([model], store, new ConversationStore(Path.Combine(_root, "conversations")),
            new SettingsStore(Path.Combine(_root, "settings.json")), downloads: manager);
        server.Start();
        using var client = new HttpClient { BaseAddress = new Uri(server.BaseUrl) };
        client.DefaultRequestHeaders.Add("Cookie", $"{LoopbackServer.TokenCookie}={server.Token}");
        JsonElement listing = JsonSerializer.Deserialize<JsonElement>(await client.GetStringAsync($"/api/agent/catalog/{model.Id}"));
        Assert.False(listing.GetProperty("available").GetBoolean());
        Assert.Equal(model.UnavailableReason, listing.GetProperty("unavailableReason").GetString());
        using HttpResponseMessage response = await client.PostAsync($"/api/agent/catalog/{model.Id}/download", null);
        Assert.Equal(HttpStatusCode.Conflict, response.StatusCode);
        JsonElement error = JsonSerializer.Deserialize<JsonElement>(await response.Content.ReadAsStringAsync());
        Assert.Equal(model.UnavailableReason, error.GetProperty("error").GetString());
        Assert.Empty(manager.All);
        Assert.False(Directory.Exists(store.DirectoryFor(model)));
        Assert.Equal(0, files.Requests);
    }

    [Fact]
    public async Task SettingsPostCannotPersistAnExplicitUnavailableSelection()
    {
        var store = new ModelStore(Path.Combine(_root, "weights"));
        string path = Path.Combine(_root, "settings.json");
        var settings = new SettingsStore(path);
        settings.Update(current => { current.SelectedModelId = "qwen3.5-9b-iq4xs"; current.MaxTokens = 1024; return current; });
        byte[] before = File.ReadAllBytes(path);
        int applied = 0;
        using var server = new LoopbackServer(NullLogger.Instance);
        // A synthetic unavailable card keeps generic readiness behavior covered
        // after the real LTX pipeline has been enabled.
        server.MapAgent([Unavailable], store, new ConversationStore(Path.Combine(_root, "conversations")),
            settings, onSettingsChanged: _ => ++applied);
        server.Start();
        using var client = new HttpClient { BaseAddress = new Uri(server.BaseUrl) };
        client.DefaultRequestHeaders.Add("Cookie", $"{LoopbackServer.TokenCookie}={server.Token}");
        using HttpResponseMessage response = await client.PostAsJsonAsync("/api/agent/settings",
            new { selectedModelId = Entry.Id, maxTokens = 8192 });
        Assert.Equal(HttpStatusCode.Conflict, response.StatusCode);
        Assert.Equal(before, File.ReadAllBytes(path));
        Assert.Equal("qwen3.5-9b-iq4xs", settings.Load().SelectedModelId);
        Assert.Equal(1024, settings.Load().MaxTokens);
        Assert.Equal(0, applied);
    }
}
