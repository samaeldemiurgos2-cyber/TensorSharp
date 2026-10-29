// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Server;
using TensorSharp.Server.Endpoints;

namespace InferenceWeb.Tests;

/// <summary>
/// <see cref="LoraAdapterDirectory"/>, the confinement behind <c>POST /v1/loras</c> and the
/// <c>--loras-dir</c> startup load (PR #282 review item 5). Lexical escapes (<c>..</c>, rooted,
/// drive, UNC, NUL, <c>~</c>, an NTFS stream), the canonical sibling-prefix escape
/// (<c>loras-evil</c>), and link escapes through a symlink or, on Windows, a junction. The
/// link cases skip where the process cannot create that kind of link (file symlinks need
/// Developer Mode or elevation on Windows; directory links fall back to a junction there,
/// which needs no privilege).
/// </summary>
public sealed class LoraAdapterDirectoryTests : IDisposable
{
    private readonly string _base = Directory.CreateTempSubdirectory("ts-lora-dir-").FullName;
    private readonly string _root;
    private readonly string _outside;
    private readonly string _sibling;
    private readonly LoraAdapterDirectory _dir;

    public LoraAdapterDirectoryTests()
    {
        _root = Path.Combine(_base, "loras");
        _outside = Path.Combine(_base, "outside");
        _sibling = _root + "-evil";
        foreach (string d in new[] { _root, Path.Combine(_root, "sub"), _outside, _sibling })
            Directory.CreateDirectory(d);
        LoraFixtures.WriteAdapter(Path.Combine(_root, "a.safetensors"));
        LoraFixtures.WriteAdapter(Path.Combine(_root, "sub", "b.safetensors"));
        LoraFixtures.WriteAdapter(Path.Combine(_base, "x.safetensors"));
        LoraFixtures.WriteAdapter(Path.Combine(_outside, "secret.safetensors"));
        LoraFixtures.WriteAdapter(Path.Combine(_sibling, "a.safetensors"));
        File.WriteAllText(Path.Combine(_outside, "config.json"), "{\"r\":1,\"lora_alpha\":1}");
        _dir = new LoraAdapterDirectory(_root);
    }

    public void Dispose() => LoraLinks.DeleteTree(_base);

    // ---- accepted --------------------------------------------------------------------

    [Fact]
    public void TryResolveWeights_RelativeFileInsideRoot_ReturnsCanonicalFullPath()
    {
        Assert.True(_dir.TryResolveWeights("a.safetensors", out string? full, out string? relative, out string? why), why);
        Assert.Equal(Path.Combine(_root, "a.safetensors"), full);
        Assert.Equal("a.safetensors", relative);
    }

    [Theory]
    [InlineData("sub/b.safetensors")]
    [InlineData("sub\\b.safetensors")]
    [InlineData("./sub/b.safetensors")]
    [InlineData("sub/./b.safetensors")]
    [InlineData("SUB/b.SAFETENSORS")]
    public void TryResolveWeights_NestedSpellings_ResolveToTheSameFile(string requested)
    {
        // Case only matters where the file system is case-sensitive.
        if (requested.StartsWith("SUB", StringComparison.Ordinal) && OperatingSystem.IsLinux())
            return;
        Assert.True(_dir.TryResolveWeights(requested, out string? full, out _, out string? why), why);
        Assert.Equal(Path.Combine(_root, "sub", "b.safetensors"), full, ignoreCase: !OperatingSystem.IsLinux());
    }

    [Fact]
    public void TryResolveWeights_AbsolutePathInsideRoot_IsAcceptedInItsRelativeForm()
    {
        string absolute = Path.Combine(_root, "sub", "b.safetensors");
        Assert.True(_dir.TryResolveWeights(absolute, out string? full, out string? relative, out string? why), why);
        Assert.Equal(absolute, full);
        Assert.Equal("sub/b.safetensors", relative);
    }

    [Fact]
    public void TryResolveWeights_GgufAdapter_IsAcceptedByName()
    {
        File.WriteAllBytes(Path.Combine(_root, "c.gguf"), new byte[] { 0x47, 0x47, 0x55, 0x46 });
        Assert.True(_dir.TryResolveWeights("c.gguf", out _, out string? relative, out string? why), why);
        Assert.Equal("c.gguf", relative);
    }

    // ---- refused -----------------------------------------------------------------------

    [Theory]
    [InlineData("../x.safetensors")]
    [InlineData("..\\x.safetensors")]
    [InlineData("sub/../../x.safetensors")]
    [InlineData("sub\\..\\..\\x.safetensors")]
    [InlineData("../outside/secret.safetensors")]
    [InlineData("../loras-evil/a.safetensors")]
    [InlineData("sub/../a.safetensors")]    // stays inside, but any ".." is refused outright
    public void TryResolveWeights_Traversal_IsRefused(string requested)
    {
        Assert.False(_dir.TryResolveWeights(requested, out string? full, out string? relative, out _));
        Assert.Null(full);
        Assert.Null(relative);
    }

    [Fact]
    public void TryResolveWeights_AbsoluteOutsideAndSiblingPrefix_AreRefused()
    {
        // The sibling directory's full path begins with the root's own characters, which a
        // plain StartsWith containment check would wave through.
        Assert.True(File.Exists(Path.Combine(_sibling, "a.safetensors")));
        Assert.False(_dir.TryResolveWeights(Path.Combine(_sibling, "a.safetensors"), out _, out _, out _));
        Assert.False(_dir.TryResolveWeights(Path.Combine(_outside, "secret.safetensors"), out _, out _, out _));
        Assert.False(_dir.TryResolveWeights(Path.Combine(_base, "x.safetensors"), out _, out _, out _));
        Assert.False(_dir.TryResolveWeights(_root, out _, out _, out _));
    }

    [Theory]
    [InlineData("/etc/passwd")]
    [InlineData("/a.safetensors")]
    [InlineData("C:\\Windows\\win.ini")]
    [InlineData("C:a.safetensors")]
    [InlineData("\\\\server\\share\\x.safetensors")]
    [InlineData("//server/share/x.safetensors")]
    [InlineData("a\u0000b.safetensors")]
    [InlineData("a.safetensors\u0000")]
    [InlineData("a.safetensors:s")]
    [InlineData("a.safetensors::$DATA")]
    [InlineData("~/x.safetensors")]
    [InlineData("a.safetensors.")]
    [InlineData("")]
    [InlineData("   ")]
    [InlineData(".")]
    [InlineData(null)]
    public void TryResolveWeights_RootedUncDriveNulColonAndEmpty_AreRefused(string? requested)
    {
        Assert.False(_dir.TryResolveWeights(requested, out string? full, out _, out _));
        Assert.Null(full);
    }

    [Theory]
    [InlineData("missing.safetensors")]
    [InlineData("sub")]
    [InlineData("a.json")]
    [InlineData("a.bin")]
    [InlineData("dir.safetensors")]
    public void TryResolveWeights_MissingWrongExtensionOrDirectory_AreRefused(string requested)
    {
        File.WriteAllText(Path.Combine(_root, "a.json"), "{}");
        File.WriteAllBytes(Path.Combine(_root, "a.bin"), new byte[16]);
        Directory.CreateDirectory(Path.Combine(_root, "dir.safetensors"));
        Assert.False(_dir.TryResolveWeights(requested, out _, out _, out _));
    }

    [Fact]
    public void TryResolveConfig_ExplicitInsideAndOutside()
    {
        File.WriteAllText(Path.Combine(_root, "sub", "cfg.json"), "{}");

        Assert.True(_dir.TryResolveConfig("sub/cfg.json", "a.safetensors", out string? full, out string? relative, out string? why), why);
        Assert.Equal(Path.Combine(_root, "sub", "cfg.json"), full);
        Assert.Equal("sub/cfg.json", relative);

        Assert.False(_dir.TryResolveConfig(Path.Combine(_outside, "config.json"), "a.safetensors", out _, out _, out _));
        Assert.False(_dir.TryResolveConfig("../outside/config.json", "a.safetensors", out _, out _, out _));
        // A weights file is not a config.
        Assert.False(_dir.TryResolveConfig("a.safetensors", "a.safetensors", out _, out _, out _));
        Assert.False(_dir.TryResolveConfig("missing.json", "a.safetensors", out _, out _, out _));
    }

    [Fact]
    public void TryResolveConfig_SiblingAdapterConfig_IsFoundBesideTheWeights_OrIsNone()
    {
        Assert.True(_dir.TryResolveConfig(null, "sub/b.safetensors", out string? full, out _, out _));
        Assert.Null(full);

        File.WriteAllText(Path.Combine(_root, "sub", "adapter_config.json"), "{}");
        Assert.True(_dir.TryResolveConfig(null, "sub/b.safetensors", out full, out string? relative, out _));
        Assert.Equal(Path.Combine(_root, "sub", "adapter_config.json"), full);
        Assert.Equal("sub/adapter_config.json", relative);

        // The top-level weights do not see the subdirectory's config, and a directory with
        // that name is not a config.
        Directory.CreateDirectory(Path.Combine(_root, "adapter_config.json"));
        Assert.True(_dir.TryResolveConfig(null, "a.safetensors", out full, out _, out _));
        Assert.Null(full);
    }

    [Theory]
    [InlineData("math-v2", true)]
    [InlineData("a.b_c", true)]
    [InlineData("A1", true)]
    [InlineData("x", true)]
    [InlineData("nonesuch", true)]
    [InlineData("", false)]
    [InlineData(null, false)]
    [InlineData("none", false)]
    [InlineData("NONE", false)]
    [InlineData("None", false)]
    [InlineData("../x", false)]
    [InlineData("a/b", false)]
    [InlineData("a\\b", false)]
    [InlineData("a b", false)]
    [InlineData(".hidden", false)]
    [InlineData("-x", false)]
    [InlineData("a\n", false)]
    [InlineData("a%2Fb", false)]
    public void IsValidName(string? name, bool valid)
    {
        Assert.Equal(valid, LoraAdapterDirectory.IsValidName(name));
    }

    [Fact]
    public void IsValidName_CapsTheLengthAt64()
    {
        Assert.True(LoraAdapterDirectory.IsValidName(new string('a', 64)));
        Assert.False(LoraAdapterDirectory.IsValidName(new string('a', 65)));
    }

    [Fact]
    public void Sanitize_RewritesResolvedPathsAndDropsTheDirectory_InEitherSeparatorForm()
    {
        string full = Path.Combine(_root, "sub", "b.safetensors");
        string message = $"LoRA file '{full}' is bad; also '{full.Replace('\\', '/')}', '{Path.Combine(_root, "other.bin")}' and '{_root}'.";

        string clean = _dir.Sanitize(message, full, "sub/b.safetensors", null, null);

        Assert.Contains("'sub/b.safetensors'", clean);
        Assert.Contains("'other.bin'", clean);
        LoraFixtures.AssertNoPath(clean, _root);
        LoraFixtures.AssertNoPath(clean, _base);
    }

    // ---- the one answer every resolution failure gets ---------------------------------

    [Fact]
    public async Task LoadAsync_EveryResolutionFailure_GetsTheSameMessage_AndRegistersNothing()
    {
        using var service = new ModelService(NullLogger<ModelService>.Instance);
        string?[] requests =
        {
            "missing.safetensors", Path.Combine(_outside, "secret.safetensors"), "../x.safetensors",
            "sub", "a.json", Path.Combine(_sibling, "a.safetensors"), "/etc/passwd", "a.safetensors:s",
        };
        var messages = new List<string?>();
        foreach (string? path in requests)
        {
            LoraLoadResult result = await _dir.LoadAsync(service, "x", path, null, null);
            Assert.Null(result.Adapter);
            Assert.Equal(400, result.StatusCode);
            Assert.Equal("path", result.Param);
            messages.Add(result.Message);
        }

        Assert.Single(messages.Distinct());
        Assert.Equal(LoraAdapterDirectory.ResolutionRefusal, messages[0]);
        Assert.DoesNotContain("not found", messages[0], StringComparison.OrdinalIgnoreCase);
        LoraFixtures.AssertNoPath(messages[0]!, _root);
        Assert.Equal(0, service.LoraRegistry.Count);
    }

    [Fact]
    public async Task LoadAsync_ExplicitConfigThatDoesNotResolve_GetsTheSameMessage()
    {
        using var service = new ModelService(NullLogger<ModelService>.Instance);
        foreach (string config in new[] { Path.Combine(_outside, "config.json"), "../outside/config.json", "missing.json", "a.safetensors" })
        {
            LoraLoadResult result = await _dir.LoadAsync(service, "x", "a.safetensors", config, null);
            Assert.Equal(400, result.StatusCode);
            Assert.Equal("config", result.Param);
            Assert.Equal(LoraAdapterDirectory.ResolutionRefusal, result.Message);
        }
        Assert.Equal(0, service.LoraRegistry.Count);
    }

    [Fact]
    public async Task LoadAsync_Loads_AndRefusesPastTheLimit_ButStillReloads()
    {
        using var service = new ModelService(NullLogger<ModelService>.Instance);
        for (int i = 0; i < LoraAdapterDirectory.MaxAdapters; i++)
        {
            LoraLoadResult ok = await _dir.LoadAsync(service, "n" + i, "a.safetensors", null, null);
            Assert.NotNull(ok.Adapter);
        }

        LoraLoadResult full = await _dir.LoadAsync(service, "one-more", "a.safetensors", null, null);
        Assert.Equal(409, full.StatusCode);
        Assert.Equal("lora_limit_reached", full.Code);
        Assert.False(service.LoraRegistry.Contains("one-more"));

        LoraLoadResult reload = await _dir.LoadAsync(service, "n0", "sub/b.safetensors", null, 0.5f);
        Assert.NotNull(reload.Adapter);
        Assert.Equal(LoraAdapterDirectory.MaxAdapters, service.LoraRegistry.Count);
    }

    // ---- links -------------------------------------------------------------------------

    [LoraFileSymlinkFact]
    public void FileSymlinkLeavingRoot_IsRefused()
    {
        Assert.True(LoraLinks.TryLinkFile(Path.Combine(_root, "leak.safetensors"), Path.Combine(_outside, "secret.safetensors")));
        Assert.False(_dir.TryResolveWeights("leak.safetensors", out _, out _, out string? why));
        Assert.Contains("symbolic link", why);
    }

    [LoraFileSymlinkFact]
    public void FileSymlinkInsideRoot_IsAccepted_AndResolvesToItsTarget()
    {
        Assert.True(LoraLinks.TryLinkFile(Path.Combine(_root, "alias.safetensors"), Path.Combine(_root, "sub", "b.safetensors")));
        Assert.True(_dir.TryResolveWeights("alias.safetensors", out string? full, out string? relative, out string? why), why);
        Assert.Equal(Path.Combine(_root, "sub", "b.safetensors"), full);
        Assert.Equal("alias.safetensors", relative);
    }

    [LoraFileSymlinkFact]
    public void SiblingAdapterConfigLinkLeavingRoot_RefusesTheLoad()
    {
        // Without a check of its own, the loader's sibling lookup would read a config the
        // guard never saw.
        Assert.True(LoraLinks.TryLinkFile(Path.Combine(_root, "sub", "adapter_config.json"), Path.Combine(_outside, "config.json")));
        Assert.False(_dir.TryResolveConfig(null, "sub/b.safetensors", out _, out _, out string? why));
        Assert.Contains("symbolic link", why);
    }

    [LoraDirectoryLinkFact]
    public void DirectoryLinkLeavingRoot_IsRefused()
    {
        Assert.True(LoraLinks.TryLinkDirectory(Path.Combine(_root, "link"), _outside));
        Assert.True(File.Exists(Path.Combine(_root, "link", "secret.safetensors")));
        Assert.False(_dir.TryResolveWeights("link/secret.safetensors", out _, out _, out string? why));
        Assert.Contains("symbolic link", why);
    }

    [LoraDirectoryLinkFact]
    public void DirectoryLinkIntoTheSiblingPrefixDirectory_IsRefused()
    {
        Assert.True(LoraLinks.TryLinkDirectory(Path.Combine(_root, "evil"), _sibling));
        Assert.False(_dir.TryResolveWeights("evil/a.safetensors", out _, out _, out _));
    }

    [LoraDirectoryLinkFact]
    public void DirectoryLinkInsideRoot_IsAccepted()
    {
        Assert.True(LoraLinks.TryLinkDirectory(Path.Combine(_root, "alias"), Path.Combine(_root, "sub")));
        Assert.True(_dir.TryResolveWeights("alias/b.safetensors", out string? full, out string? relative, out string? why), why);
        Assert.Equal(Path.Combine(_root, "sub", "b.safetensors"), full);
        Assert.Equal("alias/b.safetensors", relative);
    }

    [LoraDirectoryLinkFact]
    public void RootItselfALink_ResolvesInsideItsTarget_AndKeepsBothOutOfMessages()
    {
        string linkRoot = Path.Combine(_base, "loras-link");
        Assert.True(LoraLinks.TryLinkDirectory(linkRoot, _root));
        var dir = new LoraAdapterDirectory(linkRoot);

        Assert.True(dir.TryResolveWeights("sub/b.safetensors", out string? full, out _, out string? why), why);
        Assert.Equal(Path.Combine(_root, "sub", "b.safetensors"), full);
        Assert.True(dir.TryResolveWeights(Path.Combine(linkRoot, "a.safetensors"), out _, out _, out why), why);
        Assert.False(dir.TryResolveWeights("../x.safetensors", out _, out _, out _));

        string clean = dir.Sanitize($"bad '{full}' under '{_root}' and '{linkRoot}'", full!, "sub/b.safetensors", null, null);
        LoraFixtures.AssertNoPath(clean, _base);
    }

    // ---- startup load ------------------------------------------------------------------

    [Fact]
    public void Preload_LoadsTopLevelAdapters_AndReportsTheRestWithoutThrowing()
    {
        string root = Path.Combine(_base, "preload");
        Directory.CreateDirectory(Path.Combine(root, "nested"));
        LoraFixtures.WriteAdapter(Path.Combine(root, "good1.safetensors"));
        LoraFixtures.WriteAdapter(Path.Combine(root, "good2.SAFETENSORS"));
        LoraFixtures.WriteAdapter(Path.Combine(root, "nested", "inner.safetensors"));   // not top level
        LoraFixtures.WriteAdapter(Path.Combine(root, "bad name.safetensors"));          // not a valid name
        File.WriteAllBytes(Path.Combine(root, "junk.safetensors"), new byte[] { 1, 2, 3, 4 });
        File.WriteAllBytes(Path.Combine(root, "later.gguf"), new byte[] { 0x47, 0x47, 0x55, 0x46 });
        File.WriteAllText(Path.Combine(root, "notes.txt"), "not an adapter");
        // A file name and a .gguf of the same stem: the (broken) .gguf is refused first, so
        // the .safetensors still loads under that name.
        File.WriteAllBytes(Path.Combine(root, "twin.gguf"), new byte[] { 0x47, 0x47, 0x55, 0x46 });
        LoraFixtures.WriteAdapter(Path.Combine(root, "twin.safetensors"));
        var logger = new LoraRecordingLogger();
        using var service = new ModelService(NullLogger<ModelService>.Instance);

        LoraPreloadReport report = new LoraAdapterDirectory(root).Preload(service, logger);

        Assert.Equal(new[] { "good1", "good2", "twin" }, service.LoraRegistry.List().Select(a => a.Name));
        Assert.Equal(new[] { "good1", "good2", "twin" }, report.Loaded.OrderBy(n => n, StringComparer.Ordinal));
        Assert.Equal(
            new[] { "bad name.safetensors", "junk.safetensors", "later.gguf", "twin.gguf" },
            report.Refused.Select(r => r.File).OrderBy(f => f, StringComparer.Ordinal));
        Assert.Contains("truncated", report.Refused.Single(r => r.File == "junk.safetensors").Reason);
        Assert.Contains("truncated", report.Refused.Single(r => r.File == "later.gguf").Reason);
        foreach (var (_, reason) in report.Refused)
            LoraFixtures.AssertNoPath(reason, root);
        // One warning per refused file, and the summary line.
        Assert.Equal(4, logger.Entries.Count(e => e.Level == LogLevel.Warning));
        Assert.Contains(logger.Entries, e => e.Level == LogLevel.Information && e.Message.Contains("3 adapter(s) loaded at startup, 4 refused"));
    }

    [Fact]
    public void Preload_WithAModelLoaded_SkipsAnAdapterItCannotBind_AndStillStarts()
    {
        // Bound against the loaded model at startup: a target on a 1-D norm used to throw out of
        // the bind and out of Preload, which stopped the server before it served anything.
        string root = Path.Combine(_base, "bound");
        Directory.CreateDirectory(root);
        LoraFixtures.WriteAdapter(Path.Combine(root, "good.safetensors"));
        LoraFixtures.WriteAdapter(Path.Combine(root, "norm.safetensors"), LoraProbeModel.Norm);
        var logger = new LoraRecordingLogger();
        using var service = ProbeService();

        LoraPreloadReport report = new LoraAdapterDirectory(root).Preload(service, logger);

        Assert.Equal(new[] { "good" }, report.Loaded);
        var (file, reason) = Assert.Single(report.Refused);
        Assert.Equal("norm.safetensors", file);
        Assert.Contains("not a 2-D projection weight", reason);
        Assert.Single(logger.Entries, e => e.Level == LogLevel.Warning);
    }

    [Fact]
    public void Preload_AnUnexpectedBindFailure_SkipsEachAdapterWithAWarning()
    {
        string root = Path.Combine(_base, "failing");
        Directory.CreateDirectory(root);
        LoraFixtures.WriteAdapter(Path.Combine(root, "one.safetensors"));
        LoraFixtures.WriteAdapter(Path.Combine(root, "two.safetensors"));
        var logger = new LoraRecordingLogger();
        using var service = ProbeService();
        var model = (LoraProbeModel)service.Model;
        model.BindFailure = new InvalidOperationException("out of device memory");

        LoraPreloadReport report = new LoraAdapterDirectory(root).Preload(service, logger);

        Assert.Empty(report.Loaded);
        Assert.Equal(new[] { "one.safetensors", "two.safetensors" }, report.Refused.Select(r => r.File));
        Assert.All(report.Refused, r => Assert.Contains("failed unexpectedly", r.Reason));
        var warnings = logger.Entries.Where(e => e.Level == LogLevel.Warning).ToList();
        Assert.Equal(2, warnings.Count);
        Assert.All(warnings, w => Assert.Same(model.BindFailure, w.Exception));
        Assert.Equal(0, service.LoraRegistry.Count);
        Assert.Empty(((IAdapterAwareModel)model).BoundAdapters);
    }

    private ModelService ProbeService()
    {
        string gguf = Path.Combine(_base, "probe.gguf");
        BackendFailureWarmupTests.WriteProbeGguf(gguf);
        var service = new ModelService(NullLogger<ModelService>.Instance,
            (path, _, _, _) => new LoraProbeModel(path, new float[1], 1, 1, supportsLora: true, LoraProbeModel.Template));
        service.LoadModel(gguf, null, "cpu");
        return service;
    }

    [Fact]
    public void Preload_StopsAtTheLimit()
    {
        string root = Path.Combine(_base, "many");
        Directory.CreateDirectory(root);
        for (int i = 0; i < LoraAdapterDirectory.MaxAdapters + 2; i++)
            LoraFixtures.WriteAdapter(Path.Combine(root, $"a{i:D2}.safetensors"));
        var logger = new LoraRecordingLogger();
        using var service = new ModelService(NullLogger<ModelService>.Instance);

        LoraPreloadReport report = new LoraAdapterDirectory(root).Preload(service, logger);

        Assert.Equal(LoraAdapterDirectory.MaxAdapters, report.Loaded.Count);
        Assert.Equal(LoraAdapterDirectory.MaxAdapters, service.LoraRegistry.Count);
        Assert.False(service.LoraRegistry.Contains($"a{LoraAdapterDirectory.MaxAdapters:D2}"));
        Assert.Contains(logger.Entries, e => e.Level == LogLevel.Warning && e.Message.Contains("limit of 32 adapters"));
    }

    [LoraFileSymlinkFact]
    public void Preload_RefusesATopLevelLinkThatLeavesTheDirectory()
    {
        string root = Path.Combine(_base, "linked");
        Directory.CreateDirectory(root);
        Assert.True(LoraLinks.TryLinkFile(Path.Combine(root, "leak.safetensors"), Path.Combine(_outside, "secret.safetensors")));
        using var service = new ModelService(NullLogger<ModelService>.Instance);

        LoraPreloadReport report = new LoraAdapterDirectory(root).Preload(service, new LoraRecordingLogger());

        Assert.Empty(report.Loaded);
        var (file, reason) = Assert.Single(report.Refused);
        Assert.Equal("leak.safetensors", file);
        Assert.Contains("does not resolve to a file inside the directory", reason);
        Assert.Equal(0, service.LoraRegistry.Count);
    }
}

/// <summary>Safetensors adapter fixtures for the <c>/v1/loras</c> tests.</summary>
internal static class LoraFixtures
{
    public const string Target = "blk.0.attn_output.weight";

    /// <summary>A rank-<paramref name="rank"/> adapter on <paramref name="target"/> whose factors
    /// are all ones (or zero bytes for non-float dtypes), with <paramref name="metadataAlpha"/>,
    /// as written, as its <c>__metadata__</c> alpha when given.</summary>
    public static string WriteAdapter(string path, string target = Target, int rank = 1, int inDim = 1, int outDim = 1, string dtype = "F32",
        string? metadataAlpha = null)
    {
        int size = dtype switch { "F64" or "I64" => 8, "F32" or "I32" => 4, "F16" or "BF16" or "I16" => 2, _ => 1 };
        long aBytes = (long)rank * inDim * size;
        long bBytes = (long)outDim * rank * size;
        string header = "{" + (metadataAlpha == null ? "" : "\"__metadata__\":{\"alpha\":\"" + metadataAlpha + "\"},")
                      + "\"" + target + ".lora_a\":{\"dtype\":\"" + dtype + "\",\"shape\":[" + rank + "," + inDim + "],\"data_offsets\":[0," + aBytes + "]},"
                      + "\"" + target + ".lora_b\":{\"dtype\":\"" + dtype + "\",\"shape\":[" + outDim + "," + rank + "],\"data_offsets\":[" + aBytes + "," + (aBytes + bBytes) + "]}}";
        var data = new byte[aBytes + bBytes];
        if (dtype == "F32")
            for (int i = 0; i < data.Length; i += 4)
                BitConverter.GetBytes(1f).CopyTo(data, i);
        return WriteRaw(path, header, data);
    }

    /// <summary>A safetensors file with this exact header text and data section.</summary>
    public static string WriteRaw(string path, string header, byte[] data)
    {
        byte[] hb = Encoding.UTF8.GetBytes(header);
        using var fs = File.Create(path);
        fs.Write(BitConverter.GetBytes((ulong)hb.Length));
        fs.Write(hb);
        fs.Write(data);
        return path;
    }

    /// <summary>Neither separator form of <paramref name="path"/> appears in <paramref name="text"/>.</summary>
    public static void AssertNoPath(string text, string path)
    {
        Assert.DoesNotContain(path, text, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain(path.Replace('\\', '/'), text, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain(path.Replace('/', '\\'), text, StringComparison.OrdinalIgnoreCase);
    }

    /// <summary>
    /// <see cref="AssertNoPath"/> over a JSON response: over the raw text and over every
    /// decoded string in it. The raw check alone misses a Windows path, whose backslashes the
    /// JSON writer escapes.
    /// </summary>
    public static void AssertNoPathInJson(string json, string path)
    {
        AssertNoPath(json, path);
        using var doc = JsonDocument.Parse(json);
        foreach (string value in Strings(doc.RootElement))
            AssertNoPath(value, path);

        static IEnumerable<string> Strings(JsonElement e)
        {
            if (e.ValueKind == JsonValueKind.String)
                yield return e.GetString()!;
            else if (e.ValueKind == JsonValueKind.Object)
                foreach (var member in e.EnumerateObject())
                    foreach (string s in Strings(member.Value))
                        yield return s;
            else if (e.ValueKind == JsonValueKind.Array)
                foreach (var item in e.EnumerateArray())
                    foreach (string s in Strings(item))
                        yield return s;
        }
    }
}

/// <summary>Creating links for the confinement tests, where this process may.</summary>
internal static class LoraLinks
{
    private static readonly Lazy<bool> s_files = new(() => Probe(directory =>
    {
        string target = Path.Combine(directory, "target");
        File.WriteAllText(target, "probe");
        return TryLinkFile(Path.Combine(directory, "link"), target);
    }));

    private static readonly Lazy<bool> s_directories = new(() => Probe(directory =>
    {
        string target = Path.Combine(directory, "target");
        Directory.CreateDirectory(target);
        File.WriteAllText(Path.Combine(target, "f"), "probe");
        string link = Path.Combine(directory, "link");
        return TryLinkDirectory(link, target) && File.Exists(Path.Combine(link, "f"));
    }));

    public static bool CanLinkFiles => s_files.Value;
    public static bool CanLinkDirectories => s_directories.Value;

    public static bool TryLinkFile(string link, string target)
    {
        try
        {
            File.CreateSymbolicLink(link, target);
            return true;
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or PlatformNotSupportedException)
        {
            return false;
        }
    }

    /// <summary>A directory symlink, or on Windows, where an unprivileged process outside
    /// Developer Mode may not create one, a junction: it needs no privilege, and the guard has
    /// to follow it just the same.</summary>
    public static bool TryLinkDirectory(string link, string target)
    {
        try
        {
            Directory.CreateSymbolicLink(link, target);
            return true;
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or PlatformNotSupportedException)
        {
        }
        return OperatingSystem.IsWindows() && TryJunction(link, target);
    }

    private static bool TryJunction(string link, string target)
    {
        var start = new ProcessStartInfo("cmd.exe")
        {
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
        };
        foreach (string arg in new[] { "/c", "mklink", "/J", link, target })
            start.ArgumentList.Add(arg);
        try
        {
            using var process = Process.Start(start);
            if (process == null)
                return false;
            process.StandardOutput.ReadToEnd();
            process.StandardError.ReadToEnd();
            process.WaitForExit(30_000);
            return process.HasExited && process.ExitCode == 0 && Directory.Exists(link);
        }
        catch (Exception ex) when (ex is IOException or System.ComponentModel.Win32Exception or InvalidOperationException)
        {
            return false;
        }
    }

    private static bool Probe(Func<string, bool> attempt)
    {
        string directory = Path.Combine(Path.GetTempPath(), "ts-lora-link-probe-" + Guid.NewGuid().ToString("N"));
        try
        {
            Directory.CreateDirectory(directory);
            return attempt(directory);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            return false;
        }
        finally
        {
            DeleteTree(directory);
        }
    }

    /// <summary>
    /// Delete a test directory that may hold links: each link is removed itself, never
    /// followed into its target. Recursive <see cref="Directory.Delete(string, bool)"/> cannot
    /// be used for this on Windows, where it handles a junction as a volume mount point and
    /// fails with access denied. Best effort.
    /// </summary>
    public static void DeleteTree(string path)
    {
        try
        {
            var info = new DirectoryInfo(path);
            if (!info.Exists)
                return;
            if (info.Attributes.HasFlag(FileAttributes.ReparsePoint))
            {
                info.Delete(recursive: false);
                return;
            }
            foreach (var sub in info.GetDirectories())
                DeleteTree(sub.FullName);
            info.Delete(recursive: true);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
        }
    }
}

/// <summary>A test that needs a file symlink; skipped where this process cannot create one
/// (Windows without Developer Mode or elevation).</summary>
public sealed class LoraFileSymlinkFactAttribute : FactAttribute
{
    public LoraFileSymlinkFactAttribute()
    {
        if (!LoraLinks.CanLinkFiles)
            Skip = "Requires permission to create file symbolic links (Developer Mode or elevation on Windows).";
    }
}

/// <summary>A test that needs a directory link (a symlink, or a junction on Windows); skipped
/// where neither can be created.</summary>
public sealed class LoraDirectoryLinkFactAttribute : FactAttribute
{
    public LoraDirectoryLinkFactAttribute()
    {
        if (!LoraLinks.CanLinkDirectories)
            Skip = "Requires permission to create a directory symbolic link or junction.";
    }
}

/// <summary>Records log entries, for the tests that check what reaches the server log.</summary>
internal sealed class LoraRecordingLogger : ILogger, ILoggerProvider
{
    public ConcurrentQueue<(LogLevel Level, string Message, Exception? Exception)> Entries { get; } = new();

    public IDisposable? BeginScope<TState>(TState state) where TState : notnull => null;
    public bool IsEnabled(LogLevel logLevel) => true;

    public void Log<TState>(LogLevel logLevel, EventId eventId, TState state, Exception? exception, Func<TState, Exception?, string> formatter)
        => Entries.Enqueue((logLevel, formatter(state, exception), exception));

    public ILogger CreateLogger(string categoryName) => this;
    void IDisposable.Dispose() { }
}
