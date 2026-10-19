// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System.Security.Cryptography;
using System.Text.Json;
using Microsoft.Extensions.Logging.Abstractions;
using TensorAgent.Core.Catalog;
using TensorAgent.Core.Downloads;
using TensorAgent.Core.Hosting;
using TensorAgent.Core.Interop;
using TensorAgent.Core.Sessions;
using TensorAgent.Core.Settings;

namespace TensorAgent.Tests;

/// <summary>
/// Two catalog entries that list the same file hold one copy of it.
///
/// <para>
/// The MiniMax-H3 keyframes and references entries share 24.0 GB of companions and differ
/// only in an 11.4 GB denoiser. Stored and downloaded per entry, the pair cost 70.9 GB of
/// a device's storage and a second 24 GB transfer; with the second entry's copies linked
/// to the first's, 46.8 GB and only the denoiser's download. The entries here are small
/// stand-ins for that pair -- a shared companion under its real SHA-256, plus one file
/// each -- installed through the store's real download path against
/// <see cref="RangeServer"/>, whose request count is how a test knows a file was NOT
/// fetched.
/// </para>
/// </summary>
public sealed class SharedFileLinkTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "tensoragent-link-" + Guid.NewGuid().ToString("N"));
    private readonly List<IDisposable> _owned = new();

    public SharedFileLinkTests() => Directory.CreateDirectory(_root);

    public void Dispose()
    {
        foreach (IDisposable owned in _owned)
        {
            try { owned.Dispose(); } catch (Exception) { /* scratch */ }
        }
        try { Directory.Delete(_root, true); } catch (Exception) { /* scratch */ }
    }

    [Theory]
    [InlineData("nothing")]
    [InlineData("an interrupted download")]
    [InlineData("a wrong-sized file")]
    public async Task TheSecondEntryLinksTheCopyTheFirstHoldsInsteadOfDownloadingIt(string leftOverForB)
    {
        Pair pair = NewPair();
        ModelStore store = Store(pair);
        await store.DownloadAsync(pair.A, null, CancellationToken.None);
        Assert.Equal(1, pair.Shared.Requests);

        string first = store.PathFor(pair.A, pair.SharedOfA);
        string second = store.PathFor(pair.B, pair.SharedOfB);
        byte[] half = pair.Shared.Body[..(pair.Shared.Body.Length / 2)];
        Directory.CreateDirectory(store.DirectoryFor(pair.B));
        if (leftOverForB == "an interrupted download")
            await File.WriteAllBytesAsync(ResumableDownloader.PartPath(second), half);
        else if (leftOverForB == "a wrong-sized file")
            await File.WriteAllBytesAsync(second, half);

        var reports = new Reports();
        await store.DownloadAsync(pair.B, reports, CancellationToken.None);

        // The only request for the shared file is still the one A's install made.
        Assert.Equal(1, pair.Shared.Requests);
        Assert.Equal(1, pair.BOnly.Requests);
        Assert.Equal(InstallState.Installed, store.StateOf(pair.B));
        Assert.Equal(pair.Shared.Body, await File.ReadAllBytesAsync(second));
        Assert.True(AreOneFile(first, second), "B holds a second copy of the shared file, not a second name for A's");
        Assert.False(File.Exists(ResumableDownloader.PartPath(second)), "the stale part file was left beside the link");

        // Reported complete in one step.
        ModelDownloadProgress linked = Assert.Single(reports.All, r => r.FileName == pair.SharedOfB.FileName);
        Assert.Equal(pair.Shared.Body.Length, linked.BytesReceived);
    }

    [Fact]
    public async Task TheCopiesAreLinkedBeforeAnyTransferSoTheirEntryCanGoMidDownload()
    {
        Pair pair = NewPair();
        // B's own file comes first, as a MiniMax-H3 entry's denoiser does, and A is
        // deleted the moment that file is requested: a user freeing space for B while an
        // 11.4 GB transfer runs. Linked only once the transfer had finished, the shared
        // file found no copy left and was fetched a second time.
        ModelStore? store = null;
        HttpClient http = Own(new HttpClient(new OnRequest(pair.BOnly.Url, () => store!.Delete(pair.A))));
        store = new ModelStore(
            Path.Combine(_root, "models"), new ResumableDownloader(http, maxAttempts: 1), new[] { pair.A, pair.B });
        await store.DownloadAsync(pair.A, null, CancellationToken.None);

        var reports = new Reports();
        await store.DownloadAsync(pair.B, reports, CancellationToken.None);

        Assert.False(Directory.Exists(store.DirectoryFor(pair.A)));
        Assert.Equal(1, pair.Shared.Requests);
        Assert.Equal(1, pair.BOnly.Requests);
        Assert.Equal(InstallState.Installed, store.StateOf(pair.B));
        Assert.Equal(pair.Shared.Body, await File.ReadAllBytesAsync(store.PathFor(pair.B, pair.SharedOfB)));

        // And the progress starts from what is already here, which is what the time left
        // is worked out from: the transfer that follows is B's own file and nothing else.
        ModelDownloadProgress linked = Assert.Single(reports.All, r => r.FileName == pair.SharedOfB.FileName);
        Assert.Equal(1, linked.FileIndex);
        Assert.Equal(pair.Shared.Body.Length, linked.BytesReceived);
    }

    [Fact]
    public async Task DeletingTheEntryALinkCameFromLeavesTheOtherInstalledAndWhole()
    {
        Pair pair = NewPair();
        ModelStore store = Store(pair);
        await store.DownloadAsync(pair.A, null, CancellationToken.None);
        await store.DownloadAsync(pair.B, null, CancellationToken.None);
        Assert.Equal(1, pair.Shared.Requests);

        store.Delete(pair.A);

        Assert.False(Directory.Exists(store.DirectoryFor(pair.A)));
        Assert.Equal(InstallState.NotInstalled, store.StateOf(pair.A));
        Assert.Equal(InstallState.Installed, store.StateOf(pair.B));
        Assert.Equal(pair.Shared.Body, await File.ReadAllBytesAsync(store.PathFor(pair.B, pair.SharedOfB)));
        Assert.Equal(pair.BOnly.Body, await File.ReadAllBytesAsync(store.PathFor(pair.B, pair.B.Weights)));

        // Installed again, A takes the file back from B: the bytes never left the device.
        await store.DownloadAsync(pair.A, null, CancellationToken.None);
        Assert.Equal(1, pair.Shared.Requests);
        Assert.Equal(2, pair.AOnly.Requests);
        Assert.True(AreOneFile(store.PathFor(pair.A, pair.SharedOfA), store.PathFor(pair.B, pair.SharedOfB)));
    }

    [Fact]
    public async Task TheSweepNeverTakesAFileAClaimedEntryStillLists()
    {
        Pair pair = NewPair();
        ModelStore store = Store(pair);
        await store.DownloadAsync(pair.A, null, CancellationToken.None);
        await store.DownloadAsync(pair.B, null, CancellationToken.None);

        // Checked against the store's own catalog, both folders are claimed.
        Assert.Equal(0, store.SweepOrphanedModels());
        Assert.Equal(InstallState.Installed, store.StateOf(pair.A));
        Assert.Equal(InstallState.Installed, store.StateOf(pair.B));

        // A leaves the catalog for the retired list -- as an entry's id does when it is
        // re-pointed at another file -- and its folder goes, but B's name for the file it
        // linked from A stays.
        store.SweepOrphanedModels(new[] { pair.B }, retired: new[] { pair.A.Id });

        Assert.False(Directory.Exists(store.DirectoryFor(pair.A)));
        Assert.Equal(InstallState.Installed, store.StateOf(pair.B));
        Assert.Equal(pair.Shared.Body, await File.ReadAllBytesAsync(store.PathFor(pair.B, pair.SharedOfB)));
    }

    /// <summary>
    /// An entry re-pointed at another file of its model retires its id, and the sweep reclaims
    /// the old folder. Its successor takes over first what it can still use: LTX 2.5's move to
    /// another denoiser kept 29.1 GB of companions, and a user's Ingredients adapter in loras/
    /// that no download in the app can replace.
    /// </summary>
    [Fact]
    public async Task ARetiredEntryHandsItsSuccessorTheCompanionsAndWhatTheUserAdded()
    {
        Pair pair = NewPair();
        ModelStore store = Store(pair);
        await store.DownloadAsync(pair.A, null, CancellationToken.None);
        string adapter = Path.Combine(store.DirectoryFor(pair.A), "loras", "adapter.safetensors");
        Directory.CreateDirectory(Path.GetDirectoryName(adapter)!);
        byte[] adapterBytes = Body(4096, seed: 9);
        await File.WriteAllBytesAsync(adapter, adapterBytes);
        await File.WriteAllBytesAsync(ResumableDownloader.PartPath(store.PathFor(pair.A, pair.SharedOfA)), Body(16, seed: 10));

        store.SweepOrphanedModels(new[] { pair.B }, retired: new[] { pair.A.Id },
            successors: new Dictionary<string, string> { [pair.A.Id] = pair.B.Id });

        Assert.False(Directory.Exists(store.DirectoryFor(pair.A)));
        string successor = store.DirectoryFor(pair.B);
        Assert.Equal(adapterBytes, await File.ReadAllBytesAsync(Path.Combine(successor, "loras", "adapter.safetensors")));
        Assert.Equal(pair.Shared.Body, await File.ReadAllBytesAsync(store.PathFor(pair.B, pair.SharedOfB)));
        // The superseded weights and the part file went with the folder.
        Assert.Equal(new[] { pair.SharedOfB.FileName },
            Directory.EnumerateFiles(successor).Select(Path.GetFileName).ToArray());
        Assert.Equal(InstallState.Partial, store.StateOf(pair.B));

        // Installing the successor fetches only the file that changed.
        await store.DownloadAsync(pair.B, null, CancellationToken.None);
        Assert.Equal(1, pair.Shared.Requests);
        Assert.Equal(1, pair.BOnly.Requests);
        Assert.Equal(InstallState.Installed, store.StateOf(pair.B));
    }

    [Theory]
    [InlineData("a complete copy")]
    [InlineData("a wrong-sized file")]
    [InlineData("its own adapter")]
    public async Task TheCarryOverNeverReplacesWhatTheSuccessorAlreadyHolds(string successorHolds)
    {
        Pair pair = NewPair();
        ModelStore store = Store(pair);
        await store.DownloadAsync(pair.A, null, CancellationToken.None);
        string oldAdapter = Path.Combine(store.DirectoryFor(pair.A), "loras", "adapter.safetensors");
        Directory.CreateDirectory(Path.GetDirectoryName(oldAdapter)!);
        await File.WriteAllBytesAsync(oldAdapter, Body(4096, seed: 9));

        string shared = store.PathFor(pair.B, pair.SharedOfB);
        string newAdapter = Path.Combine(store.DirectoryFor(pair.B), "loras", "adapter.safetensors");
        Directory.CreateDirectory(Path.GetDirectoryName(newAdapter)!);
        byte[] ownAdapter = Body(2048, seed: 11);
        // A complete copy of its own: the same size (all the store checks) and other bytes, so a
        // replacement would show.
        byte[] ownCopy = Body(pair.Shared.Body.Length, seed: 12);
        if (successorHolds == "a complete copy")
            await File.WriteAllBytesAsync(shared, ownCopy);
        else if (successorHolds == "a wrong-sized file")
            await File.WriteAllBytesAsync(shared, pair.Shared.Body[..1000]);
        else
            await File.WriteAllBytesAsync(newAdapter, ownAdapter);

        store.SweepOrphanedModels(new[] { pair.B }, retired: new[] { pair.A.Id },
            successors: new Dictionary<string, string> { [pair.A.Id] = pair.B.Id });

        Assert.False(Directory.Exists(store.DirectoryFor(pair.A)));
        // Its own complete copy stays; a broken download under a listed name is replaced.
        Assert.Equal(successorHolds == "a complete copy" ? ownCopy : pair.Shared.Body, await File.ReadAllBytesAsync(shared));
        if (successorHolds == "its own adapter")
            Assert.Equal(ownAdapter, await File.ReadAllBytesAsync(newAdapter));
        else
            Assert.Equal(Body(4096, seed: 9), await File.ReadAllBytesAsync(newAdapter));
    }

    /// <summary>
    /// The sweep's delete only unlinks a link, and the carry-over never follows one: a loras/
    /// that points at the user's shared library moves as the link, with every file behind it left
    /// where it is, and a retired folder that is itself a link (moved to another disk) is not
    /// copied back onto this one.
    /// </summary>
    [SkippableFact]
    public async Task TheCarryOverMovesLinksAsLinksAndNeverEmptiesWhatTheyPointAt()
    {
        Pair pair = NewPair();
        ModelStore store = Store(pair);
        await store.DownloadAsync(pair.A, null, CancellationToken.None);
        string library = Path.Combine(_root, "shared-library");
        Directory.CreateDirectory(library);
        byte[] lora = Body(2048, seed: 13);
        await File.WriteAllBytesAsync(Path.Combine(library, "someone-elses.safetensors"), lora);
        TestPlatforms.CreateDirectorySymlink(Path.Combine(store.DirectoryFor(pair.A), "loras"), library);

        store.SweepOrphanedModels(new[] { pair.B }, retired: new[] { pair.A.Id },
            successors: new Dictionary<string, string> { [pair.A.Id] = pair.B.Id });

        Assert.Equal(lora, await File.ReadAllBytesAsync(Path.Combine(library, "someone-elses.safetensors")));
        var moved = new DirectoryInfo(Path.Combine(store.DirectoryFor(pair.B), "loras"));
        Assert.NotNull(moved.LinkTarget);
        Assert.Equal(lora, await File.ReadAllBytesAsync(Path.Combine(moved.FullName, "someone-elses.safetensors")));

        // Deleting the successor removes its names, never the library behind the link.
        store.Delete(pair.B);
        Assert.Equal(lora, await File.ReadAllBytesAsync(Path.Combine(library, "someone-elses.safetensors")));

        // A retired folder that is a link elsewhere: nothing is carried out of it.
        Pair other = NewPair();
        ModelStore second = Store(other);
        string elsewhere = Path.Combine(_root, "external-disk", other.A.Id);
        Directory.CreateDirectory(Path.GetDirectoryName(elsewhere)!);
        await second.DownloadAsync(other.A, null, CancellationToken.None);
        Directory.Move(second.DirectoryFor(other.A), elsewhere);
        TestPlatforms.CreateDirectorySymlink(second.DirectoryFor(other.A), elsewhere);
        second.SweepOrphanedModels(new[] { other.B }, retired: new[] { other.A.Id },
            successors: new Dictionary<string, string> { [other.A.Id] = other.B.Id });
        Assert.False(File.Exists(second.PathFor(other.B, other.SharedOfB)));
        Assert.Equal(other.Shared.Body, await File.ReadAllBytesAsync(Path.Combine(elsewhere, other.SharedOfA.FileName)));
    }

    [SkippableTheory]
    [InlineData(true)]
    [InlineData(false)]
    public async Task TheCarryOverNeverWritesThroughAnExistingSuccessorLink(bool wholeFolder)
    {
        Pair pair = NewPair();
        ModelStore store = Store(pair);
        await store.DownloadAsync(pair.A, null, CancellationToken.None);
        string oldAdapter = Path.Combine(store.DirectoryFor(pair.A), "loras", "adapter.safetensors");
        Directory.CreateDirectory(Path.GetDirectoryName(oldAdapter)!);
        byte[] adapter = Body(4096, seed: 19);
        await File.WriteAllBytesAsync(oldAdapter, adapter);
        string library = Path.Combine(_root, "successor-library");
        Directory.CreateDirectory(library);
        string destination = wholeFolder ? store.DirectoryFor(pair.B) : Path.Combine(store.DirectoryFor(pair.B), "loras");
        Directory.CreateDirectory(Path.GetDirectoryName(destination)!);
        TestPlatforms.CreateDirectorySymlink(destination, library);

        store.SweepOrphanedModels(new[] { pair.B }, retired: new[] { pair.A.Id },
            successors: new Dictionary<string, string> { [pair.A.Id] = pair.B.Id });

        Assert.Empty(Directory.EnumerateFileSystemEntries(library));
        Assert.NotNull(new DirectoryInfo(destination).LinkTarget);
        Assert.Equal(adapter, await File.ReadAllBytesAsync(oldAdapter));
    }

    [Fact]
    public async Task AFailedAdapterMoveKeepsTheRetiredFolderForARetry()
    {
        Pair pair = NewPair();
        ModelStore store = Store(pair);
        await store.DownloadAsync(pair.A, null, CancellationToken.None);
        string oldAdapter = Path.Combine(store.DirectoryFor(pair.A), "loras", "adapter.safetensors");
        Directory.CreateDirectory(Path.GetDirectoryName(oldAdapter)!);
        byte[] adapter = Body(4096, seed: 23);
        await File.WriteAllBytesAsync(oldAdapter, adapter);
        Directory.CreateDirectory(store.DirectoryFor(pair.B));
        string blocked = Path.Combine(store.DirectoryFor(pair.B), "loras");
        await File.WriteAllTextAsync(blocked, "This is a file, so the adapter cannot move into it.");

        void Sweep() => store.SweepOrphanedModels(new[] { pair.B }, retired: new[] { pair.A.Id },
            successors: new Dictionary<string, string> { [pair.A.Id] = pair.B.Id });
        Sweep();
        Assert.Equal(adapter, await File.ReadAllBytesAsync(oldAdapter));
        Assert.True(File.Exists(blocked));

        File.Delete(blocked);
        Sweep();
        Assert.False(Directory.Exists(store.DirectoryFor(pair.A)));
        Assert.Equal(adapter, await File.ReadAllBytesAsync(Path.Combine(store.DirectoryFor(pair.B), "loras", "adapter.safetensors")));
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public async Task AnIncompleteCopyIsNeverLinked(bool onlyAPartFile)
    {
        Pair pair = NewPair();
        ModelStore store = Store(pair);

        // A's copy is what an interrupted transfer leaves: the part file, or a truncated
        // file under the final name.
        string first = store.PathFor(pair.A, pair.SharedOfA);
        string incomplete = onlyAPartFile ? ResumableDownloader.PartPath(first) : first;
        byte[] half = pair.Shared.Body[..(pair.Shared.Body.Length / 2)];
        Directory.CreateDirectory(store.DirectoryFor(pair.A));
        await File.WriteAllBytesAsync(incomplete, half);

        Assert.Equal(pair.B.TotalBytes, store.RemainingBytes(pair.B));
        await store.DownloadAsync(pair.B, null, CancellationToken.None);

        Assert.Equal(1, pair.Shared.Requests);
        Assert.Equal(InstallState.Installed, store.StateOf(pair.B));
        Assert.Equal(pair.Shared.Body, await File.ReadAllBytesAsync(store.PathFor(pair.B, pair.SharedOfB)));
        Assert.Equal(half, await File.ReadAllBytesAsync(incomplete));
    }

    [Fact]
    public async Task AFileThatSharesOnlyItsNameAndSizeIsDownloaded()
    {
        Pair pair = NewPair();
        // Same name, same size, different bytes: a different artifact.
        RangeServer other = Own(new RangeServer(Body(pair.Shared.Body.Length, seed: 4)));
        CatalogModel b = pair.B with
        {
            Files = new[] { pair.B.Weights, Served(CatalogFileRole.TextEncoder, pair.SharedOfB.FileName, other) },
        };
        var store = new ModelStore(
            Path.Combine(_root, "models"), new ResumableDownloader(maxAttempts: 1), new[] { pair.A, b });

        await store.DownloadAsync(pair.A, null, CancellationToken.None);
        Assert.Equal(b.TotalBytes, store.RemainingBytes(b));
        await store.DownloadAsync(b, null, CancellationToken.None);

        Assert.Equal(1, other.Requests);
        Assert.Equal(other.Body, await File.ReadAllBytesAsync(store.PathFor(b, b.Files[1])));
        Assert.Equal(pair.Shared.Body, await File.ReadAllBytesAsync(store.PathFor(pair.A, pair.SharedOfA)));
    }

    [Fact]
    public async Task TheRemainingDownloadLeavesOutWhatWillBeLinked()
    {
        Pair pair = NewPair();
        ModelStore store = Store(pair);
        Assert.Equal(pair.B.TotalBytes, store.RemainingBytes(pair.B));

        await store.DownloadAsync(pair.A, null, CancellationToken.None);

        // B's own file is all that is left to fetch; the catalog's size is still the entry's.
        Assert.Equal(pair.BOnly.Body.Length, store.RemainingBytes(pair.B));
        Assert.Equal(pair.BOnly.Body.Length + pair.Shared.Body.Length, pair.B.TotalBytes);

        // And the catalog route, which is what a page reads, says the same.
        using var loopback = new LoopbackServer(NullLogger.Instance);
        loopback.MapAgent(
            new[] { pair.A, pair.B }, store,
            new ConversationStore(Path.Combine(_root, "conversations")),
            new SettingsStore(Path.Combine(_root, "settings.json")));
        loopback.Start();
        using var client = new HttpClient { BaseAddress = new Uri(loopback.BaseUrl) };
        client.DefaultRequestHeaders.Add("Cookie", $"{LoopbackServer.TokenCookie}={loopback.Token}");

        JsonElement listed = JsonSerializer.Deserialize<JsonElement>(await client.GetStringAsync("/api/agent/catalog"));
        JsonElement references = listed.GetProperty("models").EnumerateArray()
            .Single(m => m.GetProperty("id").GetString() == pair.B.Id);
        Assert.Equal(pair.BOnly.Body.Length, references.GetProperty("remainingBytes").GetInt64());
        Assert.Equal(pair.B.TotalBytes, references.GetProperty("totalBytes").GetInt64());
    }

    [Fact]
    public async Task AnOptionalCompanionAskedForIsLinkedToo()
    {
        Pair pair = NewPair();
        CatalogModel b = pair.B with { Files = new[] { pair.B.Weights, pair.SharedOfB with { Optional = true } } };
        var store = new ModelStore(
            Path.Combine(_root, "models"), new ResumableDownloader(maxAttempts: 1), new[] { pair.A, b });
        await store.DownloadAsync(pair.A, null, CancellationToken.None);

        Assert.Equal(pair.BOnly.Body.Length, store.RemainingBytes(b, includeOptional: true));
        await store.DownloadAsync(b, null, CancellationToken.None, new[] { CatalogFileRole.TextEncoder });

        Assert.Equal(1, pair.Shared.Requests);
        Assert.NotNull(store.CompanionPath(b, CatalogFileRole.TextEncoder));
        Assert.True(AreOneFile(store.PathFor(pair.A, pair.SharedOfA), store.PathFor(b, b.Files[1])));
    }

    [Fact]
    public async Task ALinkThatCannotBeMadeFallsBackToDownloadingTheFile()
    {
        Pair pair = NewPair();
        var store = new ModelStore(
            Path.Combine(_root, "models"), new ResumableDownloader(maxAttempts: 1), new[] { pair.A, pair.B })
        {
            // What link(2) answers across volumes; a file system without hard links fails alike.
            CreateHardLink = (_, _) => throw new IOException("Cross-device link (error 18)"),
        };
        await store.DownloadAsync(pair.A, null, CancellationToken.None);

        // B had fetched half of the file before, and a failed link must not cost it that half.
        string second = store.PathFor(pair.B, pair.SharedOfB);
        int half = pair.Shared.Body.Length / 2;
        Directory.CreateDirectory(store.DirectoryFor(pair.B));
        await File.WriteAllBytesAsync(ResumableDownloader.PartPath(second), pair.Shared.Body[..half]);

        await store.DownloadAsync(pair.B, null, CancellationToken.None);

        Assert.Equal(2, pair.Shared.Requests);
        Assert.Equal($"bytes={half}-", pair.Shared.RangeHeaders[^1]);
        Assert.Equal(InstallState.Installed, store.StateOf(pair.B));
        Assert.Equal(pair.Shared.Body, await File.ReadAllBytesAsync(second));
        Assert.False(AreOneFile(store.PathFor(pair.A, pair.SharedOfA), second), "B shares A's file although the link failed");
    }

    /// <summary>The interop itself, through the real system call rather than the store's seam.</summary>
    [Fact]
    public void AHardLinkIsASecondNameThatOutlivesTheFirstAndNeverReplacesAFile()
    {
        byte[] bytes = Body(4096, seed: 5);
        string first = Path.Combine(_root, "first.bin");
        string second = Path.Combine(_root, "second.bin");
        File.WriteAllBytes(first, bytes);

        HardLinks.Create(first, second);
        Assert.True(AreOneFile(first, second));
        File.Delete(first);
        Assert.Equal(bytes, File.ReadAllBytes(second));

        // A name that is taken is refused with the system's reason, and never overwritten.
        string taken = Path.Combine(_root, "taken.bin");
        byte[] other = Body(1024, seed: 6);
        File.WriteAllBytes(taken, other);
        IOException refused = Assert.Throws<IOException>(() => HardLinks.Create(second, taken));
        Assert.False(string.IsNullOrWhiteSpace(refused.Message));
        Assert.Equal(other, File.ReadAllBytes(taken));
    }

    // ---- fixtures ------------------------------------------------------------------

    /// <summary>A keyframes/references pair in miniature: one shared companion, one file each.</summary>
    private sealed record Pair(CatalogModel A, CatalogModel B, RangeServer Shared, RangeServer AOnly, RangeServer BOnly)
    {
        public CatalogFile SharedOfA => A.Files.Single(f => f.Role == CatalogFileRole.TextEncoder);
        public CatalogFile SharedOfB => B.Files.Single(f => f.Role == CatalogFileRole.TextEncoder);
    }

    private Pair NewPair()
    {
        RangeServer shared = Own(new RangeServer(Body(256 * 1024, seed: 1)));
        RangeServer aOnly = Own(new RangeServer(Body(64 * 1024, seed: 2)));
        RangeServer bOnly = Own(new RangeServer(Body(96 * 1024, seed: 3)));
        CatalogModel a = Entry("keyframes",
            Served(CatalogFileRole.Weights, "keyframes-dit.gguf", aOnly),
            Served(CatalogFileRole.TextEncoder, "text-encoder.gguf", shared));
        CatalogModel b = Entry("references",
            Served(CatalogFileRole.Weights, "references-dit.gguf", bOnly),
            Served(CatalogFileRole.TextEncoder, "text-encoder.gguf", shared));
        return new Pair(a, b, shared, aOnly, bOnly);
    }

    private ModelStore Store(Pair pair) =>
        new(Path.Combine(_root, "models"), new ResumableDownloader(maxAttempts: 1), new[] { pair.A, pair.B });

    private static CatalogFile Served(CatalogFileRole role, string name, RangeServer server) =>
        new(role, name, server.Url, server.Body.Length, Convert.ToHexStringLower(SHA256.HashData(server.Body)));

    private static CatalogModel Entry(string id, params CatalogFile[] files) => new()
    {
        Id = id,
        DisplayName = id,
        Family = CatalogFamily.Gemma4,
        Kind = CatalogArchitectureKind.Dense,
        Parameters = "test",
        Quantization = "test",
        Files = files,
        Modalities = CatalogModalities.Text,
        MinDeviceMemoryGB = 1,
        ContextLength = 128,
        KvCacheDtype = "f16",
        Sampling = new CatalogSampling(1, 1, 1, 0),
        License = "test",
    };

    private static byte[] Body(int size, int seed)
    {
        var bytes = new byte[size];
        new Random(seed).NextBytes(bytes);
        return bytes;
    }

    private T Own<T>(T value) where T : IDisposable { _owned.Add(value); return value; }

    /// <summary>
    /// Whether two paths name one file. .NET exposes no inode number, so the file is asked
    /// directly: a byte changed through one name reads back through the other only when
    /// both are the same file. The byte is put back afterwards.
    /// </summary>
    private static bool AreOneFile(string first, string second)
    {
        using var writer = new FileStream(first, FileMode.Open, FileAccess.ReadWrite, FileShare.ReadWrite);
        int original = writer.ReadByte();
        try
        {
            writer.Position = 0;
            writer.WriteByte((byte)(original ^ 0xFF));
            writer.Flush();
            using var reader = new FileStream(second, FileMode.Open, FileAccess.Read, FileShare.ReadWrite);
            return reader.ReadByte() == (original ^ 0xFF);
        }
        finally
        {
            writer.Position = 0;
            writer.WriteByte((byte)original);
        }
    }

    /// <summary>
    /// Runs an action when one URL is requested, before the request goes out: the moment a
    /// transfer starts, which is when a test changes the store under a running download.
    /// </summary>
    private sealed class OnRequest(string url, Action action) : DelegatingHandler(new SocketsHttpHandler())
    {
        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            if (string.Equals(request.RequestUri?.AbsoluteUri, url, StringComparison.Ordinal))
                action();
            return base.SendAsync(request, cancellationToken);
        }
    }

    /// <summary>
    /// Every report, recorded as it is made. The store reports a linked file inline, so that
    /// report is here by the time the download returns; a transfer's own reports come
    /// through a <see cref="Progress{T}"/> and may trail it, which is why only the linked
    /// file's is asserted.
    /// </summary>
    private sealed class Reports : IProgress<ModelDownloadProgress>
    {
        private readonly List<ModelDownloadProgress> _all = new();

        public IReadOnlyList<ModelDownloadProgress> All
        {
            get { lock (_all) return _all.ToArray(); }
        }

        public void Report(ModelDownloadProgress value)
        {
            lock (_all) _all.Add(value);
        }
    }
}
