// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using TensorSharp.AgentHost;
using TensorSharp.Chat.Studio;
using TensorSharp.Studio;
using Xunit;

namespace InferenceWeb.Tests;

/// <summary>
/// The desktop TensorAgent on a Mac is a Mac Catalyst app, and .NET reports Mac Catalyst
/// as iOS: <see cref="OperatingSystem.IsIOS"/> is true there and
/// <see cref="OperatingSystem.IsMacOS"/> is false. So a check written the obvious way
/// treats the Mac app as a phone. Bringing the app up found 22 such checks in 12 files:
/// the engine resolver looked for a statically linked GgmlOps that is not there, the
/// GGUF page-cache warm-up was skipped, Seatbelt was never offered, and code execution
/// refused to start a process at all. <see cref="HostOS"/> answers the two questions
/// separately; these tests pin its answers and keep new checks honest.
/// </summary>
public class HostOSTests
{
    private static string RepoRoot
    {
        get
        {
            var dir = new DirectoryInfo(AppContext.BaseDirectory);
            while (dir != null && !File.Exists(Path.Combine(dir.FullName, "TensorSharp.slnx")))
                dir = dir.Parent;
            Assert.NotNull(dir);
            return dir!.FullName;
        }
    }

    [Fact]
    public void ADesktopHostIsNeverAPhone()
    {
        // The test host is a desktop process (net10.0): a Mac is a Mac desktop, and
        // nothing here is an iPhone, an iPad or an Apple TV.
        Assert.Equal(OperatingSystem.IsMacOS(), HostOS.IsMacDesktop);
        Assert.False(HostOS.IsAppleMobile);
    }

    [Fact]
    public void StudioDesktopDefaultsAllowMediaSubprocesses()
    {
        Assert.True(new StudioOptions().EnableExternalProcesses);
        Assert.True(StudioHostOptions.Create().EnableExternalProcesses);
        Assert.Equal(1024, StudioHostOptions.Create(1024).MaxImportBytes);
    }

    /// <summary>
    /// Every Apple platform check in the code the Mac app shares with the phone either
    /// goes through <see cref="HostOS"/> or names Mac Catalyst itself, so its author had
    /// to decide which side the Mac app is on.
    /// </summary>
    [Fact]
    public void NoApplePlatformCheckInTheSharedAppCodeForgetsMacCatalyst()
    {
        string root = RepoRoot;
        var sources = new List<string>();
        foreach (string project in new[] { "TensorSharp.AgentHost", "TensorSharp.Studio", Path.Combine("TensorSharp.Chat", "Studio"), Path.Combine("TensorAgent", "src", "TensorAgent.Core") })
        {
            sources.AddRange(Directory.EnumerateFiles(Path.Combine(root, project), "*.cs", SearchOption.AllDirectories)
                .Where(f => !f.Contains($"{Path.DirectorySeparatorChar}bin{Path.DirectorySeparatorChar}", StringComparison.Ordinal)
                            && !f.Contains($"{Path.DirectorySeparatorChar}obj{Path.DirectorySeparatorChar}", StringComparison.Ordinal)
                            && Path.GetFileName(f) != "HostOS.cs"));
        }
        sources.Add(Path.Combine(root, "TensorSharp.Backends.GGML", "GgmlNative.cs"));
        sources.Add(Path.Combine(root, "TensorSharp.Runtime", "GgufReader.Prefault.cs"));

        var offenders = new List<string>();
        foreach (string file in sources)
        {
            // A statement is the unit a condition lives in; a line is too small (a check
            // can wrap) and a method too large.
            foreach (string statement in File.ReadAllText(file).Split(';'))
            {
                bool asksApple = statement.Contains("OperatingSystem.IsMacOS()", StringComparison.Ordinal)
                                 || statement.Contains("OperatingSystem.IsIOS()", StringComparison.Ordinal);
                if (asksApple
                    && !statement.Contains("IsMacCatalyst", StringComparison.Ordinal)
                    && !statement.Contains("HostOS.", StringComparison.Ordinal))
                {
                    offenders.Add($"{Path.GetRelativePath(root, file)}: {statement.Trim().Split('\n').Last().Trim()}");
                }
            }
        }

        Assert.True(offenders.Count == 0,
            "These checks ask about macOS or iOS without deciding about Mac Catalyst, which .NET reports as iOS "
            + "and not as macOS; use HostOS.IsMacDesktop / HostOS.IsAppleMobile, or name IsMacCatalyst():"
            + Environment.NewLine + string.Join(Environment.NewLine, offenders));
    }
}
