// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Diagnostics;
using System.Text.Json;

namespace InferenceWeb.Tests;

public sealed class CudaPtxBuildSettingsTests
{
    [Theory]
    [InlineData(true, null, "compute_120", "compute_86", "compute_61")]
    [InlineData(true, "auto", "compute_120", "compute_86", "compute_61")]
    [InlineData(true, "native", "compute_120", "compute_86", "compute_61")]
    [InlineData(true, "compute_61", "compute_120", "compute_86", "compute_61")]
    [InlineData(true, "sm_61", "compute_120", "compute_86", "compute_61")]
    [InlineData(false, "auto", "compute_120", "compute_61", "compute_120")]
    [InlineData(false, "native", "compute_86", "compute_61", "compute_86")]
    [InlineData(false, "auto", "", "compute_86", "compute_86")]
    [InlineData(false, "compute_120", "compute_86", "compute_61", "compute_120")]
    public async Task ActualMsbuildResolutionKeepsRefreshPortableAndOrdinaryBuildsGpuSpecific(
        bool refresh, string? requested, string detected, string fallback, string expected)
    {
        var properties = new Dictionary<string, string>
        {
            ["TensorSharpUpdateCommittedPtx"] = refresh.ToString(),
            ["_DetectedCudaArch"] = detected,
            ["_DetectCudaArchExitCode"] = "0",
            ["CudaArchFallback"] = fallback,
        };
        if (requested != null) properties["CudaArch"] = requested;
        var result = await RunTarget("ResolveCudaPtxArchitecture", properties);
        Assert.True(result.ExitCode == 0, result.Output + result.Error);
        using var json = JsonDocument.Parse(result.Output);
        var actual = json.RootElement.GetProperty("Properties");
        Assert.Equal(expected, actual.GetProperty("_ResolvedCudaArch").GetString());
        Assert.StartsWith(expected + "_", actual.GetProperty("_CudaBuildSettingsKey").GetString());
    }

    [Theory]
    [InlineData("compute_86")]
    [InlineData("compute_120")]
    [InlineData("sm_120")]
    public async Task RefreshRejectsExplicitNewerTargetsBeforeInvokingNvcc(string requested)
    {
        var result = await RunTarget("ResolveCudaPtxArchitecture", new()
        {
            ["TensorSharpUpdateCommittedPtx"] = "true", ["CudaArch"] = requested,
        });
        Assert.NotEqual(0, result.ExitCode);
        Assert.Contains("must target compute_61", result.Output + result.Error);
        Assert.Contains("omit TensorSharpUpdateCommittedPtx", result.Output + result.Error);
    }

    [Fact]
    public async Task CopyTargetRefusesAnInconsistentResolvedArchitectureBeforeWritingFiles()
    {
        var result = await RunTarget("UpdateCommittedCudaPtx", new()
        {
            ["TensorSharpUpdateCommittedPtx"] = "true", ["_NvccExitCode"] = "0",
            ["_ResolvedCudaArch"] = "compute_120",
        });
        Assert.NotEqual(0, result.ExitCode);
        Assert.Contains("Refusing to overwrite the portable committed CUDA PTX baseline", result.Output + result.Error);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task IntermediatePtxIsCopiedOnlyWhenNvccWasAvailable(bool nvccAvailable)
    {
        string directory = NewDirectory();
        string input = Path.Combine(directory, "intermediate"), output = Path.Combine(directory, "output");
        Directory.CreateDirectory(input);
        File.WriteAllText(Path.Combine(input, "probe.ptx"), ".target sm_120\n");
        try
        {
            var result = await RunTarget("CopyCudaPtxToOutput", new()
            {
                ["_NvccExitCode"] = nvccAvailable ? "0" : "1", ["PtxOutputDir"] = input, ["OutDir"] = output,
            });
            Assert.True(result.ExitCode == 0, result.Output + result.Error);
            Assert.Equal(nvccAvailable, File.Exists(Path.Combine(output, "cuda_kernels", "probe.ptx")));
        }
        finally { Directory.Delete(directory, recursive: true); }
    }

    [Fact]
    public async Task ActualRefreshRecompilesStaleGpuSpecificPtxEvenWithAPortableStamp()
    {
        string directory = NewDirectory();
        string tools = Path.Combine(directory, "tools"), kernels = Path.Combine(directory, "kernels");
        string intermediate = Path.Combine(directory, "intermediate"), stamps = Path.Combine(directory, "stamps");
        string baseline = Path.Combine(directory, "baseline");
        foreach (string path in new[] { tools, kernels, intermediate, stamps, baseline }) Directory.CreateDirectory(path);
        string stub = Path.Combine(tools, OperatingSystem.IsWindows() ? "nvcc.cmd" : "nvcc");
        // cmd.exe splits a batch file's arguments at '=' as well as at spaces, so the target's
        // unquoted -arch=compute_61 arrives as two, -arch and compute_61; a quoted one stays whole.
        if (OperatingSystem.IsWindows())
            File.WriteAllText(stub, """
                @echo off
                if "%~1"=="--version" exit /b 0
                setlocal enabledelayedexpansion
                :next
                if "%~1"=="" goto done
                set "argument=%~1"
                if "!argument:~0,6!"=="-arch=" set "arch=!argument:~6!"
                if "%~1"=="-arch" goto arch
                if "%~1"=="-o" goto output
                shift
                goto next
                :arch
                shift
                set "arch=%~1"
                shift
                goto next
                :output
                shift
                set "out=%~1"
                shift
                goto next
                :done
                if not defined out exit /b 2
                if not defined arch exit /b 3
                > "%out%" echo .target !arch:compute_=sm_!
                exit /b 0
                """.ReplaceLineEndings("\r\n") + "\r\n");
        else
        {
            // A raw literal keeps this source file's line endings, and a tree checked out on Windows
            // has CRLF: "#!/bin/sh\r" cannot exec, so dash keeps searching PATH and runs the real
            // toolkit's nvcc. Like nvcc, the stub rejects a bare argument that is not the kernel.
            File.WriteAllText(stub, """
                #!/bin/sh
                if [ "$1" = "--version" ]; then exit 0; fi
                while [ "$#" -gt 0 ]; do
                    case "$1" in
                        -arch=*) arch=${1#-arch=} ;;
                        -o) shift; output=$1 ;;
                        -*) ;;
                        *.cu) input=$1 ;;
                        *) echo "nvcc error : Don't know what to do with '$1'" >&2; exit 1 ;;
                    esac
                    shift
                done
                [ -n "$output" ] && [ -n "$input" ] || exit 2
                printf '.target sm_%s\n' "${arch#compute_}" > "$output"
                """.ReplaceLineEndings("\n") + "\n");
            File.SetUnixFileMode(stub, UnixFileMode.UserRead | UnixFileMode.UserWrite | UnixFileMode.UserExecute);
        }
        File.WriteAllText(Path.Combine(kernels, "probe.cu"), "// The fake toolchain checks build policy, not CUDA arithmetic.\n");
        string stale = Path.Combine(intermediate, "probe.ptx");
        string stamp = Path.Combine(stamps, "tensorsharp_cuda_compute_61_O3_extradevicevectorization.stamp");
        File.WriteAllText(stale, ".target sm_120\n");
        File.WriteAllText(stamp, "arch=compute_61\n");
        File.SetLastWriteTimeUtc(stale, DateTime.UtcNow.AddMinutes(5));
        File.SetLastWriteTimeUtc(stamp, DateTime.UtcNow.AddMinutes(5));
        try
        {
            // A stub that cannot run is skipped for the next nvcc on PATH, so prove it runs first.
            using (var probe = Process.Start(stub, "--version")!)
                Assert.True(probe.WaitForExit(10_000) && probe.ExitCode == 0, "The fake nvcc does not run by itself.");
            var properties = new Dictionary<string, string>
            {
                ["TensorSharpUpdateCommittedPtx"] = "true", ["CudaArch"] = "auto", ["CudaArchFallback"] = "compute_120",
                ["CudaKernelDir"] = kernels, ["PtxOutputDir"] = intermediate, ["CudaPtxStampDir"] = stamps,
                ["CudaCommittedPtxDir"] = baseline, ["OutDir"] = Path.Combine(directory, "output"),
            };
            // The target sets this whole nvcc flag only on Windows; presetting it skips the vswhere lookup.
            if (OperatingSystem.IsWindows()) properties["_CompilerBinDir"] = "--compiler-bindir=unused-stub-compiler";
            var result = await RunTarget("CompileCudaPtx", properties,
                new() { ["PATH"] = tools + Path.PathSeparator + Environment.GetEnvironmentVariable("PATH") });
            Assert.True(result.ExitCode == 0, result.Output + result.Error);
            Assert.Equal(".target sm_61", File.ReadAllText(Path.Combine(baseline, "probe.ptx")).Trim());
            Assert.Equal(".target sm_61", File.ReadAllText(stale).Trim());
        }
        finally { Directory.Delete(directory, recursive: true); }
    }

    [Fact]
    public async Task NoNvccFallbackReplacesANewerGpuSpecificDestinationThroughSdkCopyTargets()
    {
        string directory = NewDirectory();
        string baseline = Path.Combine(directory, "baseline"), output = Path.Combine(directory, "output");
        string tools = Path.Combine(directory, "tools");
        foreach (string path in new[] { baseline, tools, Path.Combine(output, "cuda_kernels") }) Directory.CreateDirectory(path);
        string stub = Path.Combine(tools, OperatingSystem.IsWindows() ? "nvcc.cmd" : "nvcc");
        File.WriteAllText(stub, OperatingSystem.IsWindows() ? "@exit /b 1\r\n" : "#!/bin/sh\nexit 1\n");
        if (!OperatingSystem.IsWindows())
            File.SetUnixFileMode(stub, UnixFileMode.UserRead | UnixFileMode.UserWrite | UnixFileMode.UserExecute);
        File.WriteAllText(Path.Combine(baseline, "probe.ptx"), ".target sm_61\n");
        string destination = Path.Combine(output, "cuda_kernels", "probe.ptx");
        File.WriteAllText(destination, ".target sm_120\n");
        File.SetLastWriteTimeUtc(destination, DateTime.UtcNow.AddMinutes(5));
        try
        {
            var result = await RunTarget("_CopySourceItemsToOutputDirectory", new()
            {
                ["CudaCommittedPtxDir"] = baseline, ["OutDir"] = output + Path.DirectorySeparatorChar,
                ["_GetChildProjectCopyToOutputDirectoryItems"] = "false", ["BuildProjectReferences"] = "false",
            }, new() { ["PATH"] = tools + Path.PathSeparator + Environment.GetEnvironmentVariable("PATH") });
            Assert.True(result.ExitCode == 0, result.Output + result.Error);
            Assert.Equal(".target sm_61", File.ReadAllText(destination).Trim());
        }
        finally { Directory.Delete(directory, recursive: true); }
    }

    [Theory]
    [InlineData("Cuda compilation tools, release 13.4, V13.4.92", null, "compute_75")]
    [InlineData("Cuda compilation tools, release 12.8, V12.8.93", null, "compute_61")]
    [InlineData("Cuda compilation tools, release 13.4, V13.4.92", "compute_90", "compute_90")]
    public async Task FallbackArchitectureFollowsTheToolkitMajorVersion(string versionLine, string? pinned, string expected)
    {
        string directory = NewDirectory();
        try
        {
            string tools = FakeVersionNvcc(directory, versionLine);
            var properties = new Dictionary<string, string> { ["CudaArch"] = "compute_86" };
            if (pinned != null) properties["CudaArchFallback"] = pinned;
            var result = await RunTarget("DetectCudaPtxBuildSettings", properties,
                new() { ["PATH"] = tools + Path.PathSeparator + Environment.GetEnvironmentVariable("PATH") },
                "_CudaArchFallback,_NvccMajor");
            Assert.True(result.ExitCode == 0, result.Output + result.Error);
            using var json = JsonDocument.Parse(result.Output);
            Assert.Equal(expected, json.RootElement.GetProperty("Properties").GetProperty("_CudaArchFallback").GetString());
        }
        finally { Directory.Delete(directory, recursive: true); }
    }

    [Fact]
    public async Task RefreshWithACuda13ToolkitFailsBeforeCompiling()
    {
        string directory = NewDirectory();
        try
        {
            string tools = FakeVersionNvcc(directory, "Cuda compilation tools, release 13.4, V13.4.92");
            var result = await RunTarget("ResolveCudaPtxBuildSettings", new() { ["TensorSharpUpdateCommittedPtx"] = "true" },
                new() { ["PATH"] = tools + Path.PathSeparator + Environment.GetEnvironmentVariable("PATH") });
            Assert.NotEqual(0, result.ExitCode);
            Assert.Contains("needs a CUDA 12 nvcc", result.Output + result.Error);
        }
        finally { Directory.Delete(directory, recursive: true); }
    }

    [Theory]
    [InlineData("compute_120", "", "true", "-gencode arch=compute_120a,code=sm_120a -gencode arch=compute_120,code=compute_120")]
    [InlineData("sm_86", "", "true", "-gencode arch=compute_86,code=sm_86 -gencode arch=compute_86,code=compute_86")]
    [InlineData("compute_75", "75,86,120", "true",
        "-gencode arch=compute_75,code=sm_75 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_120a,code=sm_120a -gencode arch=compute_75,code=compute_75")]
    [InlineData("compute_120", "", "false", "")]
    public async Task FatbinCarriesSassForTheTargetsPlusThePtx(string arch, string sassList, string emit, string expected)
    {
        // A list reaches MSBuild from the environment: its command line splits -p values on ',' and ';'.
        var environment = sassList.Length > 0 ? new Dictionary<string, string> { ["CudaFatbinArchitectures"] = sassList } : null;
        var result = await RunTarget("ResolveCudaPtxArchitecture", new() { ["CudaArch"] = arch, ["CudaEmitFatbin"] = emit },
            environment, "_CudaFatbinGencode,_CudaBuildSettingsKey");
        Assert.True(result.ExitCode == 0, result.Output + result.Error);
        using var json = JsonDocument.Parse(result.Output);
        var actual = json.RootElement.GetProperty("Properties");
        Assert.Equal(expected, actual.GetProperty("_CudaFatbinGencode").GetString());
        Assert.Equal(expected.Length > 0, actual.GetProperty("_CudaBuildSettingsKey").GetString()!.Contains("_fatbin"));
    }

    [Fact]
    public async Task TheCommittedBaselineRefreshBuildsNoFatbin()
    {
        var result = await RunTarget("ResolveCudaPtxArchitecture", new() { ["TensorSharpUpdateCommittedPtx"] = "true" },
            getProperties: "_CudaFatbinGencode,_ResolvedCudaArch");
        Assert.True(result.ExitCode == 0, result.Output + result.Error);
        using var json = JsonDocument.Parse(result.Output);
        Assert.Equal("", json.RootElement.GetProperty("Properties").GetProperty("_CudaFatbinGencode").GetString());
    }

    [Fact]
    public async Task FatbinAndItsPtxHashAreCopiedBesideThePtx()
    {
        string directory = NewDirectory();
        string input = Path.Combine(directory, "intermediate"), output = Path.Combine(directory, "output");
        Directory.CreateDirectory(input);
        foreach (string name in new[] { "probe.ptx", "probe.fatbin", "probe.fatbin.sha256" })
            File.WriteAllText(Path.Combine(input, name), name);
        try
        {
            var result = await RunTarget("CopyCudaPtxToOutput", new()
            {
                ["_NvccExitCode"] = "0", ["_CudaFatbinGencode"] = "-gencode arch=compute_120,code=sm_120",
                ["PtxOutputDir"] = input, ["OutDir"] = output,
            });
            Assert.True(result.ExitCode == 0, result.Output + result.Error);
            foreach (string name in new[] { "probe.ptx", "probe.fatbin", "probe.fatbin.sha256" })
                Assert.True(File.Exists(Path.Combine(output, "cuda_kernels", name)), name);
        }
        finally { Directory.Delete(directory, recursive: true); }
    }

    [Fact]
    public void ModuleLoaderPrefersOnlyTheFatbinBuiltFromThisPtx()
    {
        string directory = NewDirectory();
        try
        {
            string ptx = Path.Combine(directory, "probe.ptx");
            byte[] ptxBytes = "//\n.version 9.4\n.target sm_120\n"u8.ToArray();
            File.WriteAllBytes(ptx, ptxBytes);
            byte[] image = { 0x50, 0xed, 0x55, 0xba, 1, 2, 3 };
            File.WriteAllBytes(Path.ChangeExtension(ptx, ".fatbin"), image);
            Assert.Null(TensorSharp.Cuda.CudaModule.TryReadMatchingFatbin(ptx, ptxBytes));   // no sidecar

            string sidecar = Path.ChangeExtension(ptx, ".fatbin") + ".sha256";
            File.WriteAllText(sidecar, Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(ptxBytes)).ToLowerInvariant() + "\n");
            Assert.Equal(image, TensorSharp.Cuda.CudaModule.TryReadMatchingFatbin(ptx, ptxBytes));

            byte[] newerPtx = "//\n.version 9.4\n.target sm_120\n// edited\n"u8.ToArray();
            Assert.Null(TensorSharp.Cuda.CudaModule.TryReadMatchingFatbin(ptx, newerPtx));   // stale fatbin
            Assert.Null(TensorSharp.Cuda.CudaModule.TryReadMatchingFatbin(Path.ChangeExtension(ptx, ".bin"), ptxBytes));
        }
        finally { Directory.Delete(directory, recursive: true); }
    }

    private static string FakeVersionNvcc(string directory, string versionLine)
    {
        string tools = Path.Combine(directory, "tools");
        Directory.CreateDirectory(tools);
        string stub = Path.Combine(tools, OperatingSystem.IsWindows() ? "nvcc.cmd" : "nvcc");
        if (OperatingSystem.IsWindows())
            File.WriteAllText(stub, "@echo off\r\nif \"%~1\"==\"--version\" (echo " + versionLine + "& exit /b 0)\r\nexit /b 1\r\n");
        else
        {
            File.WriteAllText(stub, "#!/bin/sh\nif [ \"$1\" = \"--version\" ]; then echo '" + versionLine + "'; exit 0; fi\nexit 1\n");
            File.SetUnixFileMode(stub, UnixFileMode.UserRead | UnixFileMode.UserWrite | UnixFileMode.UserExecute);
        }
        return tools;
    }

    private static string NewDirectory()
    {
        string directory = Path.Combine(RepositoryRoot(), "artifacts", "tests", "CudaPtxBuildSettings", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        return directory;
    }

    private static string RepositoryRoot()
    {
        for (var directory = new DirectoryInfo(AppContext.BaseDirectory); directory != null; directory = directory.Parent)
            if (File.Exists(Path.Combine(directory.FullName, "TensorSharp.slnx"))) return directory.FullName;
        throw new DirectoryNotFoundException("TensorSharp source tree is required for the CUDA build-settings probes.");
    }

    // Execute the real project targets without restoring or compiling managed
    // projects. Architecture probes need no toolchain; the compilation probe
    // uses a fake nvcc and isolated fixture paths, never repository PTX.
    private static async Task<(int ExitCode, string Output, string Error)> RunTarget(
        string target, Dictionary<string, string> properties, Dictionary<string, string>? environment = null,
        string getProperties = "_ResolvedCudaArch,_CudaBuildSettingsKey")
    {
        string root = RepositoryRoot();
        string? host = Environment.GetEnvironmentVariable("DOTNET_HOST_PATH");
        var start = new ProcessStartInfo(string.IsNullOrEmpty(host) ? "dotnet" : host)
        {
            WorkingDirectory = root,
            UseShellExecute = false, CreateNoWindow = true,
            RedirectStandardOutput = true, RedirectStandardError = true,
        };
        foreach (string name in new[] { "CudaArch", "CudaArchFallback", "TensorSharpUpdateCommittedPtx", "TENSORSHARP_CUDA_ARCH",
                     "CudaEmitFatbin", "CudaFatbinArchitectures" })
            start.Environment.Remove(name);
        if (environment != null)
            foreach ((string name, string value) in environment) start.Environment[name] = value;
        foreach (string argument in new[] { "msbuild", Path.Combine(root, "TensorSharp.Backends.Cuda", "TensorSharp.Backends.Cuda.csproj"),
            "-nologo", "-verbosity:quiet", "-target:" + target, "-getProperty:" + getProperties })
            start.ArgumentList.Add(argument);
        foreach ((string name, string value) in properties) start.ArgumentList.Add("-property:" + name + "=" + value);
        using var process = Process.Start(start)!;
        var output = process.StandardOutput.ReadToEndAsync();
        var error = process.StandardError.ReadToEndAsync();
        using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(30));
        try { await process.WaitForExitAsync(timeout.Token); }
        catch (OperationCanceledException)
        {
            process.Kill(entireProcessTree: true);
            Assert.Fail("The CUDA build-settings probe did not complete within 30 seconds.");
        }
        return (process.ExitCode, await output, await error);
    }
}
