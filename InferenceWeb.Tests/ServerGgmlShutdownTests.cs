// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System.Collections.Concurrent;
using System.Diagnostics;
using System.Globalization;
using System.Net;
using System.Net.Sockets;
using System.Reflection;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace InferenceWeb.Tests;

/// <summary>
/// Exercises the real host's finally and ProcessExit handlers in a fresh process.
/// Loading GgmlOps in the test runner cannot hide an unwanted shutdown P/Invoke in
/// the managed CPU child. No installed model or device is required.
/// </summary>
public sealed class ServerGgmlShutdownTests : IDisposable
{
    private readonly string _dir = Directory.CreateTempSubdirectory("ts-server-shutdown-").FullName;

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { /* best effort */ }
    }

    [UnixHostShutdownFact]
    public Task CpuServer_WithoutNativeLibrary_StopsCleanlyAfterSigterm()
        => AssertStopsCleanly(useNativeBackend: false);

    [UnixHostShutdownFact]
    [Trait("Requires", "GgmlCpu")]
    public Task GgmlCpuServer_WithNativeModel_StopsCleanlyAfterSigterm()
        => AssertStopsCleanly(useNativeBackend: true);

    private async Task AssertStopsCleanly(bool useNativeBackend)
    {
        string host = StageHost(useNativeBackend);
        string model = new DenseDecoderSyntheticModelBuilder { Architecture = "mistral3" }
            .Write(Path.Combine(_dir, "tiny-shutdown.gguf"));
        string backend = useNativeBackend ? "ggml_cpu" : "cpu";
        var start = new ProcessStartInfo(DotnetHost())
        {
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = true,
            WorkingDirectory = _dir,
        };
        foreach (string arg in new[] { host, "--model", model, "--backend", backend,
                     "--urls", "http://127.0.0.1:0", "--no-webui", "--no-skills",
                     "--no-prefix-cache", "--no-multi-agent" })
            start.ArgumentList.Add(arg);
        IsolateEnvironment(start);

        var output = new ConcurrentQueue<string>();
        var listening = new TaskCompletionSource<Uri>(TaskCreationOptions.RunContinuationsAsynchronously);
        using var process = new Process { StartInfo = start };
        process.OutputDataReceived += (_, e) =>
        {
            if (e.Data == null) return;
            output.Enqueue(e.Data);
            Match match = Regex.Match(e.Data, @"Now listening on:\s*(http://127\.0\.0\.1:\d+)");
            if (match.Success)
                listening.TrySetResult(new Uri(match.Groups[1].Value));
        };
        Assert.True(process.Start(), "The isolated server process did not start.");
        process.BeginOutputReadLine();
        Task<string> stderr = process.StandardError.ReadToEndAsync();
        Task exited = process.WaitForExitAsync();
        try
        {
            Task first = await Task.WhenAny(listening.Task, exited).WaitAsync(TimeSpan.FromSeconds(60));
            if (first != listening.Task)
                Assert.Fail($"Server exited before listening (exit {process.ExitCode}).\n{await stderr}\n{Output()}");
            Uri address = await listening.Task;
            using var client = new HttpClient(new HttpClientHandler { UseProxy = false })
            {
                BaseAddress = address,
                Timeout = TimeSpan.FromSeconds(10),
            };
            using HttpResponseMessage health = await client.GetAsync("/health");
            Assert.Equal(HttpStatusCode.OK, health.StatusCode);
            using JsonDocument models = JsonDocument.Parse(await client.GetStringAsync("/v1/models"));
            Assert.Equal("tiny-shutdown", Assert.Single(models.RootElement.GetProperty("data").EnumerateArray())
                .GetProperty("id").GetString());
            // The endpoint advertises the configured path, so also require the
            // successful load log. With ggml_cpu this load constructs a real
            // GgmlContext and native tensors before DI later disposes the model.
            Assert.Contains("Startup model loaded: tiny-shutdown.gguf architecture=mistral3 backend=" + backend,
                Output(), StringComparison.Ordinal);
            if (!useNativeBackend)
            {
                // Native assets and loader paths are excluded from this staged
                // deployment. Linux additionally exposes the child's mappings,
                // catching any system-wide library that satisfied OS probing.
                if (OperatingSystem.IsLinux())
                    Assert.DoesNotContain("GgmlOps", File.ReadAllText($"/proc/{process.Id}/maps"), StringComparison.Ordinal);
            }

            await SendSigterm(process.Id);
            await exited.WaitAsync(TimeSpan.FromSeconds(30));
            string errors = await stderr.WaitAsync(TimeSpan.FromSeconds(10));
            Assert.True(process.ExitCode == 0,
                $"Server exited {process.ExitCode} after SIGTERM.\n{errors}\n{Output()}");
            Assert.DoesNotContain("Unhandled exception", errors + Output(), StringComparison.Ordinal);
            // A new socket avoids the HttpClient's existing keep-alive connection.
            using var socket = new TcpClient();
            using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(5));
            await Assert.ThrowsAsync<SocketException>(async () =>
                await socket.ConnectAsync(IPAddress.Loopback, address.Port, timeout.Token));
        }
        finally
        {
            // Timeout cleanup only: SIGKILL is never counted as a graceful exit.
            if (!process.HasExited)
            {
                try { process.Kill(entireProcessTree: true); }
                catch (InvalidOperationException) when (process.HasExited) { }
                await process.WaitForExitAsync().WaitAsync(TimeSpan.FromSeconds(10));
            }
        }

        string Output() => string.Join('\n', output);
    }

    private string StageHost(bool includeNativeAssets)
    {
        string source = HostOutputDirectory();
        string stage = Directory.CreateDirectory(Path.Combine(_dir, "host")).FullName;
        // GgmlNative also searches every repository ancestor for build outputs.
        // Merely changing cwd or removing the library beside the original DLL
        // does not create the deployment whose shutdown previously aborted.
        for (DirectoryInfo? ancestor = new(stage); ancestor != null; ancestor = ancestor.Parent)
        {
            Assert.False(Directory.Exists(Path.Combine(ancestor.FullName, ".git"))
                || File.Exists(Path.Combine(ancestor.FullName, ".git"))
                || File.Exists(Path.Combine(ancestor.FullName, "TensorSharp.slnx"))
                || File.Exists(Path.Combine(ancestor.FullName, "TensorSharp.sln"))
                || File.Exists(Path.Combine(ancestor.FullName, "Seq2SeqSharp.sln")),
                $"Native-free host staging directory has a repository ancestor: {ancestor.FullName}");
        }
        foreach (string dll in Directory.EnumerateFiles(source, "*.dll"))
        {
            try { _ = AssemblyName.GetAssemblyName(dll); }
            catch (BadImageFormatException) { continue; } // native DLLs on Windows
            File.Copy(dll, Path.Combine(stage, Path.GetFileName(dll)));
        }
        foreach (string suffix in new[] { ".deps.json", ".runtimeconfig.json" })
        {
            string name = "TensorSharp.Server.Host" + suffix;
            File.Copy(Path.Combine(source, name), Path.Combine(stage, name));
        }
        if (includeNativeAssets)
        {
            string nativeName = OperatingSystem.IsMacOS() ? "libGgmlOps.dylib" : "libGgmlOps.so";
            string native = Path.Combine(source, nativeName);
            Assert.True(File.Exists(native), $"Required native host build output not found: {native}");
            File.Copy(native, Path.Combine(stage, nativeName));
        }
        else
        {
            Assert.DoesNotContain(Directory.EnumerateFiles(stage), path =>
                Path.GetFileName(path).Contains("GgmlOps", StringComparison.OrdinalIgnoreCase));
        }
        return Path.Combine(stage, "TensorSharp.Server.Host.dll");
    }

    private void IsolateEnvironment(ProcessStartInfo start)
    {
        // Retain only runtime discovery. In particular, no TS_*/BACKEND/model
        // settings, native loader paths, startup hooks or extra probing paths
        // from a developer shell or the parent test runner reach the child.
        string[] runtimeVariables = { "PATH", "DOTNET_ROOT", "DOTNET_ROOT_X64", "DOTNET_ROOT_ARM64" };
        var runtime = runtimeVariables.Select(name => (Name: name, Value: Environment.GetEnvironmentVariable(name)))
            .Where(pair => !string.IsNullOrEmpty(pair.Value)).ToArray();
        start.Environment.Clear();
        foreach (var pair in runtime)
            start.Environment[pair.Name] = pair.Value;
        foreach (string name in new[] { "HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA", "DOTNET_CLI_HOME",
                     "XDG_DATA_HOME", "XDG_CONFIG_HOME", "XDG_CACHE_HOME", "TMPDIR", "TMP", "TEMP" })
            start.Environment[name] = _dir;
        start.Environment["TENSORSHARP_LOG_LEVEL"] = "Information";
        start.Environment["TENSORSHARP_LOG_DIR"] = Path.Combine(_dir, "logs");
        start.Environment["MAX_CONTEXT"] = "128";
        start.Environment["TS_PREFILL_WARMUP_LEN"] = "8";
        start.Environment["DOTNET_NOLOGO"] = "1";
    }

    private static async Task SendSigterm(int pid)
    {
        var start = new ProcessStartInfo("/bin/kill")
        {
            UseShellExecute = false,
            RedirectStandardError = true,
        };
        start.ArgumentList.Add("-TERM");
        start.ArgumentList.Add(pid.ToString(CultureInfo.InvariantCulture));
        using Process signal = Process.Start(start)!;
        Task<string> stderr = signal.StandardError.ReadToEndAsync();
        await signal.WaitForExitAsync().WaitAsync(TimeSpan.FromSeconds(10));
        Assert.True(signal.ExitCode == 0, "Could not signal the test-owned server: " + await stderr);
    }

    private static string HostOutputDirectory()
    {
        var root = new DirectoryInfo(AppContext.BaseDirectory);
        while (root != null && !File.Exists(Path.Combine(root.FullName, "TensorSharp.slnx")))
            root = root.Parent;
        Assert.NotNull(root);
        string output = Path.Combine(root.FullName, "TensorSharp.Server.Host", "bin");
        Assert.True(File.Exists(Path.Combine(output, "TensorSharp.Server.Host.dll")),
            "Required host build output not found: " + output);
        return output;
    }

    private static string DotnetHost()
    {
        string? configured = Environment.GetEnvironmentVariable("DOTNET_HOST_PATH");
        if (!string.IsNullOrWhiteSpace(configured) && File.Exists(configured))
            return configured;
        string? root = Environment.GetEnvironmentVariable("DOTNET_ROOT");
        if (!string.IsNullOrWhiteSpace(root) && File.Exists(Path.Combine(root, "dotnet")))
            return Path.Combine(root, "dotnet");
        return "dotnet";
    }
}

public sealed class UnixHostShutdownFactAttribute : FactAttribute
{
    public UnixHostShutdownFactAttribute()
    {
        if (!OperatingSystem.IsLinux() && !OperatingSystem.IsMacOS())
            Skip = "Requires Unix SIGTERM delivery to exercise graceful host shutdown.";
    }
}
