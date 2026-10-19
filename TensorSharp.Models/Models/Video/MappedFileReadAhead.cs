using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.IO.MemoryMappedFiles;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading;
using Microsoft.Win32.SafeHandles;

namespace TensorSharp.Models.Video;

/// <summary>Reads checkpoint ranges into the OS file cache ahead of a consumer that
/// touches them through a mapping (or uploads them from one), in the consumer's order.
/// On Apple SSDs a cold mapping faults at ~0.9 GiB/s while one sequential reader
/// reaches ~12 GiB/s; a network volume needs many concurrent reads instead (an A40
/// pod's: 0.31 GiB/s with one 8 MiB reader, 2.3 GiB/s with sixteen). The consumer
/// then lands on cached pages. Only the timing of the I/O moves: the consumer reads
/// the same bytes. A chunk is read only once it starts within the window of the stage
/// in use, and chunks whose pages are already cached are not read again.
/// TS_VIDEO_READAHEAD=0 disables it; TS_VIDEO_READAHEAD_WORKERS sets the readers.</summary>
internal sealed partial class MappedFileReadAhead : IDisposable
{
    internal const long DefaultWindowBytes = 4L << 30;
    private const int BufferBytes = 8 << 20;
    private readonly SafeFileHandle _handle;
    private readonly MemoryMappedFile _map;
    private readonly MemoryMappedViewAccessor _view;
    private readonly unsafe byte* _base;
    private readonly (long Offset, long Length)[][] _stages;
    private readonly (long Offset, int Length, long Start)[] _chunks;
    private readonly long[] _starts;
    private readonly long _window;
    private readonly Thread[] _threads;
    private readonly object _sync = new();
    private int _stage, _next = -1, _running;
    private bool _incomplete;
    private volatile bool _stopping;
    private long _readBytes, _residentBytes;
    private readonly long _started;
    internal long PlannedBytes { get; }
    internal int Workers { get; }
    internal long ReadBytes => Interlocked.Read(ref _readBytes);
    internal long ResidentBytes => Interlocked.Read(ref _residentBytes);
    internal double Milliseconds { get; private set; }
    internal bool Completed { get; private set; }
    internal string Failure { get; private set; }

    internal static bool Enabled => Environment.GetEnvironmentVariable("TS_VIDEO_READAHEAD") != "0";

    // Linux model storage is often a network volume that scales with concurrency;
    // one stream already saturates an Apple SSD.
    internal static int DefaultWorkers =>
        int.TryParse(Environment.GetEnvironmentVariable("TS_VIDEO_READAHEAD_WORKERS"), out int configured)
            ? Math.Clamp(configured, 1, 32) : OperatingSystem.IsLinux() ? 16 : 1;

    /// <summary>Starts reading <paramref name="path"/> ahead, or returns null when the
    /// read-ahead is disabled or the file cannot be opened (the consumer then reads
    /// through its mapping as before, which <paramref name="warn"/> is told once).</summary>
    internal static MappedFileReadAhead TryStart(string path, IEnumerable<IEnumerable<(long Offset, long Length)>> stages,
        Action<string> warn = null, long windowBytes = DefaultWindowBytes, int? workers = null, bool honorSwitch = true)
    {
        if ((honorSwitch && !Enabled) || string.IsNullOrEmpty(path)) return null;
        try { return new MappedFileReadAhead(path, stages, windowBytes, workers); }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException)
        {
            warn?.Invoke($"read-ahead of {Path.GetFileName(path)} is unavailable ({error.Message}); " +
                "its weights are read through mapped page faults, which is several times slower on a cold file cache.");
            return null;
        }
    }

    /// <summary>The whole file as one stage.</summary>
    internal static IEnumerable<IEnumerable<(long Offset, long Length)>> WholeFile(string path) =>
        new[] { new[] { (0L, new FileInfo(path).Length) } };

    /// <param name="stages">File ranges per consumer stage, in consumption order.</param>
    private unsafe MappedFileReadAhead(string path, IEnumerable<IEnumerable<(long Offset, long Length)>> stages,
        long windowBytes, int? workers)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(windowBytes);
        _stages = stages.Select(stage => stage.Where(range => range.Length > 0).ToArray()).ToArray();
        _starts = new long[_stages.Length + 1];
        var chunks = new List<(long, int, long)>();
        for (int i = 0; i < _stages.Length; ++i)
        {
            foreach (var (offset, length) in _stages[i])
                for (long done = 0; done < length; done += BufferBytes)
                    chunks.Add((checked(offset + done), (int)Math.Min(BufferBytes, length - done), _starts[i]));
            _starts[i + 1] = checked(_starts[i] + _stages[i].Sum(range => range.Length));
        }
        _chunks = chunks.ToArray();
        PlannedBytes = _starts[^1];
        _window = windowBytes;
        Workers = Math.Clamp(workers ?? DefaultWorkers, 1, 32);
        _handle = File.OpenHandle(path, FileMode.Open, FileAccess.Read, FileShare.Read, FileOptions.SequentialScan);
        try
        {
            // A private read-only mapping only to ask which pages are already cached.
            long length = RandomAccess.GetLength(_handle);
            if (length > 0)
            {
                _map = MemoryMappedFile.CreateFromFile(_handle, null, 0, MemoryMappedFileAccess.Read, HandleInheritability.None, leaveOpen: true);
                _view = _map.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);
                byte* pointer = null;
                _view.SafeMemoryMappedViewHandle.AcquirePointer(ref pointer);
                _base = pointer + _view.PointerOffset;
            }
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException)
        { _view?.Dispose(); _view = null; _map?.Dispose(); _map = null; }
        _started = Stopwatch.GetTimestamp();
        _running = Workers;
        _threads = new Thread[Workers];
        for (int i = 0; i < Workers; ++i)
        {
            _threads[i] = new Thread(Run) { IsBackground = true, Name = "video-readahead" };
            _threads[i].Start();
        }
    }

    /// <summary>The consumer is about to use <paramref name="stage"/>; earlier stages are done.</summary>
    internal void Enter(int stage)
    {
        lock (_sync)
        {
            if (stage <= _stage) return;
            _stage = Math.Min(stage, _stages.Length);
            Monitor.PulseAll(_sync);
        }
    }

    // Workers take chunks in consumption order; each waits until its chunk is
    // inside the window of the stage in use.
    private void Run()
    {
        var buffer = new byte[BufferBytes];
        var residency = new byte[BufferBytes / Environment.SystemPageSize + 2];
        bool finished = false;
        try
        {
            for (int index; (index = Interlocked.Increment(ref _next)) < _chunks.Length;)
            {
                var (offset, length, start) = _chunks[index];
                lock (_sync)
                {
                    while (!_stopping && start - _starts[_stage] > _window) Monitor.Wait(_sync);
                    if (_stopping) return;
                }
                if (Resident(offset, length, residency))
                { Interlocked.Add(ref _residentBytes, length); continue; }
                for (int done = 0; done < length;)
                {
                    if (_stopping) return;
                    int read = RandomAccess.Read(_handle, buffer.AsSpan(done, length - done), checked(offset + done));
                    if (read <= 0) { Failure = $"short read at file offset {offset + done}"; _stopping = true; return; }
                    done += read;
                    Interlocked.Add(ref _readBytes, read);
                }
            }
            finished = true;
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException or ObjectDisposedException)
        { Failure = error.GetType().Name + ": " + error.Message; _stopping = true; }
        finally
        {
            lock (_sync)
            {
                if (!finished) _incomplete = true;
                if (--_running == 0)
                {
                    Completed = !_incomplete;
                    Milliseconds = Stopwatch.GetElapsedTime(_started).TotalMilliseconds;
                }
            }
        }
    }

    private unsafe bool Resident(long offset, int length, byte[] vector)
    {
        if (_base == null) return false;
        long page = Environment.SystemPageSize, aligned = offset - offset % page, bytes = offset - aligned + length;
        int pages = checked((int)((bytes + page - 1) / page));
        try
        {
            fixed (byte* state = vector)
            {
                if (mincore(_base + aligned, (nuint)bytes, state) != 0) return false;
                for (int i = 0; i < pages; ++i) if ((state[i] & 1) == 0) return false;
            }
            return true;
        }
        catch (Exception error) when (error is DllNotFoundException or EntryPointNotFoundException) { return false; }
    }

    [LibraryImport("libc", EntryPoint = "mincore", SetLastError = true)]
    private static unsafe partial int mincore(void* address, nuint length, byte* vector);

    /// <summary>Drops the planned ranges from the OS file cache once the consumer is
    /// done. They are clean pages of a read-only file, so nothing is lost; the next
    /// read-ahead reads them from storage again. Returns the bytes released.</summary>
    internal unsafe long ReleaseCache()
    {
        lock (_sync) if (!_stopping) throw new InvalidOperationException("Stop the read-ahead before releasing its cache.");
        long released = 0, page = Environment.SystemPageSize;
        foreach (var stage in _stages)
            foreach (var (offset, length) in stage)
            {
                // Darwin's msync(MS_INVALIDATE) discards a mapped file's cached pages
                // (macOS and the Mac Catalyst desktop app alike).
                if ((OperatingSystem.IsMacOS() || OperatingSystem.IsMacCatalyst()) && _base != null)
                {
                    long aligned = offset - offset % page;
                    if (msync(_base + aligned, (nuint)(offset - aligned + length), MsInvalidate) == 0) released += length;
                }
                else if (OperatingSystem.IsLinux() &&
                    posix_fadvise((int)_handle.DangerousGetHandle(), offset, length, PosixFadvDontNeed) == 0)
                    released += length;
            }
        return released;
    }

    private const int MsInvalidate = 0x2, PosixFadvDontNeed = 4;

    [LibraryImport("libc", EntryPoint = "msync", SetLastError = true)]
    private static unsafe partial int msync(void* address, nuint length, int flags);

    [LibraryImport("libc", EntryPoint = "posix_fadvise")]
    private static partial int posix_fadvise(int descriptor, long offset, long length, int advice);

    internal string Describe() =>
        $"readahead={ReadBytes / (double)(1L << 30):F2}GiB read+{ResidentBytes / (double)(1L << 30):F2}GiB resident" +
        $"/{PlannedBytes / (double)(1L << 30):F2}GiB in {Milliseconds:F0}ms" + (Workers > 1 ? $" x{Workers}" : "") +
        (Failure != null ? $" failed ({Failure}); the rest was read through mapped page faults" : "");

    /// <summary>Waits until every planned chunk has been read or skipped (or the read failed).</summary>
    internal void Wait()
    {
        foreach (var thread in _threads) thread.Join();
    }

    /// <summary>Stops reading ahead and waits for every reader to finish its current chunk.</summary>
    internal void Stop()
    {
        lock (_sync) { _stopping = true; Monitor.PulseAll(_sync); }
        foreach (var thread in _threads) thread.Join();
    }

    public void Dispose()
    {
        Stop();
        if (_view != null)
        {
            _view.SafeMemoryMappedViewHandle.ReleasePointer();
            _view.Dispose();
        }
        _map?.Dispose();
        _handle.Dispose();
    }
}
