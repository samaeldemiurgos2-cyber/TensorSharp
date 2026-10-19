// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Threading;

namespace TensorSharp.Models.Media.Desktop;

public sealed partial class DesktopMediaProvider
{
    public bool SupportsH264ImageConditioning => FfmpegImageConditioning.IsAvailable;

    internal static bool HasH264ConditioningEncoder(string executable) =>
        FfmpegImageConditioning.HasH264ConditioningEncoder(executable);

    public byte[] RecompressH264Image(byte[] rgb8, int width, int height, int crf,
        CancellationToken cancellationToken = default) =>
        FfmpegImageConditioning.RecompressH264Image(rgb8, width, height, crf, cancellationToken);
}
