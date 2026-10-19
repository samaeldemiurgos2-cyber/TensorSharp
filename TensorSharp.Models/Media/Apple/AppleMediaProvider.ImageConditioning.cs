// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#if MACCATALYST
using System.Threading;
using TensorSharp.Models.Media.Desktop;

namespace TensorSharp.Models.Media.Apple;

public sealed partial class AppleMediaProvider
{
    // VideoToolbox exports MP4s, but its bitrate encoder is not the publisher's
    // CRF-18 libx264 guide transform. The Mac can run the same installed tool as
    // the desktop host while retaining ImageIO and AVFoundation for other media.
    public bool SupportsH264ImageConditioning => FfmpegImageConditioning.IsAvailable;

    public byte[] RecompressH264Image(byte[] rgb8, int width, int height, int crf,
        CancellationToken cancellationToken = default) =>
        FfmpegImageConditioning.RecompressH264Image(rgb8, width, height, crf, cancellationToken);
}
#endif
