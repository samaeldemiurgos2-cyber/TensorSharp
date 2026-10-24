// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// The seam every video-generation model sits behind, so the CLI and the server can
// drive Wan, MiniMax-H3 and anything added later through one path instead of
// type-testing each concrete model.
using TensorSharp.Models.WanVideo;

namespace TensorSharp.Models.Video
{
    /// <summary>A model that turns a text prompt (plus optional conditioning) into video.</summary>
    public interface IVideoGenerationModel
    {
        /// <summary>Generate a clip. <paramref name="p"/> null = all model defaults.</summary>
        GeneratedVideo GenerateVideo(string prompt, VideoGenerationParams p = null);

        /// <summary>Short family id for UI and telemetry, e.g. "wan" or "minimax-h3".</summary>
        string VideoModelFamily { get; }

        /// <summary>Whether this checkpoint accepts text-to-video without an input frame.</summary>
        bool SupportsTextToVideo => true;

        /// <summary>Recommended frame count for a short interactive preview, on this
        /// model's temporal grid. Zero leaves the frame count to the model's generation
        /// default. Hosts must not infer a frame grid from another model's family.</summary>
        int DefaultPreviewFrames => 0;

        /// <summary>The temporal frame-count grid accepted by this model. Valid requests have
        /// <c>FrameCountOffset + n * FrameCountMultiple</c> frames. The server and clients use
        /// these values to offer the same usable clip lengths without guessing a model family.</summary>
        int FrameCountMultiple => 1;

        /// <summary>Offset of the temporal frame-count grid.</summary>
        int FrameCountOffset => 1;

        /// <summary>Frame rate used when a request does not specify one.</summary>
        int DefaultPreviewFps => 24;

        /// <summary>Default output width offered by clients when the user has not selected a
        /// custom canvas. Zero leaves the output size to the model.</summary>
        int DefaultOutputWidth => 0;

        /// <summary>Default output height offered by clients when the user has not selected a
        /// custom canvas. Zero leaves the output size to the model.</summary>
        int DefaultOutputHeight => 0;

        /// <summary>Pixel grid for output width and height choices shown by clients.</summary>
        int ResolutionMultiple => 16;

        /// <summary>Minimum frame count required by the model's reference workflow, or zero.</summary>
        int MinimumReferenceFrames => 0;

        /// <summary>True when the model generates an audio track jointly with the video.</summary>
        bool SupportsAudio { get; }

        /// <summary>True when the model accepts a first-frame conditioning image.</summary>
        bool SupportsImageConditioning { get; }

        /// <summary>True when the model accepts a last-frame conditioning image.</summary>
        bool SupportsEndImageConditioning { get; }

        /// <summary>True when images can guide arbitrary frame positions. This is
        /// temporal image conditioning, distinct from identity reference adapters.</summary>
        bool SupportsKeyframeConditioning => false;

        /// <summary>True when the model accepts reference images/videos/audio.</summary>
        bool SupportsReferenceConditioning { get; }

        /// <summary>How many reference images or temporal image guides the model accepts,
        /// or 0 when it takes none. The conditioning capability selects their meaning.
        ///
        /// <para>A client that lets someone attach pictures needs the number BEFORE it sends
        /// anything — the alternative is posting ten images and getting the whole request
        /// refused, which is a worse way to learn the limit than a disabled control. The
        /// default answers 0 for every model that does not do reference conditioning, so a
        /// model only implements this if it has a real cap to report.</para></summary>
        int MaxReferenceImages => 0;

        /// <summary>The sampling recipe the model is bound to right now (a step-distillation
        /// LoRA plug-in's, for one), or null when it samples with its own defaults. A request
        /// that contradicts it is refused. Hosts advertise it and bind admitted jobs to its
        /// <see cref="VideoSamplingContract.Id"/>.</summary>
        VideoSamplingContract SamplingContract => null;
    }
}
