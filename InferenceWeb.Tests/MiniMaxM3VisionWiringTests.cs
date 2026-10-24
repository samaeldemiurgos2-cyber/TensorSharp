// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// MiniMax-M3.1 image input through the model's own seams (MiniMaxM3Model.Vision.cs):
// IVisionCapableModel loads the tower, IMultimodalPromptExpander routes the rendered
// ]<]image[>[ to the injector's expansion, and SetVisionEmbeddings turns the
// injector's batch-relative offset into the ABSOLUTE position the native executor
// takes image rows at. Runs on a synthetic minimax-m3 checkpoint parsed without the
// executor (three byte tokens renamed to the image control tokens) and the synthetic
// minimax_m3 mmproj, on the managed CPU backend.
using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Models.Architecture;
using TensorSharp.Models.QwenImage;
using TensorSharp.Runtime;
using Xunit;

namespace InferenceWeb.Tests
{
    [Collection(MiniMaxM3EnvironmentCollection.Name)]
    public sealed class MiniMaxM3VisionWiringTests : IDisposable
    {
        private const int ImageId = 250, StartId = 251, EndId = 252;

        private readonly string _dir = Path.Combine(Path.GetTempPath(), "ts-minimax-m3-vision-" + Guid.NewGuid().ToString("N"));
        private readonly KvCacheDtype _restoreDtype = KvCacheDtypeConfig.Current;
        private readonly bool _restoreExplicit = KvCacheDtypeConfig.IsExplicitlySet;

        public MiniMaxM3VisionWiringTests()
        {
            Directory.CreateDirectory(_dir);
            KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, false);
        }

        public void Dispose()
        {
            KvCacheDtypeConfig.RestoreForTests(_restoreDtype, _restoreExplicit);
            Directory.Delete(_dir, true);
        }

        private MiniMaxM3Model ParseModel()
        {
            var builder = new MiniMaxM3SyntheticModelBuilder
            {
                EditMetadata = kv =>
                {
                    var tokens = (SyntheticGguf.StrArr)kv.Single(k => k.Key == "tokenizer.ggml.tokens");
                    var types = (SyntheticGguf.I32Arr)kv.Single(k => k.Key == "tokenizer.ggml.token_type");
                    var names = tokens.V.ToList();
                    var kinds = types.V.ToArray();
                    names[ImageId] = ModelMultimodalInjector.MiniMaxM3ImageToken;
                    names[StartId] = ModelMultimodalInjector.MiniMaxM3ImageStartToken;
                    names[EndId] = ModelMultimodalInjector.MiniMaxM3ImageEndToken;
                    kinds[ImageId] = kinds[StartId] = kinds[EndId] = 3;   // CONTROL
                    tokens.V = names;
                    types.V = kinds;
                },
            };
            string path = builder.Write(Path.Combine(_dir, Guid.NewGuid().ToString("N") + ".gguf"));
            return new MiniMaxM3Model(path, BackendType.Cpu, 1, null, 1, null, loadExecutor: false);
        }

        private string WritePng(int width, int height)
        {
            var pixels = new float[width * height * 3];
            for (int i = 0; i < pixels.Length; i++)
                pixels[i] = (i * 37 % 255) / 255f;
            string path = Path.Combine(_dir, $"img-{width}x{height}.png");
            File.WriteAllBytes(path, ImageIO.EncodePng(new RgbImage(width, height, pixels)));
            return path;
        }

        private static List<(float[] Rows, int Count, int Position)> PendingRows(MiniMaxM3Model model)
        {
            var field = typeof(MiniMaxM3Model).GetField("_pendingVisionRows", BindingFlags.Instance | BindingFlags.NonPublic)
                ?? throw new InvalidOperationException("MiniMaxM3Model._pendingVisionRows not found.");
            return ((IEnumerable)field.GetValue(model)).Cast<(float[], int, int)>().ToList();
        }

        private static void SetCacheHead(MiniMaxM3Model model, int head)
            => (typeof(ModelBase).GetField("_cacheSeqLen", BindingFlags.Instance | BindingFlags.NonPublic)
                ?? throw new InvalidOperationException("ModelBase._cacheSeqLen not found.")).SetValue(model, head);

        [Fact]
        public void LoadVisionEncoder_RefusesAProjectorOfAnotherWidth()
        {
            using var model = ParseModel();
            string mmproj = MiniMaxM3VisionEncoderTests.WriteMmproj(proj: 24);
            try
            {
                var error = Assert.Throws<ArgumentException>(() => model.LoadVisionEncoder(mmproj));
                Assert.Contains($"hidden size is {model.Config.HiddenSize}", error.Message);
                Assert.False(((IVisionCapableModel)model).IsVisionEncoderLoaded);
                Assert.False(model.HasVisionEncoder());
            }
            finally { File.Delete(mmproj); }
        }

        [Fact]
        public void ImageWithoutTower_IsRefusedWithTheMmprojHint()
        {
            using var model = ParseModel();
            var history = new List<ChatMessage> { new() { Role = "user", Content = "x", ImagePaths = new() { WritePng(56, 56) } } };
            var error = Assert.Throws<InvalidOperationException>(() =>
                model.MultimodalInjector.ProcessPromptTokens(history, new List<int> { 10, ImageId, 11 }));
            Assert.Contains("--mmproj", error.Message);
        }

        [Fact]
        public void ImagePrompt_ExpandsAndQueuesRowsAtTheirAbsolutePositions()
        {
            using var model = ParseModel();
            int hidden = model.Config.HiddenSize;
            string mmproj = MiniMaxM3VisionEncoderTests.WriteMmproj(proj: hidden);
            try
            {
                model.LoadVisionEncoder(mmproj);
                Assert.True(((IVisionCapableModel)model).IsVisionEncoderLoaded);
                Assert.True(model.HasVisionEncoder());

                // 112x56 -> a 4x8 patch grid -> 2x4 merge blocks -> 8 image rows.
                string png = WritePng(112, 56);
                var image = new MiniMaxM3ImageProcessor(model.VisionEncoder.PatchSize, model.VisionEncoder.SpatialMergeSize)
                    .ProcessImage(png);
                float[] expectedRows;
                using (Tensor direct = model.VisionEncoder.Encode(image))
                    expectedRows = direct.GetElementsAsFloat((int)direct.ElementCount());
                int n = expectedRows.Length / hidden;
                Assert.Equal(8, n);

                var history = new List<ChatMessage> { new() { Role = "user", Content = "x", ImagePaths = new() { png } } };
                List<int> expanded = model.MultimodalInjector.ProcessPromptTokens(history, new List<int> { 10, 11, ImageId, 12 });
                var want = new List<int> { 10, 11, StartId };
                want.AddRange(Enumerable.Repeat(ImageId, n));
                want.AddRange(new[] { EndId, 12 });
                Assert.Equal(want, expanded);

                // A context trim keeps or drops the image together with both wrappers
                // (start at 2, rows 3..10, end at 11).
                Assert.Equal(2, model.MultimodalInjector.ClampTrimStart(2));
                Assert.Equal(12, model.MultimodalInjector.ClampTrimStart(3));
                Assert.Equal(12, model.MultimodalInjector.ClampTrimStart(11));
                Assert.Equal(12, model.MultimodalInjector.ClampTrimStart(12));

                // Whole prompt from an empty cache: the span starts at prompt index 3.
                Assert.True(model.MultimodalInjector.QueuePromptEmbeddings(0));
                var pending = PendingRows(model);
                Assert.Single(pending);
                Assert.Equal(3, pending[0].Position);
                Assert.Equal(n, pending[0].Count);
                Assert.Equal(expectedRows, pending[0].Rows);

                // A chunked prefill: the cache head sits at 5 (the first chunk covered the
                // wrapper and two image rows); the next chunk [5, 9) carries rows 2..5,
                // handed over relative to the chunk and queued at absolute position 5.
                model.ResetKVCache();
                Assert.False(model.HasPendingVisionRows);
                SetCacheHead(model, 5);
                Assert.True(model.MultimodalInjector.QueuePromptEmbeddingsForSlice(5, 4));
                pending = PendingRows(model);
                Assert.Single(pending);
                Assert.Equal(5, pending[0].Position);
                Assert.Equal(4, pending[0].Count);
                Assert.Equal(expectedRows.Skip(2 * hidden).Take(4 * hidden), pending[0].Rows);

                // The batch-relative offset can never point into the cached prefix.
                using var stray = new Tensor(new TensorSharp.Cpu.CpuAllocator(BlasEnum.DotNet), DType.Float32, 1, hidden);
                Assert.Throws<ArgumentOutOfRangeException>(() => model.SetVisionEmbeddings(stray, -1));
            }
            finally { File.Delete(mmproj); }
        }
    }
}
