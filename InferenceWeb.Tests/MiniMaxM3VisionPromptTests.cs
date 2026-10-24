// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// MiniMax-M3.1 image placeholder expansion: each ]<]image[>[ (200025) becomes
// ]<]start of image[>[ (200029) + N x ]<]image[>[ + ]<]end of image[>[ (200030), exactly
// HF's processing_minimax_m3_vl / llama.cpp mtmd layout, and the span is registered at
// the absolute index of its first image row.
using System;
using System.Collections.Generic;
using System.Linq;
using TensorSharp.Models;
using Xunit;

namespace InferenceWeb.Tests
{
    public class MiniMaxM3VisionPromptTests
    {
        private const int Image = 200025, Start = 200029, End = 200030;

        [Fact]
        public void SingleImage_IsBracketedAndReportsItsFirstRow()
        {
            // ]~b]user\n ]<]image[>[ Describe [e~[
            var tokens = new List<int> { 200034, 200019, 10, Image, 11, 12, 200020 };
            List<int> expanded = ModelMultimodalInjector.ExpandMiniMaxM3ImageTokens(
                tokens, Image, Start, End, new[] { 3 }, out int[] firstRows);

            Assert.Equal(new[] { 200034, 200019, 10, Start, Image, Image, Image, End, 11, 12, 200020 }, expanded);
            Assert.Equal(new[] { 4 }, firstRows);
        }

        [Fact]
        public void TwoImages_GetTheirOwnCountsAndPositions()
        {
            var tokens = new List<int> { 1, Image, 2, Image };
            List<int> expanded = ModelMultimodalInjector.ExpandMiniMaxM3ImageTokens(
                tokens, Image, Start, End, new[] { 2, 4 }, out int[] firstRows);

            Assert.Equal(new[] { 1, Start, Image, Image, End, 2, Start, Image, Image, Image, Image, End }, expanded);
            Assert.Equal(new[] { 2, 7 }, firstRows);
            for (int i = 0; i < firstRows.Length; i++)
            {
                Assert.Equal(Start, expanded[firstRows[i] - 1]);
                Assert.Equal(End, expanded[firstRows[i] + new[] { 2, 4 }[i]]);
            }
        }

        [Fact]
        public void AdjacentImages_EachGetAWrapper()
        {
            var tokens = new List<int> { Image, Image };
            List<int> expanded = ModelMultimodalInjector.ExpandMiniMaxM3ImageTokens(
                tokens, Image, Start, End, new[] { 1, 1 }, out int[] firstRows);
            Assert.Equal(new[] { Start, Image, End, Start, Image, End }, expanded);
            Assert.Equal(new[] { 1, 4 }, firstRows);
        }

        [Fact]
        public void AlreadyBracketedPlaceholder_GainsNoSecondWrapper()
        {
            var tokens = new List<int> { 7, Start, Image, End, 8 };
            List<int> expanded = ModelMultimodalInjector.ExpandMiniMaxM3ImageTokens(
                tokens, Image, Start, End, new[] { 2 }, out int[] firstRows);
            Assert.Equal(new[] { 7, Start, Image, Image, End, 8 }, expanded);
            Assert.Equal(new[] { 2 }, firstRows);
        }

        [Fact]
        public void PlaceholderAndImageCountsMustAgree()
        {
            Assert.Throws<InvalidOperationException>(() => ModelMultimodalInjector.ExpandMiniMaxM3ImageTokens(
                new List<int> { Image, Image }, Image, Start, End, new[] { 4 }, out _));
            Assert.Throws<InvalidOperationException>(() => ModelMultimodalInjector.ExpandMiniMaxM3ImageTokens(
                new List<int> { Image }, Image, Start, End, new[] { 4, 4 }, out _));
            Assert.Throws<ArgumentOutOfRangeException>(() => ModelMultimodalInjector.ExpandMiniMaxM3ImageTokens(
                new List<int> { Image }, Image, Start, End, new[] { 0 }, out _));
        }

        [Fact]
        public void ExpansionLength_IsNPlusTwoPerImage()
        {
            var tokens = Enumerable.Range(0, 50).ToList();
            tokens[10] = Image;
            tokens[30] = Image;
            List<int> expanded = ModelMultimodalInjector.ExpandMiniMaxM3ImageTokens(
                tokens, Image, Start, End, new[] { 540, 576 }, out _);
            Assert.Equal(50 - 2 + (540 + 2) + (576 + 2), expanded.Count);
        }
    }
}
