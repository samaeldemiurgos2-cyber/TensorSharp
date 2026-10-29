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
using System.Linq;

namespace TensorSharp.Server.Skills
{
    /// <summary>
    /// Shortens the tool results one skills round produced so the NEXT round's prompt still
    /// fits the model's context window, the way an agent framework bounds an observation:
    /// each result keeps its head and says, in the conversation, how much was cut and how to
    /// get the rest.
    ///
    /// <para>
    /// Observed with the CLI on Qwen3.5-0.8B under MAX_CONTEXT=4096: a 1,991-token first round
    /// asked for skills_read of a 20 KB SKILL.md, and the second round's prompt came to 6,418
    /// tokens. The engine was handed a prompt it could not hold and the process aborted. A
    /// read cap alone (<see cref="SkillRequestPlan.ReadCapFor"/>) does not settle it: the cap
    /// is per call, a round may make several, and what fits depends on how much of the window
    /// the conversation has already used.
    /// </para>
    /// <para>
    /// Lengths are measured by the caller's own renderer (<c>countPromptTokens</c>), so chat
    /// template framing, the tool declarations and the earlier rounds are all counted exactly;
    /// characters are only the unit the results are cut in. What the renderer does not see -
    /// the rows a picture, clip or video frame expands to in the prefill, where the template
    /// writes one placeholder - the caller takes off <c>promptTokenLimit</c>.
    /// </para>
    /// </summary>
    internal static class ToolResultContextBudget
    {
        /// <summary>How many times a cut is re-measured and tightened before every result is
        /// reduced to its note. Tokenization is not exactly proportional to characters.</summary>
        private const int MaxAttempts = 8;

        /// <summary>
        /// Return <paramref name="contents"/>, shortened where needed so that
        /// <paramref name="countPromptTokens"/> of the result is at most
        /// <paramref name="promptTokenLimit"/>. Results that fit are returned unchanged (the
        /// same list when nothing was cut). The shorter results keep all of their text and the
        /// longer ones share what is left; a result that cannot keep any text is replaced by a
        /// note saying it was omitted, unless the note would be the longer of the two. When even
        /// that shortest form does not fit, it is returned: the caller's prompt is over budget
        /// for reasons the results cannot fix, and the engine's context checks take it from there.
        /// </summary>
        /// <param name="contents">This round's tool results, in order.</param>
        /// <param name="promptTokenLimit">The longest prompt <paramref name="countPromptTokens"/>
        /// may return: the context window less the room kept for the reply and less any rows the
        /// prefill adds that the renderer does not see. Zero or less disables the check.</param>
        /// <param name="contextTokens">The model's context window, quoted in the notes.</param>
        /// <param name="countPromptTokens">Renders the next round's prompt with the given
        /// results in place of these and returns its token count.</param>
        /// <param name="truncated">True when any result was cut or omitted.</param>
        public static IReadOnlyList<string> Fit(
            IReadOnlyList<string> contents,
            int promptTokenLimit,
            int contextTokens,
            Func<IReadOnlyList<string>, int> countPromptTokens,
            out bool truncated)
        {
            ArgumentNullException.ThrowIfNull(contents);
            ArgumentNullException.ThrowIfNull(countPromptTokens);
            truncated = false;
            if (promptTokenLimit <= 0 || contents.Count == 0)
                return contents;

            int full = countPromptTokens(contents);
            if (full <= promptTokenLimit)
                return contents;

            string[] originals = contents.Select(c => c ?? string.Empty).ToArray();
            string[] omitted = originals.Select(c => Omitted(c.Length, contextTokens)).ToArray();
            // A result no longer than its note is kept whole: cutting or omitting it would only
            // lengthen the prompt (a 20-character "Error: no such file." is a tenth of its note).
            // Replacing it too made the floor below overstate the shortest prompt, so a round
            // whose short results fit was handed back as all notes, over the limit.
            bool[] whole = originals.Select((c, i) => c.Length <= omitted[i].Length).ToArray();
            if (whole.All(w => w))
                return contents;

            truncated = true;
            string[] shortest = originals.Select((c, i) => whole[i] ? c : omitted[i]).ToArray();
            int floor = countPromptTokens(shortest);
            if (floor >= promptTokenLimit)
                return shortest;

            // Characters to keep of the results that can be cut, first estimated from the
            // tokens their full text cost over their notes, then tightened by re-measuring.
            string[] cuttable = originals.Select((c, i) => whole[i] ? string.Empty : c).ToArray();
            long totalChars = cuttable.Sum(c => (long)c.Length);
            long contentTokens = Math.Max(1, full - floor);
            long budget = totalChars * (promptTokenLimit - floor) / contentTokens;
            for (int attempt = 0; attempt < MaxAttempts && budget > 0; attempt++)
            {
                int[] keep = Allocate(cuttable, budget);
                var candidate = new string[originals.Length];
                for (int i = 0; i < originals.Length; i++)
                {
                    candidate[i] = whole[i] || keep[i] >= originals[i].Length ? originals[i]
                        : keep[i] <= 0 ? omitted[i]
                        : Cut(originals[i], keep[i], contextTokens);
                    // A cut carries a ~225-character note, so a short result cut to a little of
                    // its text can come out longer than it was: keep it whole instead.
                    if (candidate[i].Length >= originals[i].Length)
                        candidate[i] = originals[i];
                }

                int measured = countPromptTokens(candidate);
                if (measured <= promptTokenLimit)
                    return candidate;

                // Over by (measured - limit) tokens: take that share of the kept text off,
                // and at least a tenth, so a stubborn overshoot converges.
                long kept = keep.Sum(k => (long)k);
                long over = measured - promptTokenLimit;
                long removable = Math.Max(1, measured - floor);
                long shrink = Math.Max(kept / 10, kept * over / removable + 1);
                budget = kept - shrink;
            }
            return shortest;
        }

        /// <summary>Share <paramref name="budget"/> characters across the results, smallest
        /// first: a result shorter than an even share keeps all of it and leaves the rest to
        /// the longer ones.</summary>
        internal static int[] Allocate(IReadOnlyList<string> contents, long budget)
        {
            var keep = new int[contents.Count];
            int[] order = Enumerable.Range(0, contents.Count).OrderBy(i => contents[i].Length).ToArray();
            long remaining = Math.Max(0, budget);
            for (int k = 0; k < order.Length; k++)
            {
                int i = order[k];
                long share = remaining / (order.Length - k);
                int give = (int)Math.Min(contents[i].Length, share);
                keep[i] = give;
                remaining -= give;
            }
            return keep;
        }

        /// <summary>The first <paramref name="keep"/> characters of
        /// <paramref name="content"/>, ended at a line break when one is close, plus a note
        /// with what was cut.</summary>
        internal static string Cut(string content, int keep, int contextTokens)
        {
            int end = Math.Clamp(keep, 0, content.Length);
            // Do not split a surrogate pair: the text goes straight into the prompt.
            if (end > 0 && end < content.Length && char.IsLowSurrogate(content[end]))
                end--;
            int lineBreak = end > 0 ? content.LastIndexOf('\n', end - 1) : -1;
            if (lineBreak > 0 && end - lineBreak <= Math.Max(64, end / 10))
                end = lineBreak;
            return content.Substring(0, end).TrimEnd()
                + $"\n\n[Truncated by the host to fit the model's {Window(contextTokens)}: showing the first {end} of "
                + $"{content.Length} characters. Request a smaller part if you need the rest (skills_read takes an "
                + "offset), or continue with what is shown.]";
        }

        /// <summary>The note that replaces a result none of which fits.</summary>
        internal static string Omitted(int length, int contextTokens) =>
            $"[Omitted by the host: this result ({length} characters) does not fit in what is left of the model's "
            + $"{Window(contextTokens)}. Request a smaller part (skills_read takes an offset), or answer with what "
            + "you already have.]";

        private static string Window(int contextTokens) =>
            contextTokens > 0 ? $"{contextTokens}-token context window" : "context window";
    }
}
