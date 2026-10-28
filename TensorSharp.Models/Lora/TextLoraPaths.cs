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
using System.IO;
using System.Text.RegularExpressions;

namespace TensorSharp.Models.Lora
{
    /// <summary>A runtime LoRA request named a file the server will not open.</summary>
    public sealed class LoraPathException : Exception
    {
        public LoraPathException(string message) : base(message) { }
    }

    /// <summary>
    /// Confines runtime adapter loading to the directory the operator named at startup
    /// (<c>--text-lora-dir</c> / <c>TS_TEXT_LORA_DIR</c>). A client names a file relative to
    /// that directory; anything that resolves outside it - <c>..</c> segments, an absolute
    /// path, a symbolic link that leaves it - is refused before the file system is asked
    /// whether it exists, so the refusal says nothing about the host's other files.
    /// </summary>
    public static partial class TextLoraPaths
    {
        /// <summary>Environment variable the host reads for the adapter directory.</summary>
        public const string DirectoryEnvironmentVariable = "TS_TEXT_LORA_DIR";

        /// <summary>The server flag that sets <see cref="DirectoryEnvironmentVariable"/>.</summary>
        public const string DirectoryFlag = "--text-lora-dir";

        /// <summary>Why a load is refused on a server started without a LoRA directory.</summary>
        public static readonly string ResolveWithinDisabledMessage =
            $"Runtime LoRA loading is disabled: start the server with {DirectoryFlag} <directory> " +
            $"(or {DirectoryEnvironmentVariable}) to allow adapters from that directory.";

        [GeneratedRegex("^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")]
        private static partial Regex NamePattern();

        /// <summary>Whether <paramref name="name"/> may name a runtime adapter: 1-64 letters,
        /// digits, '.', '_' or '-', starting with a letter or digit, and not <c>none</c>
        /// (which the <c>lora</c> request field reserves for the base model).</summary>
        public static bool IsValidName(string? name)
            => name != null && NamePattern().IsMatch(name) && !string.Equals(name, "none", StringComparison.OrdinalIgnoreCase);

        /// <summary>
        /// The full path of <paramref name="requested"/> inside <paramref name="root"/>, which
        /// must end in <paramref name="extension"/>.
        /// </summary>
        /// <exception cref="LoraPathException">No directory is configured, or the path leaves it
        /// or has another extension.</exception>
        public static string ResolveWithin(string? root, string requested, string extension)
        {
            if (string.IsNullOrWhiteSpace(root))
                throw new LoraPathException(ResolveWithinDisabledMessage);
            if (string.IsNullOrWhiteSpace(requested) || requested.IndexOf('\0') >= 0)
                throw new LoraPathException("The LoRA path is empty.");
            if (!requested.EndsWith(extension, StringComparison.OrdinalIgnoreCase))
                throw new LoraPathException($"The LoRA path must name a '{extension}' file.");
            if (Path.IsPathRooted(requested))
                throw new LoraPathException("The LoRA path must be relative to the server's LoRA directory.");

            string rootFull = WithSeparator(RealPath(Path.GetFullPath(root)));
            string candidate = Path.GetFullPath(Path.Combine(rootFull, requested));
            if (!IsWithin(candidate, rootFull) || !IsWithin(RealPath(candidate), rootFull))
                throw new LoraPathException("The LoRA path must name a file inside the server's LoRA directory.");
            return candidate;
        }

        /// <summary>Removes the configured directory from a loader message, so an error does not
        /// tell the client where the adapters live on the host.</summary>
        public static string Redact(string message, string? root)
        {
            if (string.IsNullOrEmpty(message) || string.IsNullOrWhiteSpace(root)) return message;
            string rootFull = WithSeparator(Path.GetFullPath(root));
            return message.Replace(rootFull, string.Empty, PathComparison)
                .Replace(rootFull.TrimEnd(Path.DirectorySeparatorChar), "<lora-dir>", PathComparison);
        }

        private static StringComparison PathComparison =>
            OperatingSystem.IsWindows() || OperatingSystem.IsMacOS() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal;

        private static bool IsWithin(string path, string rootWithSeparator)
            => path.StartsWith(rootWithSeparator, PathComparison) && path.Length > rootWithSeparator.Length;

        private static string WithSeparator(string path)
            => Path.EndsInDirectorySeparator(path) ? path : path + Path.DirectorySeparatorChar;

        /// <summary>
        /// <paramref name="fullPath"/> with every symbolic link along it followed, as far as the
        /// path exists. A link inside the directory that points outside it would otherwise pass
        /// the textual check.
        /// </summary>
        private static string RealPath(string fullPath)
        {
            string? rootOfPath = Path.GetPathRoot(fullPath);
            if (string.IsNullOrEmpty(rootOfPath)) return fullPath;
            string current = rootOfPath;
            string[] parts = fullPath[rootOfPath.Length..].Split(
                new[] { Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar }, StringSplitOptions.RemoveEmptyEntries);
            for (int i = 0; i < parts.Length; i++)
            {
                string next = Path.Combine(current, parts[i]);
                FileSystemInfo info = Directory.Exists(next) ? new DirectoryInfo(next) : new FileInfo(next);
                if (!info.Exists)
                {
                    // The rest does not exist yet; nothing below it can be a link.
                    return Path.GetFullPath(Path.Combine(next, string.Join(Path.DirectorySeparatorChar, parts[(i + 1)..])));
                }
                if (info.LinkTarget != null)
                {
                    var target = info.ResolveLinkTarget(returnFinalTarget: true);
                    next = target != null ? Path.GetFullPath(target.FullName) : next;
                }
                current = next;
            }
            return current;
        }
    }
}
