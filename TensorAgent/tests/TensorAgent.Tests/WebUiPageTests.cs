// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System.Text;
using System.Text.Json;
using TensorAgent.Core.JavaScript;
using TensorAgent.Core.Sandbox;

namespace TensorAgent.Tests;

/// <summary>
/// The page, RUN.
///
/// <para>
/// Everything else that guards <c>tensoragent.js</c> reads it as text — does it
/// mention this route, does it parse — and none of that can answer the question
/// that mattered: when a chat with a photo in it is closed and opened again, is the
/// photo there. It was not, for two reasons a source-reading test cannot see. The
/// page sent the model its attachments and told the transcript nothing about them,
/// so nothing was saved; and its own history was flattened to role and text on the
/// way out, so even within one session the second question about a picture was asked
/// with the picture removed.
/// </para>
/// <para>
/// So these load the real script into JavaScriptCore — the same engine family the
/// WebView runs — on top of <c>PageDom.js</c>, a DOM the size of what the script
/// touches, and then do what a person does: attach something, send it, reopen the
/// chat. The assertions are on the request the page made and on the elements it
/// built, because those two are the whole contract between the page, the host and
/// the user.
/// </para>
/// </summary>
[Collection(LivePythonCollection.Name)]
public sealed partial class WebUiPageTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "tensoragent-page-" + Guid.NewGuid().ToString("N"));

    public WebUiPageTests() => Directory.CreateDirectory(_root);

    public void Dispose()
    {
        try { Directory.Delete(_root, true); } catch { }
    }

    // =====================================================================================
    // running the page
    // =====================================================================================

    private static readonly string Repo = FindRepoRoot();

    private static string PageScript =>
        File.ReadAllText(Path.Combine(Repo, "TensorAgent", "src", "TensorAgent.Core", "WebUi", "tensoragent.js"));

    private static string Dom =>
        File.ReadAllText(Path.Combine(Repo, "TensorAgent", "tests", "TensorAgent.Tests", "PageDom.js"));

    /// <summary>
    /// Load the DOM, the routes a launch needs, the page itself, and then
    /// <paramref name="drive"/> — which does the user's part and prints one JSON
    /// object with whatever it wants asserted on.
    /// </summary>
    private JsonElement Run(string routes, string drive)
    {
        // The strings go in ahead of the page, as /i18n.js does in the WebView: the very
        // script the loopback server serves, in English (no test here changes the language).
        string source = Dom + "\n" + Boot(routes) + "\n" + TensorAgent.Core.Localization.PageStrings.Script()
            + "\n" + PageScript + "\n" + Settle(drive);
        var policy = new ExecutionPolicy(
            AllowScripts: true,
            AllowNetwork: false,
            WorkRoot: _root,
            ReadableRoots: Array.Empty<string>(),
            TempRoot: _root)
        {
            DefaultTimeout = TimeSpan.FromSeconds(30),
        };
        var context = new InterpreterContext(_root, new Dictionary<string, string> { ["HOME"] = _root }, policy);

        ExecutionResult result = WebJavaScript.Run(source, _root, context);

        Assert.True(result.ExitCode == 0,
            $"the page script failed to run.{Environment.NewLine}{result.Stdout}{Environment.NewLine}{result.Stderr}");

        int start = result.Stdout.IndexOf("<<RESULT>>", StringComparison.Ordinal);
        Assert.True(start >= 0, "the driver printed no result: " + result.Stdout + result.Stderr);
        return JsonSerializer.Deserialize<JsonElement>(
            result.Stdout[(start + "<<RESULT>>".Length)..].Trim());
    }

    /// <summary>The answers a launch needs before anything a test does can happen.</summary>
    private static string Boot(string routes) => """
        var R = __page.routes;
        R['/api/agent/engine'] = { model: { id: 'gemma', name: 'Gemma', state: 'ready' }, networkDisabledMessage: 'network access is disabled by the user' };
        R['/api/agent/settings'] = { thinkByDefault: false, defaultSkills: [], skillsEnabled: true, maxTokens: 2048 };
        R['/api/models'] = { loaded: 'gemma.gguf', architecture: 'gemma3', backend: 'ggml_metal', visionReady: true };
        R['/api/skills'] = { enabled: true, installable: true, skills: [{ name: 'documents', description: 'make documents' }] };
        R['/api/agent/conversations'] = { conversations: [] };
        // Not a launch: the default is a page coming back inside an app that is
        // already running, which is the case that resumes the last chat. The tests
        // below that hand the page a saved conversation are testing THAT path, and a
        // cold launch would open an empty chat instead and assert nothing.
        R['/api/agent/launch'] = { cold: false };
        R['/api/agent/events'] = {};
        R['/api/sessions?conversation=new'] = { sessionId: 's1', conversationId: 'c1', messages: [], think: false, skills: [] };
        """ + "\n" + routes;

    /// <summary>
    /// Let the page finish booting, then drive it, then print. Everything the page
    /// does is promise-chained off its own boot, so the driver waits by giving the
    /// event loop a few turns rather than by reaching into the page's internals.
    /// </summary>
    /// <remarks>
    /// Two waits are offered to a driver. <c>settle(n)</c> gives the event loop
    /// <c>n</c> turns of one millisecond each, which is how a promise chain is let
    /// run to its end. <c>wait(ms)</c> is a single real <c>setTimeout</c> of that
    /// many milliseconds, for a page timer a test must outlast — a retry the page
    /// schedules 800 ms out is waited for with <c>wait(900)</c>, not with
    /// <c>settle(900)</c>, which would take as long and say nothing about order.
    /// The engine's timers are wall-clock, so both waits are real time; a run has
    /// thirty seconds in total.
    /// </remarks>
    private static string Settle(string drive) => """
        function settle(times) {
          return new Promise(function (done) {
            var left = times;
            (function tick() {
              if (left-- <= 0) { done(); return; }
              setTimeout(tick, 1);
            })();
          });
        }
        function wait(ms) {
          return new Promise(function (done) { setTimeout(done, ms); });
        }
        settle(40)
          .then(function () { return (function () {
        """ + drive + """
          })(); })
          .then(function (value) { return settle(40).then(function () { return value; }); })
          .then(function (value) { console.log('<<RESULT>>' + JSON.stringify(value)); })
          .catch(function (e) { console.log('<<RESULT>>' + JSON.stringify({ error: String((e && e.stack) || e) })); });
        """;

    private static string FindRepoRoot()
    {
        var directory = new DirectoryInfo(AppContext.BaseDirectory);
        while (directory is not null)
        {
            if (Directory.Exists(Path.Combine(directory.FullName, "TensorAgent", "skills")))
                return directory.FullName;
            directory = directory.Parent;
        }
        throw new DirectoryNotFoundException($"no TensorAgent above {AppContext.BaseDirectory}");
    }

    [WebJavaScriptTheory]
    [InlineData(131072, 8192, "128K model context", "8K active")]
    [InlineData(262144, 16384, "256K model context", "16K active")]
    [InlineData(262144, 32768, "256K model context", "32K active")]
    public void HeaderShowsTheModelsOwnContextSeparatelyFromTheActiveLimit(
        int modelContext, int activeContext, string expectedModel, string expectedActive)
    {
        JsonElement result = Run($$"""
            R['/api/models'] = { loaded: 'model.gguf', architecture: 'qwen35',
                loadedBackend: 'ggml_metal', visionReady: true,
                contextTokens: {{activeContext}}, modelContextTokens: {{modelContext}} };
            """, """
            var detail = __page.byId['model'].querySelector('.sub');
            return { detail: detail ? detail.textContent : '' };
            """);

        string detail = result.GetProperty("detail").GetString()!;
        Assert.Contains(expectedModel, detail, StringComparison.Ordinal);
        Assert.Contains(expectedActive, detail, StringComparison.Ordinal);
    }

    [WebJavaScriptFact]
    public void HeaderKeepsCompatibilityWhenTheHostReportsOnlyOneContext()
    {
        JsonElement result = Run("""
            R['/api/models'] = { loaded: 'model.gguf', architecture: 'gemma4',
                loadedBackend: 'ggml_metal', visionReady: true, contextTokens: 8192 };
            """, """
            var detail = __page.byId['model'].querySelector('.sub');
            return { detail: detail ? detail.textContent : '' };
            """);

        string detail = result.GetProperty("detail").GetString()!;
        Assert.Contains("8K context", detail, StringComparison.Ordinal);
        Assert.DoesNotContain("active", detail, StringComparison.Ordinal);
    }

    [WebJavaScriptFact]
    public void HeaderDoesNotPresentAnEffectiveFallbackAsModelMetadata()
    {
        JsonElement result = Run("""
            R['/api/models'] = { loaded: 'model.gguf', architecture: 'unknown',
                loadedBackend: 'ggml_cpu', visionReady: false,
                contextTokens: 4096, modelContextTokens: 0 };
            """, """
            var detail = __page.byId['model'].querySelector('.sub');
            return { detail: detail ? detail.textContent : '' };
            """);

        string detail = result.GetProperty("detail").GetString()!;
        Assert.Contains("4K active context", detail, StringComparison.Ordinal);
        Assert.DoesNotContain("model context", detail, StringComparison.Ordinal);
    }

    // =====================================================================================
    // what a message with attachments carries
    // =====================================================================================

    /// <summary>
    /// A photo, a recording, a clip and a document, attached and sent.
    ///
    /// <para>
    /// Six lists, and the request needs all of them to be right at once: the vision
    /// encoder reads <c>imagePaths</c> (a clip's frames go in there too), the
    /// transcript and the interpreter read <c>attachments</c>, and a document's text
    /// has to be in the content or the model is answering about a file it never saw.
    /// The one that was missing entirely is <c>attachments</c> — which is why nothing
    /// came back — and the one that was WRONG is <c>filePaths</c>, a name no parser on
    /// the other side has ever read, so every PDF and every clip was silently dropped.
    /// </para>
    /// </summary>
    [WebJavaScriptFact]
    public void EveryKindOfAttachmentReachesTheRequestUnderTheNameTheServerReads()
    {
        JsonElement result = Run(string.Empty, """
            window.TensorAgent.addAttachment({ ok: true, file: 'a1.png', fileName: 'cat.png', mediaType: 'image', url: '/uploads/a1.png' });
            window.TensorAgent.addAttachment({ ok: true, file: 'a2.wav', fileName: 'note.wav', mediaType: 'audio', url: '/uploads/a2.wav' });
            window.TensorAgent.addAttachment({ ok: true, file: 'a3.mp4', fileName: 'clip.mp4', mediaType: 'video', url: '/uploads/a3.mp4',
                                               frames: ['a3_0001.png', 'a3_0002.png'] });
            window.TensorAgent.addAttachment({ ok: true, file: 'a4.md', fileName: 'notes.md', mediaType: 'text', url: '/uploads/a4.md',
                                               textContent: 'SECRET-MARKER' });
            __page.byId['text'].value = 'What is in these?';
            __page.byId['send'].dispatch('click');
            return settle(10).then(function () {
              return { sent: __page.requests('/api/chat').map(function (c) { return c.body; }) };
            });
            """);

        JsonElement sent = result.GetProperty("sent");
        Assert.Equal(1, sent.GetArrayLength());
        JsonElement message = sent[0].GetProperty("messages")[0];

        Assert.Equal(new[] { "a1.png", "a3_0001.png", "a3_0002.png" }, Strings(message, "imagePaths"));
        Assert.Equal(new[] { "a1.png" }, Strings(message, "stillImagePaths"));
        Assert.Equal(new[] { "a3.mp4" }, Strings(message, "videoFilePaths"));
        Assert.Equal(new[] { "a2.wav" }, Strings(message, "audioPaths"));
        Assert.Equal(new[] { "a4.md" }, Strings(message, "textFilePaths"));
        Assert.Equal(new[] { "notes.md" }, Strings(message, "textFileNames"));
        Assert.True(message.GetProperty("isVideo").GetBoolean());

        // The document's own text, in the message. Without it the model is answering
        // about a file whose name is all it was ever given.
        string content = message.GetProperty("content").GetString()!;
        Assert.Contains("SECRET-MARKER", content, StringComparison.Ordinal);
        Assert.Contains("[File: notes.md]", content, StringComparison.Ordinal);
        Assert.EndsWith("What is in these?", content, StringComparison.Ordinal);

        // And the chips, which is what a reopened chat is rebuilt from.
        JsonElement attachments = message.GetProperty("attachments");
        Assert.Equal(4, attachments.GetArrayLength());
        Assert.Equal("cat.png", attachments[0].GetProperty("fileName").GetString());
        Assert.Equal("image", attachments[0].GetProperty("mediaType").GetString());
        Assert.Equal(new[] { "a3_0001.png", "a3_0002.png" }, Strings(attachments[2], "frames"));

        // The name that was never read by anything, and must not come back.
        Assert.False(message.TryGetProperty("filePaths", out _),
            "filePaths is not a field any parser on the server reads; anything sent under it is dropped");
    }

    [WebJavaScriptFact]
    public void SelectedFilesUploadTogetherInOrderAndAllReachTheChatRequest()
    {
        JsonElement result = Run("""
            R['/api/upload'] = { ok: true, files: [
              { ok: true, file: 'first.png', fileName: 'photo.png', mediaType: 'image', url: '/uploads/first.png' },
              { ok: true, file: 'second.txt', fileName: 'notes.txt', mediaType: 'text', textContent: 'SECOND-FILE-EVIDENCE' },
              { ok: true, file: 'third.csv', fileName: 'table.csv', mediaType: 'text', fileBacked: true }
            ] };
            """, """
            var input = __page.byId['file-input'];
            input.files = [{ name: 'photo.png' }, { name: 'notes.txt' }, { name: 'table.csv' }];
            input.value = 'selected';
            input.dispatch('change', { target: input });
            return settle(10).then(function () {
              var chips = __page.byId['chips'].querySelectorAll('.nm').map(function (n) { return n.textContent; });
              __page.byId['text'].value = 'Compare all three files';
              __page.byId['send'].dispatch('click');
              return settle(10).then(function () {
                return { uploads: __page.requests('/api/upload'), selected: input.value, chips: chips,
                         sent: __page.requests('/api/chat').map(function (c) { return c.body; }) };
              });
            });
            """);

        JsonElement upload = Assert.Single(result.GetProperty("uploads").EnumerateArray());
        Assert.Equal("POST", upload.GetProperty("method").GetString());
        JsonElement[] parts = upload.GetProperty("parts").EnumerateArray().ToArray();
        Assert.Equal(new[] { "photo.png", "notes.txt", "table.csv" },
            parts.Select(part => part.GetProperty("fileName").GetString()));
        Assert.All(parts, part => Assert.Equal("file", part.GetProperty("name").GetString()));
        Assert.Equal(string.Empty, result.GetProperty("selected").GetString());
        Assert.Equal(new[] { "photo.png", "notes.txt", "table.csv" }, Strings(result, "chips"));
        JsonElement sent = Assert.Single(result.GetProperty("sent").EnumerateArray());
        JsonElement message = sent.GetProperty("messages")[0];
        Assert.Equal(new[] { "first.png" }, Strings(message, "imagePaths"));
        Assert.Equal(new[] { "second.txt", "third.csv" }, Strings(message, "textFilePaths"));
        Assert.Equal(new[] { "notes.txt", "table.csv" }, Strings(message, "textFileNames"));
        Assert.Contains("SECOND-FILE-EVIDENCE", message.GetProperty("content").GetString(), StringComparison.Ordinal);
        Assert.Equal(new[] { "first.png", "second.txt", "third.csv" },
            message.GetProperty("attachments").EnumerateArray().Select(a => a.GetProperty("file").GetString()));
        Assert.True(message.GetProperty("attachments")[2].GetProperty("fileBacked").GetBoolean());
    }

    [WebJavaScriptFact]
    public void SendingDuringQueuedUploadsKeepsTheDraftUntilEverySelectionIsAttached()
    {
        JsonElement result = Run("""
            var uploadGates = [], nextReply = 0;
            var replies = [
              { ok: true, files: [
                { ok: true, file: 'one.txt', fileName: 'one.txt', mediaType: 'text', textContent: 'ONE' },
                { ok: true, file: 'two.txt', fileName: 'two.txt', mediaType: 'text', textContent: 'TWO' }
              ] },
              { ok: true, file: 'three.txt', fileName: 'three.txt', mediaType: 'text', textContent: 'THREE' }
            ];
            R['/api/upload'] = function () { return replies[nextReply++]; };
            var originalFetch = fetch;
            fetch = function (url, init) {
              if (url !== '/api/upload') return originalFetch(url, init);
              return new Promise(function (resolve) {
                uploadGates.push(function () { originalFetch(url, init).then(resolve); });
              });
            };
            """, """
            var input = __page.byId['file-input'];
            input.files = [{ name: 'one.txt' }, { name: 'two.txt' }];
            input.dispatch('change', { target: input });
            input.files = [{ name: 'three.txt' }];
            input.dispatch('change', { target: input });
            __page.byId['text'].value = 'Use every selected file';
            __page.byId['send'].dispatch('click');
            var before = { sent: __page.requests('/api/chat').length, draft: __page.byId['text'].value };
            return settle(10).then(function () {
              before.started = uploadGates.length;
              uploadGates[0]();
              return settle(10);
            }).then(function () {
              __page.byId['send'].dispatch('click');
              var between = { sent: __page.requests('/api/chat').length, count: window.TensorAgent.attachmentCount() };
              uploadGates[1]();
              return settle(10).then(function () {
                __page.byId['send'].dispatch('click');
                return settle(10).then(function () {
                  return { before: before, between: between, notices: __page.notices(),
                           sent: __page.requests('/api/chat').map(function (c) { return c.body; }) };
                });
              });
            });
            """);

        Assert.Equal(0, result.GetProperty("before").GetProperty("sent").GetInt32());
        Assert.Equal("Use every selected file", result.GetProperty("before").GetProperty("draft").GetString());
        Assert.Equal(1, result.GetProperty("before").GetProperty("started").GetInt32());
        Assert.Equal(0, result.GetProperty("between").GetProperty("sent").GetInt32());
        Assert.Equal(2, result.GetProperty("between").GetProperty("count").GetInt32());
        Assert.Contains(Strings(result, "notices"), text => text.Contains("wait for file uploads", StringComparison.Ordinal));
        JsonElement sent = Assert.Single(result.GetProperty("sent").EnumerateArray());
        Assert.Equal(new[] { "one.txt", "two.txt", "three.txt" }, Strings(sent.GetProperty("messages")[0], "textFilePaths"));
    }

    [WebJavaScriptFact]
    public void OneSelectedFileAcceptsTheExistingUploadResponse()
    {
        JsonElement result = Run("""
            R['/api/upload'] = { ok: true, file: 'one.txt', fileName: 'notes.txt', mediaType: 'text',
                                 textContent: 'evidence', warning: 'Read the warning' };
            """, """
            var input = __page.byId['file-input'];
            input.files = [{ name: 'notes.txt' }];
            input.dispatch('change', { target: input });
            return settle(10).then(function () {
              return { uploads: __page.requests('/api/upload'), count: window.TensorAgent.attachmentCount(),
                       notices: __page.notices(), errors: __page.errorNotices() };
            });
            """);

        Assert.Single(result.GetProperty("uploads").EnumerateArray());
        Assert.Equal(1, result.GetProperty("count").GetInt32());
        Assert.Contains("Read the warning", Strings(result, "notices"));
        Assert.Empty(Strings(result, "errors"));
    }

    [WebJavaScriptTheory]
    [InlineData("{ __status: 413, body: { error: 'File too large' } }", "File too large")]
    [InlineData("{ __reject: 'Load failed' }", "Load failed")]
    [InlineData("{ ok: true, file: 'only.txt' }", "did not return every uploaded file")]
    [InlineData("{ ok: true, files: [{ ok: true, file: 'one.txt' }, { ok: false, error: 'bad file' }] }", "did not return every uploaded file")]
    public void FailedOrIncompleteBatchNeverAttachesAPartialSelection(string reply, string expectedError)
    {
        JsonElement result = Run("R['/api/upload'] = " + reply + ";", """
            window.TensorAgent.addAttachment({ ok: true, file: 'existing.txt', fileName: 'existing.txt', mediaType: 'text' });
            var input = __page.byId['file-input'];
            input.files = [{ name: 'one.txt' }, { name: 'two.txt' }];
            input.dispatch('change', { target: input });
            return settle(10).then(function () {
              return { uploads: __page.requests('/api/upload').length, count: window.TensorAgent.attachmentCount(),
                       chips: __page.byId['chips'].querySelectorAll('.nm').map(function (n) { return n.textContent; }),
                       errors: __page.errorNotices() };
            });
            """);

        Assert.Equal(1, result.GetProperty("uploads").GetInt32());
        Assert.Equal(1, result.GetProperty("count").GetInt32());
        Assert.Equal(new[] { "existing.txt" }, Strings(result, "chips"));
        Assert.Contains(expectedError, Assert.Single(Strings(result, "errors")), StringComparison.Ordinal);
    }

    [WebJavaScriptFact]
    public void AnImageStaysInTheComposerWhenTheLoadedModelHasNoVisionProjector()
    {
        JsonElement result = Run("""
            var modelReads = 0;
            R['/api/models'] = function () {
              modelReads++;
              return modelReads === 1
                ? { loaded: 'Qwen3.5-9B-IQ4_XS.gguf', architecture: 'qwen35',
                    loadedBackend: 'ggml_metal', visionReady: true }
                : { loaded: 'Qwen3.5-9B-IQ4_XS.gguf', architecture: 'qwen35',
                    loadedBackend: 'ggml_metal', loadedMmProj: 'stale-or-wrong.gguf', visionReady: false };
            };
            """, """
            window.TensorAgent.addAttachment({ ok: true, file: 'a1.png', fileName: 'chart.png',
                                               mediaType: 'image', url: '/uploads/a1.png' });
            __page.byId['text'].value = 'What does this chart show?';
            __page.byId['send'].dispatch('click');
            return settle(10).then(function () {
              var action = __page.byId['chat'].querySelector('.notice-action');
              if (action) action.dispatch('click');
              return {
                sent: __page.requests('/api/chat').length,
                text: __page.byId['text'].value,
                attachments: window.TensorAgent.attachmentCount(),
                history: window.TensorAgent.history(),
                notice: __page.byId['chat'].textContent,
                routes: __page.requests('/api/agent/events').map(function (c) { return c.body; })
              };
            });
            """);

        Assert.Equal(0, result.GetProperty("sent").GetInt32());
        Assert.Equal("What does this chart show?", result.GetProperty("text").GetString());
        Assert.Equal(1, result.GetProperty("attachments").GetInt32());
        Assert.Equal(0, result.GetProperty("history").GetArrayLength());
        Assert.Contains(result.GetProperty("routes").EnumerateArray(), route =>
            route.TryGetProperty("type", out JsonElement type) && type.GetString() == "open-route" &&
            route.TryGetProperty("route", out JsonElement name) && name.GetString() == "models");
    }

    [WebJavaScriptFact]
    public void ATextOnlyModelCanHandAnImageFileToASelectedHostSkill()
    {
        JsonElement result = Run("""
            R['/api/agent/settings'] = { thinkByDefault: false, defaultSkills: ['documents'],
                                         skillsEnabled: true, maxTokens: 2048 };
            R['/api/models'] = { loaded: 'text-only.gguf', architecture: 'llama',
                                 loadedBackend: 'ggml_metal', visionReady: false,
                                 acceptsVisionProjector: false };
            """, """
            window.TensorAgent.addAttachment({ ok: true, file: 'a1.png', fileName: 'photo.png',
                                               mediaType: 'image', url: '/uploads/a1.png' });
            __page.byId['text'].value = 'Put this photo into a PDF.';
            __page.byId['send'].dispatch('click');
            return settle(10).then(function () {
              return { sent: __page.requests('/api/chat').map(function (c) { return c.body; }) };
            });
            """);

        JsonElement request = Assert.Single(result.GetProperty("sent").EnumerateArray());
        Assert.Equal(new[] { "documents" }, Strings(request, "skills"));
        Assert.Equal(new[] { "a1.png" }, Strings(request.GetProperty("messages")[0], "imagePaths"));
    }

    [WebJavaScriptFact]
    public void AServerVisionRefusalRestoresTheExactImageDraft()
    {
        JsonElement result = Run("""
            R['/api/agent/settings'] = { thinkByDefault: false, defaultSkills: ['summarize'],
                                         skillsEnabled: true, maxTokens: 2048 };
            R['/api/models'] = { loaded: 'text-only.gguf', architecture: 'llama',
                                 loadedBackend: 'ggml_metal', visionReady: false,
                                 acceptsVisionProjector: false };
            R['/api/chat'] = { __status: 400,
                body: { code: 'vision_not_ready', error: 'This skill cannot read the image file.' } };
            """, """
            window.TensorAgent.addAttachment({ ok: true, file: 'a1.png', fileName: 'photo.png',
                                               mediaType: 'image', url: '/uploads/a1.png' });
            __page.byId['text'].value = 'Summarize this image.';
            __page.byId['send'].dispatch('click');
            return settle(15).then(function () {
              return {
                sent: __page.requests('/api/chat').length,
                text: __page.byId['text'].value,
                attachments: window.TensorAgent.attachmentCount(),
                history: window.TensorAgent.history()
              };
            });
            """);

        Assert.Equal(1, result.GetProperty("sent").GetInt32());
        Assert.Equal("Summarize this image.", result.GetProperty("text").GetString());
        Assert.Equal(1, result.GetProperty("attachments").GetInt32());
        Assert.Empty(result.GetProperty("history").EnumerateArray());
    }

    [WebJavaScriptFact]
    public void AnImageDraftIsNotSentWhenTheModelUnloadsDuringCapabilityRefresh()
    {
        JsonElement result = Run("""
            var modelReads = 0;
            R['/api/agent/settings'] = { thinkByDefault: false, defaultSkills: ['documents'],
                                         skillsEnabled: true, maxTokens: 2048 };
            R['/api/models'] = function () {
              modelReads++;
              return modelReads === 1
                ? { loaded: 'text-only.gguf', architecture: 'llama', visionReady: false,
                    acceptsVisionProjector: false }
                : { loaded: null, architecture: null, visionReady: false,
                    acceptsVisionProjector: false };
            };
            """, """
            window.TensorAgent.addAttachment({ ok: true, file: 'a1.png', fileName: 'photo.png',
                                               mediaType: 'image', url: '/uploads/a1.png' });
            __page.byId['text'].value = 'Put this into a PDF.';
            __page.byId['send'].dispatch('click');
            return settle(10).then(function () {
              return {
                sent: __page.requests('/api/chat').length,
                text: __page.byId['text'].value,
                attachments: window.TensorAgent.attachmentCount(),
                history: window.TensorAgent.history()
              };
            });
            """);

        Assert.Equal(0, result.GetProperty("sent").GetInt32());
        Assert.Equal("Put this into a PDF.", result.GetProperty("text").GetString());
        Assert.Equal(1, result.GetProperty("attachments").GetInt32());
        Assert.Empty(result.GetProperty("history").EnumerateArray());
    }

    [WebJavaScriptFact]
    public void AFileBackedCsvKeepsItsPathAndChipWithoutPuttingRowsInThePrompt()
    {
        JsonElement result = Run(string.Empty, """
            window.TensorAgent.addAttachment({ ok: true, file: 'a5.csv', fileName: 'responses.csv',
                                               mediaType: 'text', url: '/uploads/a5.csv', fileBacked: true });
            __page.byId['text'].value = 'Please analyze this form.';
            __page.byId['send'].dispatch('click');
            return { sent: __page.requests('/api/chat').map(function (c) { return c.body; }) };
            """);

        JsonElement message = result.GetProperty("sent")[0].GetProperty("messages")[0];
        Assert.Equal("Please analyze this form.", message.GetProperty("content").GetString());
        Assert.DoesNotContain("[File:", message.GetProperty("content").GetString(), StringComparison.Ordinal);
        Assert.Equal(new[] { "a5.csv" }, Strings(message, "textFilePaths"));
        Assert.Equal(new[] { "responses.csv" }, Strings(message, "textFileNames"));

        JsonElement attachment = Assert.Single(message.GetProperty("attachments").EnumerateArray());
        Assert.Equal("a5.csv", attachment.GetProperty("file").GetString());
        Assert.Equal("responses.csv", attachment.GetProperty("fileName").GetString());
        Assert.Equal("text", attachment.GetProperty("mediaType").GetString());
        Assert.True(attachment.GetProperty("fileBacked").GetBoolean());
        Assert.False(attachment.TryGetProperty("textContent", out _));
    }

    /// <summary>
    /// The second question about a picture is asked with the picture.
    ///
    /// <para>
    /// The history was mapped to <c>{role, content}</c> on its way into the body, so
    /// turn two of "what is in this photo / and what colour is the car" reached the
    /// model with no photo attached to turn one. It answers anyway, which is what made
    /// this expensive to notice.
    /// </para>
    /// </summary>
    [WebJavaScriptFact]
    public void AFollowUpQuestionStillCarriesTheEarlierTurnsPicture()
    {
        JsonElement result = Run("""
            R['/api/chat'] = { __sse: [{ token: 'A cat.' }, { done: true, truncated: false }] };
            """, """
            window.TensorAgent.addAttachment({ ok: true, file: 'a1.png', fileName: 'cat.png', mediaType: 'image', url: '/uploads/a1.png' });
            __page.byId['text'].value = 'What is this?';
            __page.byId['send'].dispatch('click');
            return settle(20).then(function () {
              __page.byId['text'].value = 'And what colour?';
              __page.byId['send'].dispatch('click');
              return settle(20);
            }).then(function () {
              return { sent: __page.requests('/api/chat').map(function (c) { return c.body; }) };
            });
            """);

        JsonElement sent = result.GetProperty("sent");
        Assert.Equal(2, sent.GetArrayLength());
        JsonElement messages = sent[1].GetProperty("messages");
        Assert.Equal(3, messages.GetArrayLength());
        Assert.Equal(new[] { "a1.png" }, Strings(messages[0], "imagePaths"));
        Assert.Equal("A cat.", messages[1].GetProperty("content").GetString());
        Assert.Equal("And what colour?", messages[2].GetProperty("content").GetString());
    }

    /// <summary>
    /// A file a turn produced is a link in the transcript AND a line in the history,
    /// so the next message does not delete it from the saved chat.
    /// </summary>
    [WebJavaScriptFact]
    public void AFileATurnProducedBecomesALinkAndSurvivesInTheHistory()
    {
        JsonElement result = Run("""
            R['/api/chat'] = { __sse: [
              { skill_step: 'shell', skill: 'documents', detail: 'make_pdf.py', ok: true,
                files: [{ name: 'photo.pdf', bytes: 40960, url: '/api/code/artifacts/r/photo.pdf' }] },
              { tool_progress: 'finished', tool: 'shell', seconds: 3 },
              { token: 'Done.' },
              { done: true, truncated: false }
            ] };
            """, """
            __page.byId['text'].value = 'make a pdf';
            __page.byId['send'].dispatch('click');
            return settle(30).then(function () {
              return { transcript: __page.transcript(), history: window.TensorAgent.history() };
            });
            """);

        JsonElement turns = result.GetProperty("transcript");
        JsonElement answer = turns[turns.GetArrayLength() - 1];
        Assert.Contains(answer.GetProperty("media").EnumerateArray(),
            m => m.GetProperty("src").GetString() == "/api/code/artifacts/r/photo.pdf"
                 && m.GetProperty("text").GetString()!.Contains("photo.pdf", StringComparison.Ordinal));

        JsonElement history = result.GetProperty("history");
        JsonElement assistant = history[history.GetArrayLength() - 1];
        Assert.Equal("photo.pdf", assistant.GetProperty("artifacts")[0].GetProperty("name").GetString());
    }

    [WebJavaScriptFact]
    public void AGuardedArtifactAppearsOnlyWhenItsDedicatedVerifiedFrameArrives()
    {
        JsonElement result = Run("""
            R['/api/chat'] = { __sse: [
              { skill_step: 'skills_run', skill: 'documents', detail: 'scripts/make_pptx.py',
                ok: true, files: null },
              { tool_progress: 'finished', tool: 'skills_run', seconds: 3 },
              { artifact_verified: true,
                files: [{ name: 'apple-report.pptx', bytes: 15569,
                          url: '/api/code/artifacts/verified/apple-report.pptx' }] },
              { token: 'Done.' },
              { done: true, truncated: false }
            ] };
            """, """
            __page.byId['text'].value = 'make the Apple report';
            __page.byId['send'].dispatch('click');
            return settle(30).then(function () {
              return { transcript: __page.transcript(), history: window.TensorAgent.history() };
            });
            """);

        JsonElement turns = result.GetProperty("transcript");
        JsonElement answer = turns[turns.GetArrayLength() - 1];
        Assert.Contains(answer.GetProperty("media").EnumerateArray(),
            item => item.GetProperty("src").GetString()
                == "/api/code/artifacts/verified/apple-report.pptx");

        JsonElement history = result.GetProperty("history");
        JsonElement assistant = history[history.GetArrayLength() - 1];
        JsonElement artifact = Assert.Single(assistant.GetProperty("artifacts").EnumerateArray());
        Assert.Equal("apple-report.pptx", artifact.GetProperty("name").GetString());
        Assert.Equal(15569, artifact.GetProperty("bytes").GetInt32());
    }

    // =====================================================================================
    // reopening a saved chat
    // =====================================================================================

    /// <summary>
    /// The whole bug, from the user's side: a saved chat with media in it, reopened.
    ///
    /// <para>
    /// The transcript here is what the host really stores — the same property names,
    /// because <c>StoredMessage</c> is the page's own message shape — and what is
    /// asserted is that each kind of attachment comes back as the thing it is: a
    /// picture as a picture, a recording as something playable, a clip as a clip, a
    /// document as a link that opens it. A row of paperclip labels would satisfy a
    /// weaker test and would not be the file coming back.
    /// </para>
    /// </summary>
    /// <summary>
    /// A file the app uploads reaches the composer, however many lines it has.
    ///
    /// <para>
    /// The regression this exists for: the app used to build
    /// <c>window.TensorAgent.addAttachment({…json…})</c> as a JavaScript SOURCE string
    /// and hand it to MAUI, which wraps every script as
    /// <c>try{JSON.stringify(eval('&lt;script&gt;'))}catch(e){'null'};</c>. The script
    /// therefore became the body of a single-quoted literal, and JSON's <c>\n</c> was
    /// read as that literal's escape — so a text file with two lines produced a real
    /// newline inside an unterminated string, eval threw, MAUI's own catch returned the
    /// STRING "null", and the app discarded it. Upload succeeded, nothing attached,
    /// nothing said. An apostrophe in a file name closed the literal and did the same.
    /// </para>
    /// <para>
    /// So the payload here carries every character that used to break it: newlines, a
    /// double quote, an apostrophe, a backslash and a non-ASCII character — and it
    /// arrives base64'd, which is the fix.
    /// </para>
    /// </summary>
    [WebJavaScriptFact]
    public void AMultiLineFileTheAppUploadsReachesTheComposerIntact()
    {
        string json = JsonSerializer.Serialize(new
        {
            ok = true,
            file = "a1.txt",
            fileName = "Bob's \"notes\".txt",
            mediaType = "text",
            url = "/uploads/a1.txt",
            textContent = "line one\nline two\ttabbed\n\"quoted\" and O'Brien \\ backslash\nnaïve café\n",
        });
        string payload = Convert.ToBase64String(Encoding.UTF8.GetBytes(json));

        JsonElement result = Run("""
            R['/api/chat'] = { __sse: [{ token: 'Read it.' }, { done: true, truncated: false }] };
            """, $$"""
            var answer = window.TensorAgent.__fromHost('addAttachment', '{{payload}}');
            __page.byId['text'].value = 'What is in it?';
            __page.byId['send'].dispatch('click');
            return settle(20).then(function () {
              return {
                answer: answer,
                sent: __page.requests('/api/chat').map(function (c) { return c.body; })
              };
            });
            """);

        Assert.Equal("ok", result.GetProperty("answer").GetString());

        JsonElement body = result.GetProperty("sent")[0];
        JsonElement messages = body.GetProperty("messages");
        string content = messages[messages.GetArrayLength() - 1].GetProperty("content").GetString()!;

        // The file arrived whole: the envelope, every line, and the characters that used
        // to close the literal early.
        Assert.Contains("[File: Bob's \"notes\".txt]", content, StringComparison.Ordinal);
        Assert.Contains("line one\nline two", content, StringComparison.Ordinal);
        Assert.Contains("O'Brien \\ backslash", content, StringComparison.Ordinal);
        Assert.Contains("naïve café", content, StringComparison.Ordinal);
        Assert.Contains("[End of file]", content, StringComparison.Ordinal);
    }

    // =====================================================================================
    // content shared into the app
    // =====================================================================================

    [WebJavaScriptFact]
    public void AShareWaitsForLaunchThenPreservesTheDraftAndAppliesEveryValidPart()
    {
        JsonElement result = Run("""
            document.getElementById('text').value = 'Keep this draft  ';
            R['/api/agent/share/claim'] = function () {
              return { share: {
                id: 'share-1',
                text: 'What can you tell me about this?\n\nShared text:\n\u0022\u0022\u0022\nこんにちは 🌍\nsecond line\n\u0022\u0022\u0022',
                attachments: [
                  { ok: true, file: 'shared.png', fileName: '旅行.png', mediaType: 'image', url: '/uploads/shared.png' },
                  { ok: true, file: 'shared.txt', fileName: 'notes.txt', mediaType: 'text', url: '/uploads/shared.txt', textContent: 'café\nline two' },
                  { ok: false, error: 'broken.mov was not attached.' }
                ],
                notices: ['Only part of the page text fit.'],
                title: 'Travel note',
                newChat: true
              } };
            };
            """, """
            return {
              text: __page.byId['text'].value,
              attachments: window.TensorAgent.attachmentCount(),
              chips: __page.byId['chips'].querySelectorAll('.chip').map(function (c) { return c.textContent; }),
              notices: __page.byId['chat'].textContent,
              sent: __page.requests('/api/chat').length,
              calls: __page.calls.map(function (c) {
                return { path: c.path, method: c.method, type: c.body && c.body.type };
              })
            };
            """);

        const string shared = "What can you tell me about this?\n\nShared text:\n\"\"\"\nこんにちは 🌍\nsecond line\n\"\"\"";
        Assert.Equal("Keep this draft " + shared, result.GetProperty("text").GetString());
        Assert.Equal(2, result.GetProperty("attachments").GetInt32());
        Assert.Equal(new[] { "↗Travel note✕", "Select areaEditing target旅行.png✕", "📄notes.txt✕" },
            result.GetProperty("chips").EnumerateArray().Select(c => c.GetString()).ToArray());
        Assert.Contains("Only part of the page text fit.", result.GetProperty("notices").GetString(), StringComparison.Ordinal);
        Assert.Contains("broken.mov was not attached.", result.GetProperty("notices").GetString(), StringComparison.Ordinal);
        Assert.Equal(0, result.GetProperty("sent").GetInt32());

        var calls = result.GetProperty("calls").EnumerateArray().ToList();
        int launchSession = calls.FindIndex(c => c.GetProperty("path").GetString() == "/api/sessions");
        int ready = calls.FindIndex(c =>
            c.GetProperty("path").GetString() == "/api/agent/events" &&
            c.TryGetProperty("type", out JsonElement type) && type.GetString() == "ready");
        int claim = calls.FindIndex(c => c.GetProperty("path").GetString() == "/api/agent/share/claim");
        int shareSession = calls.FindIndex(claim + 1, c => c.GetProperty("path").GetString() == "/api/sessions");
        Assert.True(launchSession >= 0 && launchSession < ready && ready < claim && claim < shareSession,
            "startup must open its launch chat, announce ready, claim, then await the share's new chat");
        Assert.Equal("POST", calls[claim].GetProperty("method").GetString());
        Assert.DoesNotContain(calls, c => c.GetProperty("path").GetString() == "/api/agent/share/ack");
    }

    [WebJavaScriptFact]
    public void RepeatedClaimsAndConcurrentNudgesDoNotDuplicateAnAppliedShare()
    {
        JsonElement result = Run("""
            document.getElementById('text').value = 'base';
            R['/api/agent/share/claim'] = function () {
              return { share: {
                id: 'share-retry', text: 'shared once', newChat: false,
                attachments: [{ ok: true, file: 'one.pdf', fileName: 'one.pdf', mediaType: 'text', url: '/uploads/one.pdf' }]
              } };
            };
            """, """
            var claimsBefore = __page.requests('/api/agent/share/claim').length;
            var hostAnswer = window.TensorAgent.__fromHost('takeShare', 'e30=');
            document.dispatch('visibilitychange', {});
            var immediateClaims = __page.requests('/api/agent/share/claim').length - claimsBefore;
            return settle(20).then(function () {
              return {
                hostAnswer: hostAnswer,
                immediateClaims: immediateClaims,
                claims: __page.requests('/api/agent/share/claim').length - claimsBefore,
                text: __page.byId['text'].value,
                attachments: window.TensorAgent.attachmentCount(),
                sharedChips: __page.byId['chips'].querySelectorAll('.shared').length,
                sent: __page.requests('/api/chat').length
              };
            });
            """);

        Assert.Equal("ok", result.GetProperty("hostAnswer").GetString());
        Assert.Equal(1, result.GetProperty("immediateClaims").GetInt32());
        Assert.InRange(result.GetProperty("claims").GetInt32(), 1, 2);
        Assert.Equal("base shared once", result.GetProperty("text").GetString());
        Assert.Equal(1, result.GetProperty("attachments").GetInt32());
        Assert.Equal(1, result.GetProperty("sharedChips").GetInt32());
        Assert.Equal(0, result.GetProperty("sent").GetInt32());
    }

    [WebJavaScriptFact]
    public void AShareIsNotAppliedUntilItsRequestedNewChatOpensSuccessfully()
    {
        JsonElement result = Run("""
            document.getElementById('text').value = 'draft';
            var opens = 0;
            R['/api/sessions?conversation=new'] = function () {
              opens++;
              if (opens === 2) return { __status: 500, body: { error: 'not yet' } };
              return { sessionId: 's' + opens, conversationId: 'c' + opens, messages: [], think: false, skills: [] };
            };
            R['/api/agent/share/claim'] = function () {
              return { share: { id: 'share-new-chat', text: 'the shared text', attachments: [], newChat: true } };
            };
            """, """
            var before = {
              text: __page.byId['text'].value,
              sharedChips: __page.byId['chips'].querySelectorAll('.shared').length
            };
            window.TensorAgent.takeShare();
            return settle(20).then(function () {
              return {
                before: before,
                after: {
                  text: __page.byId['text'].value,
                  attachments: window.TensorAgent.attachmentCount(),
                  sharedChips: __page.byId['chips'].querySelectorAll('.shared').length
                }
              };
            });
            """);

        Assert.Equal("draft", result.GetProperty("before").GetProperty("text").GetString());
        Assert.Equal(0, result.GetProperty("before").GetProperty("sharedChips").GetInt32());
        Assert.Equal("draft the shared text", result.GetProperty("after").GetProperty("text").GetString());
        Assert.Equal(0, result.GetProperty("after").GetProperty("attachments").GetInt32());
        Assert.Equal(1, result.GetProperty("after").GetProperty("sharedChips").GetInt32());
    }

    [WebJavaScriptFact]
    public void AShareNeverAutoSendsEvenIfALegacyPayloadRequestsIt()
    {
        JsonElement result = Run("""
            R['/api/agent/share/claim'] = function () {
              return { share: { id: 'share-send', text: 'Send this shared text', attachments: [], newChat: false, autoSend: true } };
            };
            R['/api/chat'] = { __sse: [{ token: 'Done.' }, { done: true, truncated: false }] };
            """, """
            return {
              sent: __page.requests('/api/chat').map(function (c) { return c.body; }),
              text: __page.byId['text'].value,
              sharedChips: __page.byId['chips'].querySelectorAll('.shared').length
            };
            """);

        Assert.Empty(result.GetProperty("sent").EnumerateArray());
        Assert.Equal("Send this shared text", result.GetProperty("text").GetString());
        Assert.Equal(1, result.GetProperty("sharedChips").GetInt32());
    }

    [WebJavaScriptFact]
    public void NavigationPreservesSharedTextAttachmentAndIdUntilTheyAreSent()
    {
        JsonElement result = Run("""
            R['/api/agent/share/claim'] = function () {
              return { share: { id: 'share-nav', text: 'shared across chats', title: 'Shared article', newChat: false,
                attachments: [{ ok: true, file: 'article.png', fileName: 'article.png', mediaType: 'image', url: '/uploads/article.png' }] } };
            };
            R['/api/sessions?conversation=saved'] = { sessionId: 's2', conversationId: 'saved',
              messages: [{ role: 'user', content: 'an older question' }], think: false, skills: [] };
            R['/api/chat'] = { __sse: [{ token: 'Done.' }, { done: true, truncated: false }] };
            """, """
            window.TensorAgent.openConversation('saved');
            return settle(10).then(function () {
              var preserved = {
                text: __page.byId['text'].value,
                attachments: window.TensorAgent.attachmentCount(),
                sharedChips: __page.byId['chips'].querySelectorAll('.shared').length
              };
              __page.byId['send'].dispatch('click');
              return settle(10).then(function () {
                return {
                  preserved: preserved,
                  sent: __page.requests('/api/chat').map(function (c) { return c.body; })
                };
              });
            });
            """);

        JsonElement preserved = result.GetProperty("preserved");
        Assert.Equal("shared across chats", preserved.GetProperty("text").GetString());
        Assert.Equal(1, preserved.GetProperty("attachments").GetInt32());
        Assert.Equal(1, preserved.GetProperty("sharedChips").GetInt32());

        JsonElement request = Assert.Single(result.GetProperty("sent").EnumerateArray());
        Assert.Equal(new[] { "share-nav" }, Strings(request, "shareIds"));
        JsonElement message = request.GetProperty("messages")[1];
        Assert.Equal(new[] { "article.png" }, Strings(message, "stillImagePaths"));
        Assert.Contains("shared across chats", message.GetProperty("content").GetString()!, StringComparison.Ordinal);
    }

    [WebJavaScriptFact]
    public void ExplicitDiscardRemovesOnlyTheTrackedSharedDraft()
    {
        JsonElement result = Run("""
            document.getElementById('text').value = 'keep me';
            R['/api/agent/share/claim'] = function () {
              return { share: { id: 'share-discard', text: 'remove me', title: 'Disposable', newChat: false,
                attachments: [{ ok: true, file: 'discard.png', fileName: 'discard.png', mediaType: 'image', url: '/uploads/discard.png' }] } };
            };
            R['/api/agent/share/discard'] = function () { return { ok: true }; };
            """, """
            __page.byId['text'].value += ' typed later';
            var shared = __page.byId['chips'].querySelector('.shared');
            shared.querySelector('button').dispatch('click');
            return settle(10).then(function () {
              return {
                text: __page.byId['text'].value,
                attachments: window.TensorAgent.attachmentCount(),
                sharedChips: __page.byId['chips'].querySelectorAll('.shared').length,
                discards: __page.requests('/api/agent/share/discard').map(function (c) { return c.body.id; }),
                sent: __page.requests('/api/chat').length
              };
            });
            """);

        Assert.Equal("keep me typed later", result.GetProperty("text").GetString());
        Assert.Equal(0, result.GetProperty("attachments").GetInt32());
        Assert.Equal(0, result.GetProperty("sharedChips").GetInt32());
        Assert.Equal(new[] { "share-discard" }, Strings(result, "discards"));
        Assert.Equal(0, result.GetProperty("sent").GetInt32());
    }

    [WebJavaScriptFact]
    public void IndependentQueuedSharesUseDifferentFreshChatsAndNeverMerge()
    {
        JsonElement result = Run("""
            var offered = 0;
            var sessionOpens = 0;
            R['/api/sessions?conversation=new'] = function () {
              sessionOpens++;
              return { sessionId: 's' + sessionOpens, conversationId: 'c' + sessionOpens,
                messages: [], think: false, skills: [] };
            };
            var shares = [
              // Version-1 envelopes could opt out of a new chat. An envelope boundary
              // is authoritative now, so even these legacy values must stay separate.
              { id: 'share-a', text: 'first', title: 'First', newChat: false,
                attachments: [{ ok: true, file: 'a.txt', fileName: 'a.txt', mediaType: 'text',
                  textContent: 'first attachment', url: '/uploads/a.txt' }] },
              { id: 'share-b', text: 'second', title: 'Second', newChat: false,
                attachments: [{ ok: true, file: 'b.png', fileName: 'b.png', mediaType: 'image', url: '/uploads/b.png' }] }
            ];
            R['/api/agent/share/claim'] = function () { return { share: shares[offered] || null }; };
            R['/api/chat'] = function () {
              offered = 1;
              return { __sse: [{ token: 'Answer.' }, { done: true, truncated: false }] };
            };
            """, """
            var claimsBeforeSend = __page.requests('/api/agent/share/claim').length;
            __page.byId['send'].dispatch('click');
            var sent = __page.requests('/api/chat')[0].body;
            window.TensorAgent.takeShare();
            var whileGenerating = {
              generating: window.TensorAgent.isGenerating(),
              claims: __page.requests('/api/agent/share/claim').length,
              text: __page.byId['text'].value
            };
            return settle(20).then(function () {
              return {
                claimsBeforeSend: claimsBeforeSend,
                whileGenerating: whileGenerating,
                sent: sent,
                sessionOpens: sessionOpens,
                finalClaims: __page.requests('/api/agent/share/claim').length,
                text: __page.byId['text'].value,
                attachments: window.TensorAgent.attachmentCount(),
                history: window.TensorAgent.history(),
                chips: __page.byId['chips'].querySelectorAll('.shared').map(function (c) { return c.textContent; })
              };
            });
            """);

        JsonElement during = result.GetProperty("whileGenerating");
        Assert.True(during.GetProperty("generating").GetBoolean());
        Assert.Equal(result.GetProperty("claimsBeforeSend").GetInt32(), during.GetProperty("claims").GetInt32());
        Assert.Equal(string.Empty, during.GetProperty("text").GetString());
        Assert.Equal(new[] { "share-a" }, Strings(result.GetProperty("sent"), "shareIds"));
        Assert.Equal("s2", result.GetProperty("sent").GetProperty("sessionId").GetString());

        Assert.Equal(result.GetProperty("claimsBeforeSend").GetInt32() + 1,
            result.GetProperty("finalClaims").GetInt32());
        Assert.Equal(3, result.GetProperty("sessionOpens").GetInt32());
        Assert.Equal("second", result.GetProperty("text").GetString());
        Assert.DoesNotContain("first", result.GetProperty("text").GetString()!, StringComparison.Ordinal);
        Assert.Equal(1, result.GetProperty("attachments").GetInt32());
        Assert.Empty(result.GetProperty("history").EnumerateArray());
        Assert.Equal(new[] { "↗Second✕" }, Strings(result, "chips"));
    }

    /// <summary>
    /// Coming back returns to the chat the page was in, not to the newest saved one.
    ///
    /// <para>
    /// The pairing that breaks the obvious implementation: <c>fresh</c> is the empty
    /// chat a launch just opened, and it is not in the conversation list at all — a
    /// chat with no messages is never listed. <c>yesterday</c> is. So a page that
    /// resumed "the most recent conversation" would take the user from the clean
    /// composer they were handed on launch into an old transcript, the first time
    /// WebKit reclaimed the content process behind a trip to the Models screen. Which
    /// is the thing this whole change exists to stop.
    /// </para>
    /// </summary>
    [WebJavaScriptFact]
    public void ComingBackOpensTheChatThePageWasInAndNotTheNewestSavedOne()
    {
        JsonElement result = Run("""
            R['/api/agent/launch'] = { cold: false, conversation: 'fresh' };
            R['/api/agent/conversations'] = { conversations: [{ id: 'yesterday', title: 'Old', updatedAt: '2026-09-01T10:00:00Z', messageCount: 4 }] };
            R['/api/sessions?conversation=fresh'] = {
              sessionId: 's1', conversationId: 'fresh', messages: [], think: false, skills: []
            };
            R['/api/sessions?conversation=yesterday'] = {
              sessionId: 's2', conversationId: 'yesterday', think: false, skills: [],
              messages: [{ role: 'user', content: 'yesterday' }, { role: 'assistant', content: 'indeed' }]
            };
            """, """
            return {
              opened: __page.requests('/api/sessions').map(function (c) { return c.url; }),
              shown: __page.transcript().map(function (t) { return t.text; }).join(' | ')
            };
            """);

        var opened = result.GetProperty("opened").EnumerateArray().Select(u => u.GetString()!).ToList();
        Assert.Contains("/api/sessions?conversation=fresh", opened);
        Assert.DoesNotContain("/api/sessions?conversation=yesterday", opened);
        Assert.DoesNotContain("indeed", result.GetProperty("shown").GetString()!, StringComparison.Ordinal);
    }

    /// <summary>
    /// Launching the app opens a clean chat, not yesterday's.
    ///
    /// <para>
    /// The page used to open the most recent conversation on every load, so the app
    /// always reopened mid-thought — reasonable on a desktop tab that stays open,
    /// wrong for something the user launches from a home screen. It now starts empty
    /// and leaves the previous chat one tap away in the menu.
    /// </para>
    /// <para>
    /// Only on a LAUNCH, which is the half this test exists to pin. The identical page
    /// load happens when WebKit kills the content process of a WebView whose view left
    /// the window, and that one must still come back to the chat the user was reading,
    /// possibly with an answer still being generated for it on the host side. The page
    /// cannot tell the two apart from inside, so it asks; every other test here runs
    /// with the <c>cold: false</c> default and exercises that path.
    /// </para>
    /// </summary>
    [WebJavaScriptFact]
    public void LaunchingTheAppOpensAnEmptyChatRatherThanTheLastOne()
    {
        JsonElement result = Run("""
            R['/api/agent/launch'] = { cold: true };
            R['/api/agent/conversations'] = { conversations: [{ id: 'saved', title: 'Old chat', updatedAt: '2026-09-01T10:00:00Z', messageCount: 2 }] };
            R['/api/sessions?conversation=saved'] = {
              sessionId: 's9', conversationId: 'saved', think: false, skills: [],
              messages: [{ role: 'user', content: 'yesterday' }, { role: 'assistant', content: 'indeed' }]
            };
            """, """
            return {
              shown: __page.transcript().map(function (t) { return t.text; }).join(' | '),
              history: window.TensorAgent.history(),
              opened: __page.requests('/api/sessions').map(function (c) { return c.url; })
            };
            """);

        // The session that was opened is the whole behaviour: "new" is a fresh chat,
        // "saved" is yesterday's. Asserted on the request rather than on an empty
        // transcript, because a fresh chat is not empty on screen — it carries the
        // "New chat" placeholder, which is exactly what should be there.
        var opened = result.GetProperty("opened").EnumerateArray().Select(u => u.GetString()!).ToList();
        Assert.Contains("/api/sessions?conversation=new", opened);
        Assert.DoesNotContain("/api/sessions?conversation=saved", opened);

        // And nothing of yesterday's chat came with it — not on screen, and not in the
        // history the next message would be sent with, which is the half that would
        // quietly carry the old conversation into the new one's context.
        string shown = result.GetProperty("shown").GetString()!;
        Assert.DoesNotContain("yesterday", shown, StringComparison.Ordinal);
        Assert.DoesNotContain("indeed", shown, StringComparison.Ordinal);
        Assert.Equal(0, result.GetProperty("history").GetArrayLength());
    }

    [WebJavaScriptFact]
    public void AReopenedChatShowsItsPicturesSoundsClipsAndDocumentsAgain()
    {
        JsonElement result = Run("""
            R['/api/agent/conversations'] = { conversations: [{ id: 'saved', title: 'Old chat', updatedAt: '2026-09-01T10:00:00Z', messageCount: 2 }] };
            R['/api/sessions?conversation=saved'] = {
              sessionId: 's9', conversationId: 'saved', think: false, skills: [],
              messages: [
                { role: 'user',
                  content: '[File: notes.md]\nBODY\n[End of file]\n\nLook at these',
                  imagePaths: ['a1.png', 'a3_0001.png'],
                  stillImagePaths: ['a1.png'],
                  videoFilePaths: ['a3.mp4'],
                  audioPaths: ['a2.wav'],
                  textFilePaths: ['a4.md'],
                  textFileNames: ['notes.md'],
                  isVideo: true,
                  attachments: [
                    { file: 'a1.png', fileName: 'cat.png', mediaType: 'image' },
                    { file: 'a2.wav', fileName: 'note.wav', mediaType: 'audio' },
                    { file: 'a3.mp4', fileName: 'clip.mp4', mediaType: 'video', frames: ['a3_0001.png'] },
                    { file: 'a4.md', fileName: 'notes.md', mediaType: 'text' },
                    { file: 'a5.heic', fileName: 'IMG_1.heic', mediaType: 'image', previewFile: 'a5-preview.png' }
                  ] },
                { role: 'assistant', content: 'Here it is.', thinking: 'hmm',
                  artifacts: [{ name: 'report.pdf', bytes: 2048, url: '/api/code/artifacts/r/report.pdf' }] }
              ]
            };
            """, """
            return { transcript: __page.transcript(), history: window.TensorAgent.history() };
            """);

        JsonElement turns = result.GetProperty("transcript");
        Assert.Equal(2, turns.GetArrayLength());

        JsonElement user = turns[0];
        var media = user.GetProperty("media").EnumerateArray()
            .Select(m => (Tag: m.GetProperty("tag").GetString()!, Src: m.GetProperty("src").GetString()!))
            .ToList();

        Assert.Contains(media, m => m.Tag == "IMG" && m.Src == "/uploads/a1.png");
        Assert.Contains(media, m => m.Tag == "AUDIO" && m.Src == "/uploads/a2.wav");
        Assert.Contains(media, m => m.Tag == "VIDEO" && m.Src == "/uploads/a3.mp4");
        Assert.Contains(media, m => m.Tag == "A" && m.Src == "/uploads/a4.md");
        // A HEIC is shown through the PNG the host wrote beside it: no browser
        // renders HEIC, so pointing at the original is a broken image with a 200.
        Assert.Contains(media, m => m.Tag == "IMG" && m.Src == "/uploads/a5-preview.png");

        // The user's own sentence, not the file they attached, in the bubble. The
        // page renders Markdown into innerHTML, so that is where the words are.
        string bubble = user.GetProperty("html").GetString()!;
        Assert.Contains("Look at these", bubble, StringComparison.Ordinal);
        Assert.DoesNotContain("End of file", bubble, StringComparison.Ordinal);

        // The file the turn produced is a link again, not a sentence about a link.
        Assert.Contains(turns[1].GetProperty("media").EnumerateArray(),
            m => m.GetProperty("src").GetString() == "/api/code/artifacts/r/report.pdf");

        // And the history the next request is built from kept every path, so a
        // follow-up question is asked with the same attachments the user can see.
        JsonElement history = result.GetProperty("history");
        Assert.Equal(2, history.GetArrayLength());
        Assert.Equal(new[] { "a1.png", "a3_0001.png" }, Strings(history[0], "imagePaths"));
        Assert.Equal(new[] { "a2.wav" }, Strings(history[0], "audioPaths"));
        Assert.Equal("hmm", history[1].GetProperty("thinking").GetString());
        Assert.Equal("report.pdf", history[1].GetProperty("artifacts")[0].GetProperty("name").GetString());
    }

    /// <summary>
    /// Reopening a chat and then saying something else does not erase what the host
    /// saved about it.
    ///
    /// <para>
    /// The host writes the whole <c>messages</c> array it is sent over the stored
    /// copy, on the reasonable ground that the page is the authority on what the
    /// conversation contains. That makes any field the page forgets a field the next
    /// message DELETES from disk — so the answer's reasoning and the PDF it produced
    /// survive being read and are then thrown away by the follow-up question. This is
    /// the round trip, in one test.
    /// </para>
    /// </summary>
    [WebJavaScriptFact]
    public void TheNextMessageDoesNotWipeTheAttachmentsAndFilesAlreadySaved()
    {
        JsonElement result = Run("""
            R['/api/agent/conversations'] = { conversations: [{ id: 'saved', title: 'Old', updatedAt: '2026-09-01T10:00:00Z', messageCount: 2 }] };
            R['/api/sessions?conversation=saved'] = {
              sessionId: 's9', conversationId: 'saved', think: false, skills: [],
              messages: [
                { role: 'user', content: 'Look', imagePaths: ['a1.png'], stillImagePaths: ['a1.png'],
                  attachments: [{ file: 'a1.png', fileName: 'cat.png', mediaType: 'image' }] },
                { role: 'assistant', content: 'A cat.', thinking: 'hmm',
                  artifacts: [{ name: 'report.pdf', bytes: 2048, url: '/api/code/artifacts/r/report.pdf' }] }
              ]
            };
            """, """
            __page.byId['text'].value = 'Anything else?';
            __page.byId['send'].dispatch('click');
            return settle(10).then(function () {
              return { sent: __page.requests('/api/chat').map(function (c) { return c.body; }) };
            });
            """);

        JsonElement messages = result.GetProperty("sent")[0].GetProperty("messages");
        Assert.Equal(3, messages.GetArrayLength());
        Assert.Equal(new[] { "a1.png" }, Strings(messages[0], "imagePaths"));
        Assert.Equal("cat.png", messages[0].GetProperty("attachments")[0].GetProperty("fileName").GetString());
        Assert.Equal("hmm", messages[1].GetProperty("thinking").GetString());
        Assert.Equal("report.pdf", messages[1].GetProperty("artifacts")[0].GetProperty("name").GetString());
    }

    // =====================================================================================
    // the skills switch
    // =====================================================================================

    /// <summary>
    /// Off means off: the setting is saved, the selection is dropped, and the next
    /// message names no skill.
    /// </summary>
    [WebJavaScriptFact]
    public void TurningSkillsOffSavesTheSettingAndSendsNoSkills()
    {
        JsonElement result = Run("""
            R['/api/agent/settings'] = function (call) {
              return call.method === 'POST' ? call.body
                : { thinkByDefault: false, defaultSkills: ['documents'], skillsEnabled: true, maxTokens: 2048 };
            };
            """, """
            var before = [];
            __page.byId['text'].value = 'first';
            __page.byId['send'].dispatch('click');
            before = __page.requests('/api/chat').map(function (c) { return c.body; });

            __page.byId['skills-master'].checked = false;
            __page.byId['skills-master'].dispatch('change');
            return settle(20).then(function () {
              __page.byId['text'].value = 'second';
              __page.byId['send'].dispatch('click');
              return {
                before: before,
                after: __page.requests('/api/chat').map(function (c) { return c.body; }),
                saved: __page.requests('/api/agent/settings').filter(function (c) { return c.method === 'POST'; })
                         .map(function (c) { return c.body; }),
                chips: __page.byId['skillchips'].children.length,
                listClass: __page.byId['skills-list'].className,
              };
            });
            """);

        // With it on, the chat request names the default skill.
        Assert.Equal(new[] { "documents" }, Strings(result.GetProperty("before")[0], "skills"));

        // Saved as a setting, not merely held in the page: it is a preference, and it
        // is what the host reads to build no skill plan at all.
        JsonElement saved = result.GetProperty("saved");
        Assert.True(saved.GetArrayLength() >= 1, "turning the switch off saved nothing");
        Assert.False(saved[saved.GetArrayLength() - 1].GetProperty("skillsEnabled").GetBoolean());

        JsonElement after = result.GetProperty("after");
        Assert.Equal(2, after.GetArrayLength());
        Assert.False(after[1].TryGetProperty("skills", out _),
            "the message after the switch went off still named a skill");
        Assert.False(after[1].GetProperty("skills_discovery").GetBoolean());
        Assert.Equal(0, result.GetProperty("chips").GetInt32());
        Assert.Contains("off", result.GetProperty("listClass").GetString()!, StringComparison.Ordinal);
    }

    /// <summary>
    /// A chat saved when skills were on does not switch them back on for itself.
    /// The setting is the wider fact; a stored selection is a memory of one chat.
    /// </summary>
    [WebJavaScriptFact]
    public void ASavedChatDoesNotReviveSkillsThatWereTurnedOff()
    {
        JsonElement result = Run("""
            R['/api/agent/settings'] = { thinkByDefault: false, defaultSkills: [], skillsEnabled: false, maxTokens: 2048 };
            R['/api/skills'] = { enabled: false, installable: true, skills: [{ name: 'documents', description: 'd' }] };
            R['/api/agent/conversations'] = { conversations: [{ id: 'saved', title: 'Old', updatedAt: '2026-09-01T10:00:00Z', messageCount: 1 }] };
            R['/api/sessions?conversation=saved'] = {
              sessionId: 's9', conversationId: 'saved', think: false, skills: ['documents'],
              messages: [{ role: 'user', content: 'hello' }]
            };
            """, """
            __page.byId['text'].value = 'again';
            __page.byId['send'].dispatch('click');
            return { sent: __page.requests('/api/chat').map(function (c) { return c.body; }),
                     chips: __page.byId['skillchips'].children.length };
            """);

        Assert.Equal(1, result.GetProperty("sent").GetArrayLength());
        Assert.False(result.GetProperty("sent")[0].TryGetProperty("skills", out _),
            "a saved chat re-selected a skill after the feature was switched off");
        Assert.False(result.GetProperty("sent")[0].GetProperty("skills_discovery").GetBoolean());
        Assert.Equal(0, result.GetProperty("chips").GetInt32());
    }

    [WebJavaScriptFact]
    public void DeselectingEverySkillSendsAnExplicitEmptySelection()
    {
        JsonElement result = Run("""
            R['/api/agent/settings'] = {
              thinkByDefault: false, defaultSkills: ['documents'], skillsEnabled: true, maxTokens: 2048
            };
            """, """
            window.TensorAgent.setSkills([]);
            __page.byId['text'].value = 'search something and create a PowerPoint report';
            __page.byId['send'].dispatch('click');
            return { sent: __page.requests('/api/chat').map(function (c) { return c.body; }) };
            """);

        JsonElement request = Assert.Single(result.GetProperty("sent").EnumerateArray());
        Assert.True(request.TryGetProperty("skills", out JsonElement skills));
        Assert.Equal(JsonValueKind.Array, skills.ValueKind);
        Assert.Empty(skills.EnumerateArray());
        Assert.False(request.GetProperty("skills_discovery").GetBoolean());
    }

    [WebJavaScriptFact]
    public void RoutedNetworkPreflightRestoresThePromptAndOffersTheSettingInline()
    {
        JsonElement result = Run("""
            R['/api/agent/settings'] = function (call) {
              return call.method === 'POST' ? call.body
                : { thinkByDefault: false, defaultSkills: [], skillsEnabled: true,
                    allowNetwork: false, maxTokens: 2048 };
            };
            R['/api/chat'] = { __status: 503,
              body: { code: 'network_disabled',
                      error: 'network access is disabled by the user for this research workflow' } };
            """, """
            var prompt = '搜索apple M6的信息，并对比M5芯片，然后生成pptx报告';
            __page.byId['text'].value = prompt;
            __page.byId['send'].dispatch('click');
            return settle(20).then(function () {
              var action = __page.byId['chat'].querySelector('.notice-action');
              var label = action && action.textContent;
              if (action) action.dispatch('click');
              return settle(10).then(function () {
                return {
                  prompt: __page.byId['text'].value,
                  history: window.TensorAgent.history(),
                  label: label,
                  saved: __page.requests('/api/agent/settings')
                    .filter(function (c) { return c.method === 'POST'; })
                    .map(function (c) { return c.body; })
                };
              });
            });
            """);

        Assert.Equal("搜索apple M6的信息，并对比M5芯片，然后生成pptx报告",
            result.GetProperty("prompt").GetString());
        Assert.Empty(result.GetProperty("history").EnumerateArray());
        Assert.Equal("Turn on Network", result.GetProperty("label").GetString());
        JsonElement saved = Assert.Single(result.GetProperty("saved").EnumerateArray());
        Assert.True(saved.GetProperty("allowNetwork").GetBoolean());
    }

    [WebJavaScriptFact]
    public void RoutedSetupPreflightRestoresThePromptAndAttachmentsAndOpensSettings()
    {
        JsonElement result = Run("""
            R['/api/chat'] = { __status: 503,
              body: { code: 'routed_workflow_unavailable',
                      error: 'This routed workflow cannot start: skills_run is unavailable.' } };
            """, """
            window.TensorAgent.addAttachment({ ok: true, file: 'a5.md', fileName: 'notes.md',
                                               mediaType: 'text', url: '/uploads/a5.md',
                                               textContent: 'existing notes' });
            var prompt = '搜索apple M6的信息，并对比M5芯片，然后生成pptx报告';
            __page.byId['text'].value = prompt;
            __page.byId['send'].dispatch('click');
            return settle(20).then(function () {
              var action = __page.byId['chat'].querySelector('.notice-action');
              var label = action && action.textContent;
              if (action) action.dispatch('click');
              return settle(10).then(function () {
                return {
                  prompt: __page.byId['text'].value,
                  attachments: window.TensorAgent.attachmentCount(),
                  history: window.TensorAgent.history(),
                  label: label,
                  routes: __page.requests('/api/agent/events').map(function (c) { return c.body; })
                };
              });
            });
            """);

        Assert.Equal("搜索apple M6的信息，并对比M5芯片，然后生成pptx报告",
            result.GetProperty("prompt").GetString());
        Assert.Equal(1, result.GetProperty("attachments").GetInt32());
        Assert.Empty(result.GetProperty("history").EnumerateArray());
        Assert.Equal("Open Settings", result.GetProperty("label").GetString());
        Assert.Contains(result.GetProperty("routes").EnumerateArray(), route =>
            route.TryGetProperty("type", out JsonElement type) && type.GetString() == "open-route"
            && route.TryGetProperty("route", out JsonElement name) && name.GetString() == "settings");
    }

    [WebJavaScriptTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void ReopenedChatPreservesWhetherAnEmptySkillSelectionWasExplicit(bool explicitSelection)
    {
        string explicitLiteral = explicitSelection ? "true" : "false";
        JsonElement result = Run($$"""
            R['/api/agent/conversations'] = {
              conversations: [{ id: 'saved', title: 'Old', updatedAt: '2026-09-01T10:00:00Z', messageCount: 1 }]
            };
            R['/api/sessions?conversation=saved'] = {
              sessionId: 's9', conversationId: 'saved', think: false, skills: [],
              skillsExplicit: {{explicitLiteral}},
              messages: [{ role: 'user', content: 'hello' }]
            };
            """, """
            __page.byId['text'].value = 'search Apple M6 and compare M5, then make a pptx';
            __page.byId['send'].dispatch('click');
            return { sent: __page.requests('/api/chat').map(function (c) { return c.body; }) };
            """);

        JsonElement request = Assert.Single(result.GetProperty("sent").EnumerateArray());
        if (explicitSelection)
        {
            Assert.True(request.TryGetProperty("skills", out JsonElement skills));
            Assert.Empty(skills.EnumerateArray());
        }
        else
        {
            Assert.False(request.TryGetProperty("skills", out _),
                "an untouched saved chat disabled host-side skill discovery");
        }
    }

    // =====================================================================================
    // the file the model made
    // =====================================================================================

    /// <summary>
    /// Tapping a generated file asks the APP to open it instead of navigating.
    ///
    /// <para>
    /// Navigating is what used to happen, and it is the reported bug: the route serves
    /// its files as attachments (program-written content must never render in the origin
    /// holding the launch token) and a WKWebView with no download delegate drops an
    /// attachment silently -- so what the user saw was the route's own 404 body,
    /// <c>{"error":"not found"}</c>. Both anchors are covered here, because they are
    /// built by different code: the file card this page renders, and the markdown link
    /// the model copies into its answer.
    /// </para>
    /// </summary>
    /// <summary>
    /// An answer cannot put script into the page. render() writes the model's text into
    /// innerHTML, and a link or an image puts part of it inside a double-quoted
    /// attribute; the escape used to leave quotes alone, so an answer containing
    /// <c>![x" onerror="...](y)</c> ran script in the page that holds the launch token
    /// (found 2026-09-30, confirmed in Chromium). The model repeats whatever a shared
    /// page or file says, so the text is not the user's.
    /// </summary>
    [WebJavaScriptFact]
    public void AnAnswerCannotPutScriptIntoThePage()
    {
        string answer = JsonSerializer.Serialize(
            "![x\" onerror=\"alert(1)](nope.png) [click](x\"onmouseover=\"alert(2)) [js](javascript:alert(3)) "
            + "![pixel](https://tracker.example/p.png) [docs](https://example.com/a?b=1&c=2) "
            + "![chart](/api/code/artifacts/run1/chart.png) [report](/api/code/artifacts/run1/report.pdf)");
        JsonElement result = Run($$"""
            R['/api/sessions?conversation=new'] = { sessionId: 's1', conversationId: 'c1', messages: [], think: false, skills: [] };
            R['/api/chat'] = { __sse: [{ token: {{answer}} }, { done: true, truncated: false }] };
            """, """
            __page.byId['text'].value = 'summarise this page';
            __page.byId['send'].dispatch('click');
            return settle(30).then(function () {
              return { html: __page.transcript().map(function (t) { return t.html; }).join('\n') };
            });
            """);

        string html = result.GetProperty("html").GetString()!;
        // Nothing the model wrote closed an attribute or chose a script URL...
        Assert.DoesNotContain("onerror=\"", html, StringComparison.Ordinal);
        Assert.DoesNotContain("onmouseover=\"", html, StringComparison.Ordinal);
        Assert.DoesNotContain("href=\"javascript:", html, StringComparison.OrdinalIgnoreCase);
        // ...and nothing from another origin is fetched as an image: it is a link instead.
        Assert.DoesNotContain("src=\"https://tracker.example", html, StringComparison.Ordinal);
        Assert.Contains("<a href=\"https://tracker.example/p.png\"", html, StringComparison.Ordinal);
        // What an answer legitimately links to still works.
        Assert.Contains("<a href=\"https://example.com/a?b=1&amp;c=2\"", html, StringComparison.Ordinal);
        Assert.Contains("<img alt=\"chart\" src=\"/api/code/artifacts/run1/chart.png\">", html, StringComparison.Ordinal);
        Assert.Contains("<a href=\"/api/code/artifacts/run1/report.pdf\"", html, StringComparison.Ordinal);
    }

    /// <summary>
    /// A table in an answer is drawn as one. Models answer comparisons with Markdown
    /// tables, and the page printed them as rows of pipes -- though its stylesheet had
    /// table rules waiting for them.
    /// </summary>
    [WebJavaScriptFact]
    public void AMarkdownTableInAnAnswerIsDrawnAsATable()
    {
        string answer = JsonSerializer.Serialize(
            "Each person pays:\n| Person | Pays |\n|---|---:|\n| Had drinks | **$70.39** |\n| No drinks | $42.07 |\nThat covers $224.91.");
        JsonElement result = Run($$"""
            R['/api/sessions?conversation=new'] = { sessionId: 's1', conversationId: 'c1', messages: [], think: false, skills: [] };
            R['/api/chat'] = { __sse: [{ token: {{answer}} }, { done: true, truncated: false }] };
            """, """
            __page.byId['text'].value = 'split the bill';
            __page.byId['send'].dispatch('click');
            return settle(30).then(function () {
              return { html: __page.transcript().map(function (t) { return t.html; }).join('\n') };
            });
            """);

        string html = result.GetProperty("html").GetString()!;
        Assert.Contains(
            "<p>Each person pays:</p><table><thead><tr><th>Person</th><th>Pays</th></tr></thead><tbody>"
            + "<tr><td>Had drinks</td><td><strong>$70.39</strong></td></tr>"
            + "<tr><td>No drinks</td><td>$42.07</td></tr></tbody></table><p>That covers $224.91.</p>",
            html, StringComparison.Ordinal);
        Assert.DoesNotContain("|---|", html, StringComparison.Ordinal);
    }

    [WebJavaScriptTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void NativeDictationCapabilityControlsTheComposerGesture(bool supported)
    {
        string payload = Convert.ToBase64String(Encoding.UTF8.GetBytes(JsonSerializer.Serialize(new
        {
            dictation = supported,
            composerHint = "Message… or press Windows+H to dictate",
        })));
        JsonElement result = Run("", $$"""
            var box = __page.byId['text'];
            var blurred = false;
            box.blur = function () { blurred = true; };
            box.value = 'Keep this draft';
            var ready = window.TensorAgent.__fromHost('nativeReady', '{{payload}}');
            box.dispatch('pointerdown', { clientX: 100, clientY: 100 });
            return wait(550).then(function () {
              box.dispatch('pointerup');
              var voice = document.body.classList.contains('voice');
              __page.byId['hold'].dispatch('pointerdown', { preventDefault: function () {} });
              window.TensorAgent.refreshModel();
              return settle(10).then(function () {
                return { ready: ready, supported: window.TensorAgent.canDictate(), voice: voice,
                  blurred: blurred, draft: box.value, placeholder: box.placeholder,
                  starts: __page.requests('/api/agent/events').filter(function (r) {
                    return r.body.type === 'dictate-start';
                  }).length };
              });
            });
            """);

        Assert.Equal("ok", result.GetProperty("ready").GetString());
        Assert.Equal(supported, result.GetProperty("supported").GetBoolean());
        Assert.Equal(supported, result.GetProperty("voice").GetBoolean());
        Assert.Equal(supported, result.GetProperty("blurred").GetBoolean());
        Assert.Equal(supported ? 1 : 0, result.GetProperty("starts").GetInt32());
        Assert.Equal("Keep this draft", result.GetProperty("draft").GetString());
        Assert.Equal(supported ? "Message… or hold to talk" : "Message… or press Windows+H to dictate",
            result.GetProperty("placeholder").GetString());
    }

    [WebJavaScriptFact]
    public void TappingAGeneratedFileAsksTheAppToOpenItRatherThanNavigating()
    {
        JsonElement result = Run("""
            R['/api/sessions?conversation=new'] = { sessionId: 's1', conversationId: 'c1', messages: [], think: false, skills: [] };
            R['/api/chat'] = { __sse: [
              { skill_step: 'shell', skill: 'documents', detail: 'make_pdf.py', ok: true,
                files: [{ name: 'photo.pdf', bytes: 2048, url: '/api/code/artifacts/run1/photo.pdf' }] },
              { tool_progress: 'finished', tool: 'shell', seconds: 3 },
              { token: 'Here is your PDF: [photo.pdf](/api/code/artifacts/run1/photo.pdf)' },
              { done: true, truncated: false }
            ] };
            """, """
            window.TensorAgent.nativeReady();
            __page.byId['text'].value = 'turn this photo into a pdf';
            __page.byId['send'].dispatch('click');
            return settle(30).then(function () {
              return { links: __page.transcript().reduce(function (all, t) { return all.concat(t.media); }, [])
                .filter(function (m) { return m.tag === 'A'; })
                .map(function (m) { return m.src; }) };
            });
            """);

        // Both links exist and both point at the route.
        string[] links = Strings(result, "links");
        Assert.Contains("/api/code/artifacts/run1/photo.pdf", links);
    }

    /// <summary>
    /// The click itself, on the file card the page rendered: claimed by the page and
    /// handed to the app, with nothing navigated.
    /// </summary>
    [WebJavaScriptFact]
    public void TheClickOnAGeneratedFileIsHandedToTheApp()
    {
        JsonElement result = Run("""
            R['/api/sessions?conversation=new'] = { sessionId: 's1', conversationId: 'c1', messages: [], think: false, skills: [] };
            R['/api/chat'] = { __sse: [
              { skill_step: 'shell', skill: 'documents', detail: 'make_pdf.py', ok: true,
                files: [{ name: 'photo.pdf', bytes: 2048, url: '/api/code/artifacts/run1/photo%20one.pdf' }] },
              { tool_progress: 'finished', tool: 'shell', seconds: 3 },
              { token: 'Done.' },
              { done: true, truncated: false }
            ] };
            """, """
            window.TensorAgent.nativeReady();
            __page.byId['text'].value = 'turn this photo into a pdf';
            __page.byId['send'].dispatch('click');
            return settle(30).then(function () {
              var anchor = null;
              (function walk(node) {
                node.children.forEach(function (c) {
                  if (c.tagName === 'A' && String(c.href || '').indexOf('/api/code/artifacts/') === 0) anchor = c;
                  walk(c);
                });
              })(__page.byId['chat']);
              var prevented = false;
              document.dispatch('click', { target: anchor, button: 0, defaultPrevented: false,
                                           preventDefault: function () { prevented = true; } });
              return { found: !!anchor, prevented: prevented,
                       asked: __page.requests('/api/agent/events').map(function (c) { return c.body; }) };
            });
            """);

        Assert.True(result.GetProperty("found").GetBoolean(), "the file card rendered no link to tap");
        Assert.True(result.GetProperty("prevented").GetBoolean(),
            "the tap navigated instead of being handed to the app");
        JsonElement[] asked = result.GetProperty("asked").EnumerateArray()
            .Where(e => e.TryGetProperty("type", out JsonElement t) && t.GetString() == "open-file")
            .ToArray();
        JsonElement open = Assert.Single(asked);
        // The URL is passed through as the page holds it -- percent-encoded -- because
        // the app decodes it one segment at a time against the artifact store.
        Assert.Equal("/api/code/artifacts/run1/photo%20one.pdf", open.GetProperty("url").GetString());
    }

    // =====================================================================================
    // the harness itself
    // =====================================================================================
    //
    // What follows tests PageDom.js, not the page: that the fake network can fail the
    // way WebKit's does, that a stream can die or go quiet, and that aborting is real.
    // A test of the page's behaviour under those conditions is only as good as the
    // model of the conditions, so the model is checked on its own first.

    [WebJavaScriptFact]
    public void HarnessARouteThatRejectsMakesFetchRejectWithATypeError()
    {
        JsonElement result = Run("""
            R['/api/harness/down'] = { __reject: 'Load failed' };
            R['/api/harness/default'] = function () { return { __reject: true }; };
            R['/api/harness/throws'] = function () { throw new Error('route exploded'); };
            var tries = 0;
            R['/api/harness/flaky'] = function () { return ++tries === 1 ? { __reject: 'Load failed' } : { ok: true, tries: tries }; };
            """, """
            function outcome(p) {
              return p.then(
                function (res) { return res.json().then(function (body) { return { resolved: true, body: body }; }); },
                function (e) { return { resolved: false, name: e && e.name, message: e && e.message, typeError: e instanceof TypeError }; });
            }
            var threwSynchronously = false, flakyPromise;
            try { flakyPromise = fetch('/api/harness/throws'); } catch (e) { threwSynchronously = true; }
            return Promise.all([
              outcome(fetch('/api/harness/down')),
              outcome(fetch('/api/harness/default')),
              outcome(flakyPromise || Promise.reject(new Error('never made'))),
              outcome(fetch('/api/harness/flaky')),
              outcome(fetch('/api/harness/flaky')),
            ]).then(function (all) {
              return { threwSynchronously: threwSynchronously, down: all[0], byDefault: all[1], throws: all[2],
                       flakyFirst: all[3], flakySecond: all[4],
                       recorded: __page.requests('/api/harness/flaky').length };
            });
            """);

        Assert.False(result.GetProperty("threwSynchronously").GetBoolean(), "a route that throws must reject fetch, not throw out of it");

        JsonElement down = result.GetProperty("down");
        Assert.False(down.GetProperty("resolved").GetBoolean());
        Assert.Equal("TypeError", down.GetProperty("name").GetString());
        Assert.Equal("Load failed", down.GetProperty("message").GetString());
        Assert.True(down.GetProperty("typeError").GetBoolean());

        JsonElement byDefault = result.GetProperty("byDefault");
        Assert.Equal("TypeError", byDefault.GetProperty("name").GetString());
        Assert.Equal("Load failed", byDefault.GetProperty("message").GetString());

        JsonElement throws = result.GetProperty("throws");
        Assert.False(throws.GetProperty("resolved").GetBoolean());
        Assert.Equal("route exploded", throws.GetProperty("message").GetString());

        // A route function decides per call, so "down once, then back" is one counter.
        Assert.False(result.GetProperty("flakyFirst").GetProperty("resolved").GetBoolean());
        Assert.True(result.GetProperty("flakySecond").GetProperty("resolved").GetBoolean());
        Assert.Equal(2, result.GetProperty("flakySecond").GetProperty("body").GetProperty("tries").GetInt32());
        Assert.Equal(2, result.GetProperty("recorded").GetInt32());
    }

    [WebJavaScriptFact]
    public void HarnessAHangingStreamReadRejectsWithAbortErrorWhenItsSignalFires()
    {
        JsonElement result = Run("""
            R['/api/harness/quiet'] = { __sse: [{ token: 'one' }], __then: 'hang' };
            """, """
            var ctrl = new AbortController();
            var fired = 0;
            ctrl.signal.addEventListener('abort', function () { fired++; });
            var abortedBefore = ctrl.signal.aborted;
            return fetch('/api/harness/quiet', { signal: ctrl.signal }).then(function (res) {
              var reader = res.body.getReader();
              return reader.read().then(function (first) {
                var settled = null;
                var second = reader.read().then(
                  function (r) { settled = { resolved: true, done: r.done }; },
                  function (e) { settled = { resolved: false, name: e && e.name }; });
                return wait(30).then(function () {
                  var stillPending = settled === null;
                  ctrl.abort();
                  return second.then(function () {
                    return reader.read().then(
                      function () { return { resolved: true }; },
                      function (e) { return { resolved: false, name: e && e.name }; });
                  }).then(function (third) {
                    return { abortedBefore: abortedBefore, abortedAfter: ctrl.signal.aborted, fired: fired,
                             first: first, stillPending: stillPending, second: settled, third: third };
                  });
                });
              });
            });
            """);

        Assert.False(result.GetProperty("abortedBefore").GetBoolean());
        Assert.True(result.GetProperty("abortedAfter").GetBoolean());
        Assert.Equal(1, result.GetProperty("fired").GetInt32());

        JsonElement first = result.GetProperty("first");
        Assert.False(first.GetProperty("done").GetBoolean());
        Assert.Equal("data: {\"token\":\"one\"}\n", first.GetProperty("value").GetString());

        Assert.True(result.GetProperty("stillPending").GetBoolean(), "a hanging read settled on its own");
        JsonElement second = result.GetProperty("second");
        Assert.False(second.GetProperty("resolved").GetBoolean());
        Assert.Equal("AbortError", second.GetProperty("name").GetString());
        // And the reader stays dead: every read after the abort is refused the same way.
        Assert.Equal("AbortError", result.GetProperty("third").GetProperty("name").GetString());
    }

    [WebJavaScriptFact]
    public void HarnessAStreamCanDropAfterItsFramesAndTakeRealTimePerRead()
    {
        JsonElement result = Run("""
            R['/api/harness/drops'] = { __sse: [{ token: 'a' }, { token: 'b' }], __then: 'reject', __delay: 20 };
            R['/api/harness/plain'] = { fine: true };
            """, """
            var began = Date.now(), frames = [];
            var dead = new AbortController(); dead.abort();
            var preAborted = fetch('/api/harness/plain', { signal: dead.signal }).then(
              function () { return { resolved: true }; },
              function (e) { return { resolved: false, name: e && e.name }; });
            return fetch('/api/harness/drops').then(function (res) {
              var reader = res.body.getReader();
              function pump() {
                return reader.read().then(function (r) {
                  if (r.done) return { how: 'ended' };
                  frames.push(r.value);
                  return pump();
                }, function (e) { return { how: 'rejected', name: e && e.name, message: e && e.message }; });
              }
              return pump();
            }).then(function (end) {
              return preAborted.then(function (pre) {
                var n = __page.element('div'); n.className = 'notice error'; n.textContent = 'It broke.';
                var b = __page.element('button'); b.className = 'notice-action'; b.textContent = 'Retry';
                n.appendChild(b);
                __page.byId['chat'].appendChild(n);
                var plain = __page.element('div'); plain.className = 'notice'; plain.textContent = 'Just so you know.';
                __page.byId['chat'].appendChild(plain);
                return { elapsed: Date.now() - began, frames: frames, end: end, preAborted: pre,
                         errorNotices: __page.errorNotices(), notices: __page.notices(),
                         recordedDead: __page.requests('/api/harness/plain').length };
              });
            });
            """);

        Assert.Equal(new[] { "data: {\"token\":\"a\"}\n", "data: {\"token\":\"b\"}\n" }, Strings(result, "frames"));
        JsonElement end = result.GetProperty("end");
        Assert.Equal("rejected", end.GetProperty("how").GetString());
        Assert.Equal("TypeError", end.GetProperty("name").GetString());
        Assert.Equal("Load failed", end.GetProperty("message").GetString());
        // Three reads (two frames and the drop) at 20 ms each: the delay is real time.
        Assert.True(result.GetProperty("elapsed").GetInt32() >= 55,
            "the delayed reads came back in " + result.GetProperty("elapsed").GetInt32() + " ms");

        JsonElement pre = result.GetProperty("preAborted");
        Assert.False(pre.GetProperty("resolved").GetBoolean());
        Assert.Equal("AbortError", pre.GetProperty("name").GetString());
        Assert.Equal(1, result.GetProperty("recordedDead").GetInt32());

        Assert.Equal(new[] { "It broke.Retry" }, Strings(result, "errorNotices"));
        Assert.Equal(new[] { "It broke.Retry", "Just so you know." }, Strings(result, "notices"));
    }

    [WebJavaScriptTheory]
    [InlineData("text")]
    [InlineData("imageUrl")]
    [InlineData("videoUrl")]
    public void ReopeningACompletedTranscriptDoesNotReplayItsFinishedTurn(string kind)
    {
        JsonElement result = Run($$"""
            var reply = { role: 'assistant', content: 'Finished answer.' };
            var frame = { token: 'Finished answer.' };
            if ('{{kind}}' !== 'text') {
              reply['{{kind}}'] = frame['{{kind}}'] = '/uploads/finished.{{(kind == "videoUrl" ? "mp4" : "png")}}';
            }
            R['/api/sessions?conversation=saved'] = { sessionId: 's2', conversationId: 'saved',
              messages: [{ role: 'user', content: 'Make it.' }, reply],
              activeTurn: { id: 'finished', running: false, state: 'Completed' } };
            R['/api/agent/turns'] = { turn: { id: 'finished', running: false, state: 'Completed' } };
            R['/api/agent/turns/finished'] = { __sse: [frame, { done: true }] };
            """, """
            window.TensorAgent.openConversation('saved');
            return settle(10).then(function () {
              window.TensorAgent.resumeTurn();
              return settle(30);
            }).then(function () {
              return { transcript: __page.transcript(), history: window.TensorAgent.history(),
                       attaches: __page.requests('/api/agent/turns/finished').length,
                       generating: window.TensorAgent.isGenerating() };
            });
            """);

        Assert.False(result.TryGetProperty("error", out _), result.ToString());
        Assert.Equal(2, result.GetProperty("history").GetArrayLength());
        Assert.Single(Bubbles(result.GetProperty("transcript")), t => t.GetProperty("role").GetString() == "assistant");
        Assert.Equal(0, result.GetProperty("attaches").GetInt32());
        Assert.False(result.GetProperty("generating").GetBoolean());
    }

    [WebJavaScriptFact]
    public void ASendLostBeforeTheTurnIdIsNeverAnsweredWithAnEarlierTurnsAnswer()
    {
        // The host keeps a finished turn for an hour, so "what is this conversation
        // generating" answers with the PREVIOUS question's turn long after it ended.
        // A page recovering a request whose turn it never learned must not take that
        // for its own, or the old answer appears under the new question and both are
        // saved.
        JsonElement result = Run($$"""
            var sends = 0;
            R['/api/chat'] = function () {
              sends++;
              return sends === 1
                ? { __status: 200, headers: { {{TurnHeader}} }, body: { __sse: [{ token: 'Paris.' }, { done: true, truncated: false }] } }
                : { __reject: 'Load failed' };
            };
            R['/api/agent/turns'] = { turn: { id: 't1', running: false } };
            R['/api/agent/turns/t1'] = { __sse: [{ token: 'Paris.' }, { done: true, truncated: false }] };
            """, """
            window.TensorAgent.__testing.resumeDelays([10]);
            __page.byId['text'].value = 'capital of France?';
            __page.byId['send'].dispatch('click');
            return wait(200).then(function () {
              __page.byId['text'].value = 'and of Spain?';
              __page.byId['send'].dispatch('click');
              return wait(400);
            }).then(function () {
              return {
                transcript: __page.transcript(),
                history: window.TensorAgent.history(),
                text: __page.byId['text'].value,
                generating: window.TensorAgent.isGenerating(),
                attaches: __page.requests('/api/agent/turns/t1').length,
              };
            });
            """);

        // The first question and its answer, and nothing else: the second question is
        // back in the composer because the host never took it.
        var turns = Bubbles(result.GetProperty("transcript"));
        Assert.Equal(2, turns.Count);
        Assert.Equal("user", turns[0].GetProperty("role").GetString());
        Assert.Contains("Paris.", turns[1].GetProperty("html").GetString());
        var history = result.GetProperty("history").EnumerateArray().ToList();
        Assert.Equal(2, history.Count);
        Assert.Equal("capital of France?", history[0].GetProperty("content").GetString());
        Assert.Equal("and of Spain?", result.GetProperty("text").GetString());
        Assert.False(result.GetProperty("generating").GetBoolean());
        // Never re-read: the page had already read t1 to its end.
        Assert.Equal(0, result.GetProperty("attaches").GetInt32());
    }

    [WebJavaScriptFact]
    public void StoppingDuringARecoveryDisarmsIt()
    {
        JsonElement result = Run($$"""
            R['/api/chat'] = { __reject: 'Load failed' };
            R['/api/agent/turns'] = { turn: null };
            R['/api/agent/turns/t9/stop'] = { stopped: true };
            """, """
            window.TensorAgent.__testing.resumeDelays([250]);
            __page.byId['text'].value = 'a question';
            __page.byId['send'].dispatch('click');
            return wait(60).then(function () {
              var pendingBefore = window.TensorAgent.__testing.recovery().pending;
              window.TensorAgent.stop();
              var pendingAfter = window.TensorAgent.__testing.recovery().pending;
              __page.byId['text'].value = 'something else entirely';
              return wait(500).then(function () {
                return {
                  pendingBefore: pendingBefore,
                  pendingAfter: pendingAfter,
                  text: __page.byId['text'].value,
                  generating: window.TensorAgent.isGenerating(),
                  lookups: __page.requests('/api/agent/turns').length,
                };
              });
            });
            """);

        Assert.True(result.GetProperty("pendingBefore").GetBoolean(), "the recovery was never armed");
        Assert.False(result.GetProperty("pendingAfter").GetBoolean(), "stopping left the recovery timer armed");
        // The composer is what the user typed after stopping, not that plus a message
        // the recovery decided to hand back minutes later.
        Assert.Equal("something else entirely", result.GetProperty("text").GetString());
        Assert.False(result.GetProperty("generating").GetBoolean());
    }

    [WebJavaScriptFact]
    public void AResumeDuringASendsPrefillLeavesTheRequestAlone()
    {
        // The host answers a chat request's headers only with its first frame, so for
        // the whole of a prefill there is nothing to deliver and nothing to trust by.
        // Aborting the request there loses the turn id AND the acknowledgement of any
        // shared draft it named, which makes every later send a 409.
        JsonElement result = Run($$"""
            var shareTaken = false;
            R['/api/agent/share/claim'] = function () {
              return shareTaken ? { share: null }
                : { share: { id: 'share-1', text: 'look at this', newChat: false, attachments: [] } };
            };
            R['/api/chat'] = function () {
              shareTaken = true;
              return { __status: 200, headers: { {{TurnHeader}} }, body: { __sse: [{ token: 'Well' }, { done: true, truncated: false }], __delay: 400 } };
            };
            R['/api/agent/turns'] = { turn: { id: 't1', running: true } };
            """, """
            window.TensorAgent.__fromHost('takeShare', 'e30=');
            return wait(120).then(function () {
              var shared = __page.byId['chips'].querySelectorAll('.shared').length;
              __page.byId['send'].dispatch('click');
              window.TensorAgent.resumeTurn();
              return wait(60).then(function () {
                document.visibilityState = 'hidden';
                document.dispatch('visibilitychange', {});
                document.visibilityState = 'visible';
                document.dispatch('visibilitychange', {});
                window.TensorAgent.resumeTurn();
                return wait(1800).then(function () {
                  return {
                    sharedBefore: shared,
                    sends: __page.requests('/api/chat').length,
                    sharedAfter: __page.byId['chips'].querySelectorAll('.shared').length,
                    lookups: __page.requests('/api/agent/turns').length,
                    transcript: __page.transcript(),
                    generating: window.TensorAgent.isGenerating(),
                    kinds: JSON.parse(window.TensorAgent.diagnostics()).events.map(function (e) { return e.k; }),
                  };
                });
              });
            });
            """);

        Assert.Equal(1, result.GetProperty("sharedBefore").GetInt32());
        Assert.Equal(1, result.GetProperty("sends").GetInt32());
        // The request was answered, so the share was consumed and its chip is gone.
        Assert.Equal(0, result.GetProperty("sharedAfter").GetInt32());
        Assert.Equal(0, result.GetProperty("lookups").GetInt32());
        var assistant = Bubbles(result.GetProperty("transcript")).Where(t => t.GetProperty("role").GetString() == "assistant").ToList();
        Assert.Single(assistant);
        Assert.Contains("Well", assistant[0].GetProperty("html").GetString());
        Assert.False(result.GetProperty("generating").GetBoolean());
        // The resume looked at the request, saw one that had not been answered yet, and
        // left it alone rather than superseding it.
        Assert.Contains("resume-skip", result.GetProperty("kinds").EnumerateArray().Select(k => k.GetString()));
    }

    [WebJavaScriptFact]
    public void AStreamThatGoesSilentIsReplacedByTheWatchdogWithoutAnyEvent()
    {
        // Nothing fires when a connection dies without a FIN: no error, no visibility
        // change, no host nudge. The host writes a keep-alive every five seconds, so
        // silence is the signal.
        JsonElement result = Run($$"""
            R['/api/chat'] = { __status: 200, headers: { {{TurnHeader}} }, body: { __sse: [{ token: 'Hel' }], __then: 'hang' } };
            R['/api/agent/turns'] = { turn: { id: 't1', running: true } };
            R['/api/agent/turns/t1'] = { __sse: [{ token: 'Hel' }, { token: 'lo' }, { done: true, truncated: false }] };
            """, """
            window.TensorAgent.__testing.streamTrustMs(50);
            window.TensorAgent.__testing.watchdogMs(60);
            __page.byId['text'].value = 'hi';
            __page.byId['send'].dispatch('click');
            return wait(700).then(function () {
              return {
                transcript: __page.transcript(),
                generating: window.TensorAgent.isGenerating(),
                attaches: __page.requests('/api/agent/turns/t1').length,
                kinds: JSON.parse(window.TensorAgent.diagnostics()).events.map(function (e) { return e.k; }),
              };
            });
            """);

        var assistant = Bubbles(result.GetProperty("transcript")).Where(t => t.GetProperty("role").GetString() == "assistant").ToList();
        Assert.Single(assistant);
        Assert.Contains("Hello", assistant[0].GetProperty("html").GetString());
        Assert.False(result.GetProperty("generating").GetBoolean());
        Assert.Equal(1, result.GetProperty("attaches").GetInt32());
        Assert.Contains("watchdog", result.GetProperty("kinds").EnumerateArray().Select(k => k.GetString()));
    }

    [WebJavaScriptFact]
    public void AnErrorInATurnIsSaidOnceHoweverOftenTheTurnIsReplayed()
    {
        JsonElement result = Run($$"""
            R['/api/chat'] = { __status: 200, headers: { {{TurnHeader}} }, body: { __sse: [{ error: 'The tool could not run.' }, { token: 'Sorry' }], __then: 'reject' } };
            R['/api/agent/turns'] = { turn: { id: 't1', running: true } };
            R['/api/agent/turns/t1'] = { __sse: [{ error: 'The tool could not run.' }, { token: 'Sorry' }, { done: true, truncated: false }] };
            """, """
            window.TensorAgent.__testing.resumeDelays([10]);
            __page.byId['text'].value = 'run it';
            __page.byId['send'].dispatch('click');
            return wait(400).then(function () {
              return { errors: __page.errorNotices(), generating: window.TensorAgent.isGenerating() };
            });
            """);

        var errors = result.GetProperty("errors").EnumerateArray().Select(e => e.GetString()).ToList();
        Assert.Single(errors, e => e!.Contains("The tool could not run.", StringComparison.Ordinal));
        Assert.False(result.GetProperty("generating").GetBoolean());
    }

    [WebJavaScriptFact]
    public void AChatThatCannotBeReopenedKeepsWhatIsOnTheScreen()
    {
        JsonElement result = Run("""
            var sessions = 0;
            R['/api/sessions?conversation=new'] = function () {
              sessions++;
              return sessions === 1
                ? { sessionId: 's1', conversationId: 'c1', messages: [], think: false, skills: [] }
                : { __reject: 'Load failed' };
            };
            R['/api/chat'] = { __sse: [{ token: 'An answer.' }, { done: true, truncated: false }] };
            """, """
            __page.byId['text'].value = 'a question';
            __page.byId['send'].dispatch('click');
            return wait(200).then(function () {
              return window.TensorAgent.openConversation(null);
            }).then(function () {
              return wait(400);
            }).then(function () {
              return { transcript: __page.transcript().map(function (t) { return t.html; }), errors: __page.errorNotices() };
            });
            """);

        string shown = string.Join(" | ", result.GetProperty("transcript").EnumerateArray().Select(t => t.GetString()));
        Assert.Contains("An answer.", shown, StringComparison.Ordinal);
        var reopenErrors = result.GetProperty("errors").EnumerateArray().Select(e => e.GetString()).ToList();
        Assert.Contains(reopenErrors, e => e!.Contains("Could not open that chat", StringComparison.Ordinal));
    }

    private static string[] Strings(JsonElement parent, string name) =>
        parent.TryGetProperty(name, out JsonElement list) && list.ValueKind == JsonValueKind.Array
            ? list.EnumerateArray().Select(e => e.GetString()!).ToArray()
            : Array.Empty<string>();

    // =====================================================================================
    // coming back to a stream the page could not read
    // =====================================================================================
    //
    // The report these guard: "switch to another app while the model is answering, come
    // back later, see 'Could not open the shared item: Load failed', and nothing works".
    // Behind it were four separate things the page did wrong once its stream was gone:
    // trusted a stream object that would never deliver again, retried a lookup once and
    // gave up, mistook a stream that ended without the host's `done` for a finished
    // answer, and let a POST that failed at the transport level surface as a share error.

    private const string TurnHeader = "'X-TensorAgent-Turn': 't1'";

    /// <summary>
    /// The transcript without the empty-state placeholder: the fake DOM registers ids
    /// lazily, so the page's <c>$('empty').remove()</c> removes a different element and
    /// the placeholder stays in the chat's children for the life of the test.
    /// </summary>
    private static List<JsonElement> Bubbles(JsonElement transcript) => transcript.EnumerateArray()
        .Where(t => !(t.GetProperty("html").GetString() ?? string.Empty).StartsWith("<h1>TensorAgent</h1>", StringComparison.Ordinal))
        .ToList();

    [WebJavaScriptFact]
    public void AStaleStreamIsReplacedWhenThePageBecomesVisibleAgain()
    {
        JsonElement result = Run($$"""
            R['/api/chat'] = { __status: 200, headers: { {{TurnHeader}} }, body: { __sse: [{ token: 'Hel' }], __then: 'hang' } };
            R['/api/agent/turns'] = { turn: { id: 't1', running: true } };
            R['/api/agent/turns/t1'] = { __sse: [{ token: 'Hel' }, { token: 'lo' }, { done: true, truncated: false }] };
            """, """
            window.TensorAgent.__testing.streamTrustMs(0);
            __page.byId['text'].value = 'hi';
            __page.byId['send'].dispatch('click');
            return settle(20).then(function () {
              var during = { generating: window.TensorAgent.isGenerating(), lookups: __page.requests('/api/agent/turns').length };
              document.visibilityState = 'hidden';
              document.dispatch('visibilitychange', {});
              document.visibilityState = 'visible';
              document.dispatch('visibilitychange', {});
              return wait(500).then(function () {
                return {
                  during: during,
                  transcript: __page.transcript(),
                  history: window.TensorAgent.history(),
                  generating: window.TensorAgent.isGenerating(),
                  attaches: __page.requests('/api/agent/turns/t1').length,
                  errors: __page.errorNotices(),
                  diag: JSON.parse(window.TensorAgent.diagnostics()),
                };
              });
            });
            """);

        Assert.True(result.GetProperty("during").GetProperty("generating").GetBoolean(), "the answer was streaming before the page was hidden");
        Assert.Equal(0, result.GetProperty("during").GetProperty("lookups").GetInt32());
        var assistant = Bubbles(result.GetProperty("transcript")).Where(t => t.GetProperty("role").GetString() == "assistant").ToList();
        Assert.Single(assistant);
        Assert.Contains("Hello", assistant[0].GetProperty("html").GetString());
        var history = result.GetProperty("history").EnumerateArray().ToList();
        Assert.Equal(2, history.Count);
        Assert.Equal("Hello", history[1].GetProperty("content").GetString());
        Assert.False(result.GetProperty("generating").GetBoolean(), "the page still said it was working after the replayed answer finished");
        Assert.Equal(1, result.GetProperty("attaches").GetInt32());
        Assert.Empty(result.GetProperty("errors").EnumerateArray());
        var kinds = result.GetProperty("diag").GetProperty("events").EnumerateArray().Select(e => e.GetProperty("k").GetString()).ToList();
        Assert.Contains("visibility", kinds);
        Assert.Contains("supersede", kinds);
        Assert.Contains("attach", kinds);
        Assert.Contains("finish", kinds);
        Assert.False(result.GetProperty("diag").GetProperty("attached").GetBoolean());
    }

    [WebJavaScriptFact]
    public void AStreamThatIsStillDeliveringIsLeftAloneByANudge()
    {
        JsonElement result = Run($$"""
            R['/api/chat'] = { __status: 200, headers: { {{TurnHeader}} }, body: { __sse: [{ token: 'Hel' }], __then: 'hang' } };
            R['/api/agent/turns'] = { turn: { id: 't1', running: true } };
            """, """
            __page.byId['text'].value = 'hi';
            __page.byId['send'].dispatch('click');
            return settle(20).then(function () {
              var nudged = window.TensorAgent.resumeTurn();
              return settle(20).then(function () {
                return {
                  nudged: nudged,
                  lookups: __page.requests('/api/agent/turns').length,
                  generating: window.TensorAgent.isGenerating(),
                  delivering: JSON.parse(window.TensorAgent.diagnostics()).delivering,
                };
              });
            });
            """);

        Assert.True(result.GetProperty("nudged").GetBoolean());
        Assert.Equal(0, result.GetProperty("lookups").GetInt32());
        Assert.True(result.GetProperty("generating").GetBoolean());
        Assert.True(result.GetProperty("delivering").GetBoolean());
    }

    [WebJavaScriptFact]
    public void AStreamThatRejectsMidAnswerIsTakenUpAgainWithBackoffNotOnce()
    {
        JsonElement result = Run($$"""
            var lookups = 0;
            R['/api/chat'] = { __status: 200, headers: { {{TurnHeader}} }, body: { __sse: [{ token: 'Hel' }], __then: 'reject' } };
            R['/api/agent/turns'] = function () {
              lookups++;
              return lookups < 3 ? { __reject: 'Load failed' } : { turn: { id: 't1', running: true } };
            };
            R['/api/agent/turns/t1'] = { __sse: [{ token: 'Hel' }, { token: 'lo' }, { done: true, truncated: false }] };
            """, """
            window.TensorAgent.__testing.resumeDelays([10, 20, 30, 40, 50]);
            __page.byId['text'].value = 'hi';
            __page.byId['send'].dispatch('click');
            return wait(600).then(function () {
              return {
                lookups: __page.requests('/api/agent/turns').length,
                transcript: __page.transcript(),
                history: window.TensorAgent.history(),
                generating: window.TensorAgent.isGenerating(),
                errors: __page.errorNotices(),
                kinds: JSON.parse(window.TensorAgent.diagnostics()).events.map(function (e) { return e.k; }),
              };
            });
            """);

        Assert.True(result.GetProperty("lookups").GetInt32() >= 3, "the page gave up before the third lookup");
        var assistant = Bubbles(result.GetProperty("transcript")).Where(t => t.GetProperty("role").GetString() == "assistant").ToList();
        Assert.Single(assistant);
        Assert.Contains("Hello", assistant[0].GetProperty("html").GetString());
        Assert.Equal(2, result.GetProperty("history").GetArrayLength());
        Assert.False(result.GetProperty("generating").GetBoolean());
        Assert.Empty(result.GetProperty("errors").EnumerateArray());
        var kinds = result.GetProperty("kinds").EnumerateArray().Select(k => k.GetString()).ToList();
        Assert.Contains("stream-failed", kinds);
        Assert.Contains("resume-retry", kinds);
        Assert.Contains("resume-lookup-failed", kinds);
    }

    [WebJavaScriptFact]
    public void AStreamThatEndsWithoutADoneFrameIsTakenUpAgainWhenTheTurnIsStillRunning()
    {
        JsonElement result = Run($$"""
            R['/api/chat'] = { __status: 200, headers: { {{TurnHeader}} }, body: { __sse: [{ token: 'Hel' }] } };
            R['/api/agent/turns'] = { turn: { id: 't1', running: true } };
            R['/api/agent/turns/t1'] = { __sse: [{ token: 'Hel' }, { token: 'lo' }, { done: true, truncated: false }] };
            """, """
            window.TensorAgent.__testing.resumeDelays([10]);
            __page.byId['text'].value = 'hi';
            __page.byId['send'].dispatch('click');
            return wait(300).then(function () {
              return {
                transcript: __page.transcript(),
                history: window.TensorAgent.history(),
                generating: window.TensorAgent.isGenerating(),
                attaches: __page.requests('/api/agent/turns/t1').length,
                kinds: JSON.parse(window.TensorAgent.diagnostics()).events.map(function (e) { return e.k; }),
              };
            });
            """);

        var assistant = Bubbles(result.GetProperty("transcript")).Where(t => t.GetProperty("role").GetString() == "assistant").ToList();
        Assert.Single(assistant);
        Assert.Contains("Hello", assistant[0].GetProperty("html").GetString());
        // ONE assistant entry, carrying the whole answer -- not a fragment and then the whole.
        var history = result.GetProperty("history").EnumerateArray().ToList();
        Assert.Equal(2, history.Count);
        Assert.Equal("Hello", history[1].GetProperty("content").GetString());
        Assert.False(result.GetProperty("generating").GetBoolean());
        Assert.Equal(1, result.GetProperty("attaches").GetInt32());
        Assert.Contains("eof-without-done", result.GetProperty("kinds").EnumerateArray().Select(k => k.GetString()));
    }

    [WebJavaScriptFact]
    public void AStreamThatEndsWithoutADoneFrameIsReadAgainWhenTheTurnFinishedMeanwhile()
    {
        // The connection died as the answer was ending. The host still has the turn --
        // finished turns are kept for an hour -- so its replay, which ends with the
        // frame that says the turn is over, is the whole answer; the fragment this
        // reader was left holding is not.
        JsonElement result = Run($$"""
            R['/api/chat'] = { __status: 200, headers: { {{TurnHeader}} }, body: { __sse: [{ token: 'Hel' }] } };
            R['/api/agent/turns'] = { turn: { id: 't1', running: false } };
            R['/api/agent/turns/t1'] = { __sse: [{ token: 'Hel' }, { token: 'lo there' }, { done: true, truncated: false }] };
            """, """
            __page.byId['text'].value = 'hi';
            __page.byId['send'].dispatch('click');
            return wait(300).then(function () {
              return {
                transcript: __page.transcript(),
                history: window.TensorAgent.history(),
                generating: window.TensorAgent.isGenerating(),
                attaches: __page.requests('/api/agent/turns/t1').length,
              };
            });
            """);

        var assistant = Bubbles(result.GetProperty("transcript")).Where(t => t.GetProperty("role").GetString() == "assistant").ToList();
        Assert.Single(assistant);
        Assert.Contains("Hello there", assistant[0].GetProperty("html").GetString());
        var history = result.GetProperty("history").EnumerateArray().ToList();
        Assert.Equal(2, history.Count);
        Assert.Equal("Hello there", history[1].GetProperty("content").GetString());
        Assert.False(result.GetProperty("generating").GetBoolean());
        Assert.Equal(1, result.GetProperty("attaches").GetInt32());
    }

    [WebJavaScriptFact]
    public void AStreamThatEndsWithoutADoneFrameFinishesWithWhatItHasWhenTheTurnIsGone()
    {
        JsonElement result = Run($$"""
            R['/api/chat'] = { __status: 200, headers: { {{TurnHeader}} }, body: { __sse: [{ token: 'Hel' }] } };
            R['/api/agent/turns'] = { turn: null };
            """, """
            __page.byId['text'].value = 'hi';
            __page.byId['send'].dispatch('click');
            return wait(300).then(function () {
              return {
                transcript: __page.transcript(),
                history: window.TensorAgent.history(),
                generating: window.TensorAgent.isGenerating(),
                attaches: __page.requests('/api/agent/turns/t1').length,
              };
            });
            """);

        var assistant = Bubbles(result.GetProperty("transcript")).Where(t => t.GetProperty("role").GetString() == "assistant").ToList();
        Assert.Single(assistant);
        Assert.Contains("Hel", assistant[0].GetProperty("html").GetString());
        var history = result.GetProperty("history").EnumerateArray().ToList();
        Assert.Equal(2, history.Count);
        Assert.Equal("Hel", history[1].GetProperty("content").GetString());
        Assert.False(result.GetProperty("generating").GetBoolean());
        Assert.Equal(0, result.GetProperty("attaches").GetInt32());
    }

    [WebJavaScriptFact]
    public void ASendThatLosesItsStreamBeforeTheTurnIdAttachesToTheTurnTheHostStarted()
    {
        JsonElement result = Run("""
            R['/api/chat'] = { __reject: 'Load failed' };
            R['/api/agent/turns'] = { turn: { id: 't1', running: true } };
            R['/api/agent/turns/t1'] = { __sse: [{ token: 'Hel' }, { token: 'lo' }, { done: true, truncated: false }] };
            """, """
            window.TensorAgent.__testing.resumeDelays([10]);
            __page.byId['text'].value = 'hi';
            __page.byId['send'].dispatch('click');
            return wait(300).then(function () {
              return {
                transcript: __page.transcript(),
                history: window.TensorAgent.history(),
                generating: window.TensorAgent.isGenerating(),
                errors: __page.errorNotices(),
                text: __page.byId['text'].value,
              };
            });
            """);

        var turns = Bubbles(result.GetProperty("transcript"));
        Assert.Equal(2, turns.Count);
        Assert.Equal("user", turns[0].GetProperty("role").GetString());
        Assert.Contains("Hello", turns[1].GetProperty("html").GetString());
        Assert.Equal(2, result.GetProperty("history").GetArrayLength());
        Assert.False(result.GetProperty("generating").GetBoolean());
        Assert.Empty(result.GetProperty("errors").EnumerateArray());
        Assert.Equal("", result.GetProperty("text").GetString());
    }

    [WebJavaScriptFact]
    public void ASendThatNeverReachedTheHostGoesBackIntoTheComposer()
    {
        JsonElement result = Run("""
            R['/api/chat'] = { __reject: 'Load failed' };
            R['/api/agent/turns'] = { turn: null };
            """, """
            window.TensorAgent.__testing.resumeDelays([10]);
            window.TensorAgent.addAttachment({ ok: true, file: 'a1.png', fileName: 'cat.png', mediaType: 'image', url: '/uploads/a1.png' });
            __page.byId['text'].value = 'what is this';
            __page.byId['send'].dispatch('click');
            return wait(300).then(function () {
              return {
                transcript: __page.transcript(),
                history: window.TensorAgent.history(),
                generating: window.TensorAgent.isGenerating(),
                text: __page.byId['text'].value,
                attachments: window.TensorAgent.attachmentCount(),
              };
            });
            """);

        Assert.Empty(Bubbles(result.GetProperty("transcript")));
        Assert.Empty(result.GetProperty("history").EnumerateArray());
        Assert.False(result.GetProperty("generating").GetBoolean());
        Assert.Equal("what is this", result.GetProperty("text").GetString());
        Assert.Equal(1, result.GetProperty("attachments").GetInt32());
    }

    [WebJavaScriptFact]
    public void WhenTheHostCannotBeReachedForLongEnoughThePageHandsTheScreenBackAndSaysSo()
    {
        JsonElement result = Run($$"""
            R['/api/chat'] = { __status: 200, headers: { {{TurnHeader}} }, body: { __sse: [{ token: 'Hel' }], __then: 'reject' } };
            R['/api/agent/turns'] = { __reject: 'Load failed' };
            """, """
            window.TensorAgent.__testing.resumeDelays([5, 5]);
            __page.byId['text'].value = 'hi';
            __page.byId['send'].dispatch('click');
            return wait(300).then(function () {
              return {
                transcript: __page.transcript(),
                history: window.TensorAgent.history(),
                generating: window.TensorAgent.isGenerating(),
                errors: __page.errorNotices(),
                lookups: __page.requests('/api/agent/turns').length,
              };
            });
            """);

        Assert.False(result.GetProperty("generating").GetBoolean());
        Assert.Equal(2, result.GetProperty("lookups").GetInt32());
        var history = result.GetProperty("history").EnumerateArray().ToList();
        Assert.Equal(2, history.Count);
        Assert.Equal("Hel", history[1].GetProperty("content").GetString());
        var errors = result.GetProperty("errors").EnumerateArray().Select(e => e.GetString()).ToList();
        Assert.Contains(errors, e => e!.Contains("Lost the connection", StringComparison.Ordinal));
        Assert.DoesNotContain(errors, e => e!.Contains("Load failed", StringComparison.Ordinal));
    }

    [WebJavaScriptFact]
    public void AShareClaimThatFailsAtTheTransportIsRetriedOnceAndNotReported()
    {
        // Install the failure after boot. On Windows, forty 1 ms timer ticks
        // can outlast the retry delay, so boot's automatic claim may already
        // have consumed a failure installed in the initial route table.
        JsonElement result = Run("R['/api/agent/share/claim'] = { share: null };", """
            var claims = 0;
            R['/api/agent/share/claim'] = function () {
              claims++;
              return claims === 1 ? { __reject: 'Load failed' } : { share: null };
            };
            var claimsBefore = __page.requests('/api/agent/share/claim').length;
            window.TensorAgent.__fromHost('takeShare', 'e30=');
            return wait(500).then(function () {
              return {
                claims: __page.requests('/api/agent/share/claim').length - claimsBefore,
                errors: __page.errorNotices(),
              };
            });
            """);

        Assert.Equal(2, result.GetProperty("claims").GetInt32());
        Assert.Empty(result.GetProperty("errors").EnumerateArray());
    }

    [WebJavaScriptFact]
    public void AShareClaimTheHostRefusesIsReportedWithoutARetry()
    {
        JsonElement result = Run("""
            R['/api/agent/share/claim'] = { __status: 500, body: { error: 'broken' } };
            """, """
            var claimsBefore = __page.requests('/api/agent/share/claim').length;
            window.TensorAgent.__fromHost('takeShare', 'e30=');
            return wait(500).then(function () {
              return {
                claims: __page.requests('/api/agent/share/claim').length - claimsBefore,
                errors: __page.errorNotices(),
              };
            });
            """);

        Assert.Equal(1, result.GetProperty("claims").GetInt32());
        var errors = result.GetProperty("errors").EnumerateArray().Select(e => e.GetString()).ToList();
        Assert.Contains(errors, e => e!.Contains("Could not open the shared item", StringComparison.Ordinal) && e.Contains("HTTP 500", StringComparison.Ordinal));
    }

    [WebJavaScriptFact]
    public void ALongAnswerOfThousandsOfFramesRendersWholeAndOnce()
    {
        // A page re-attaching to a long answer is handed every frame so far at once;
        // the answer is painted per chunk rather than per frame, and must still come
        // out complete, in one bubble.
        JsonElement result = Run("""
            var frames = [];
            for (var i = 0; i < 2000; i++) frames.push({ token: 'w' + i + ' ' });
            frames.push({ done: true, truncated: false });
            R['/api/chat'] = { __sse: frames };
            """, """
            __page.byId['text'].value = 'hi';
            __page.byId['send'].dispatch('click');
            return settle(20).then(function () {
              var assistant = __page.transcript().filter(function (t) { return t.role === 'assistant' && t.html.indexOf('<h1>TensorAgent</h1>') !== 0; });
              return { bubbles: assistant.length, words: (assistant[0] && assistant[0].html || '').split('w').length - 1, generating: window.TensorAgent.isGenerating() };
            });
            """);

        Assert.Equal(1, result.GetProperty("bubbles").GetInt32());
        Assert.Equal(2000, result.GetProperty("words").GetInt32());
        Assert.False(result.GetProperty("generating").GetBoolean());
    }

    // ---- an image model (ImageTurns on the host) ----------------------------------------

    private const string ImageModel = """
        R['/api/models'] = { loaded: 'qwen_image_2.1_Q4_K_M.gguf', architecture: 'qwen_image',
                             loadedBackend: 'ggml_metal', visionReady: false };
        """;

    /// <summary>
    /// A picture denoises over tens of steps, and the host sends a small preview with
    /// some of them. They are one picture getting sharper, so the page refreshes one
    /// image in place and the finished picture replaces the last preview: the turn ends
    /// with exactly one image in the bubble, and the history keeps the finished one.
    /// </summary>
    [WebJavaScriptFact]
    public void AnImageModelsTurnIsOnePictureThatThePreviewsBecome()
    {
        JsonElement result = Run(ImageModel + """
            R['/api/chat'] = { __sse: [
              { image_step: 1, image_steps: 3, preview: null },
              { image_step: 2, image_steps: 3, preview: 'data:image/png;base64,AAAA' },
              { image_step: 3, image_steps: 3, preview: 'data:image/png;base64,BBBB' },
              { imageUrl: '/uploads/lighthouse.png', width: 1024, height: 1024 },
              { done: true, sessionId: 's1', tokenCount: 0, elapsed: 3, tokPerSec: 0, truncated: false }
            ] };
            """, """
            var placeholder = __page.byId['text'].placeholder;
            __page.byId['text'].value = 'a lighthouse at dusk';
            __page.byId['send'].dispatch('click');
            return settle(30).then(function () {
              var turns = __page.transcript();
              var answer = turns[turns.length - 1];
              return {
                placeholder: placeholder,
                sent: __page.requests('/api/chat').map(function (c) { return c.body; }),
                role: answer.role,
                images: answer.media.filter(function (m) { return m.tag === 'IMG'; }).map(function (m) { return m.src; }),
                history: window.TensorAgent.history(),
                generating: window.TensorAgent.isGenerating(),
                errors: __page.errorNotices()
              };
            });
            """);

        Assert.StartsWith("Describe a picture", result.GetProperty("placeholder").GetString(), StringComparison.Ordinal);
        JsonElement sent = Assert.Single(result.GetProperty("sent").EnumerateArray());
        Assert.Equal("a lighthouse at dusk", sent.GetProperty("messages")[0].GetProperty("content").GetString());
        Assert.Equal("assistant", result.GetProperty("role").GetString());
        Assert.Equal(new[] { "/uploads/lighthouse.png" }, Strings(result, "images"));
        JsonElement history = result.GetProperty("history");
        JsonElement made = history[history.GetArrayLength() - 1];
        Assert.Equal("assistant", made.GetProperty("role").GetString());
        Assert.Equal("/uploads/lighthouse.png", made.GetProperty("imageUrl").GetString());
        Assert.False(result.GetProperty("generating").GetBoolean());
        Assert.Empty(Strings(result, "errors"));
    }

    /// <summary>
    /// With LoRA plug-ins on, the host names them on every step, and the progress says
    /// what the picture is drawn with; without them it reads as it always did.
    /// </summary>
    [Fact]
    public void AnImageModelsProgressNamesTheLoraPlugInsItDrawsWith()
    {
        JsonElement result = Run(ImageModel + """
            R['/api/chat'] = { __sse: [
              { image_step: 1, image_steps: 6, preview: null, image_loras: ['Viggle Turbo', 'Film Stills'] },
              { image_step: 6, image_steps: 6, preview: null, image_loras: ['Viggle Turbo', 'Film Stills'] },
              { imageUrl: '/uploads/cafe.png', width: 1024, height: 1024 },
              { done: true, sessionId: 's1', truncated: false }
            ] };
            """, """
            __page.byId['text'].value = 'a cafe at night';
            __page.byId['send'].dispatch('click');
            return settle(30).then(function () { return { progress: __page.progress() }; });
            """);

        Assert.Contains("Drawing with Viggle Turbo + Film Stills… step 1 of 6", Strings(result, "progress"));
    }

    /// <summary>
    /// The LoRA sheet of an image model: every plug-in with what can be done to it, and a
    /// speed plug-in turned on replacing the one that was on, because two step schedules
    /// cannot both apply. What the page sends is the whole choice, in order.
    /// </summary>
    [Fact]
    public void TheLoraSheetTurnsOnOneSpeedPlugInAtATime()
    {
        JsonElement result = Run(ImageModel + """
            var lorasState = { loadedModel: 'qwen-image-2.1-q4km', loadedModelName: 'Qwen-Image 2.1', supportsLoras: true, minStrength: 0.1, maxStrength: 1.5,
              chosen: [{ id: 'pruna', strength: 1 }, { id: 'film', strength: 0.6 }], loras: [
              { id: 'viggle', name: 'Viggle Turbo', kind: 'Speed', steps: 6, purpose: 'Fast.', license: 'QRL', totalBytes: 679604800,
                state: 'Installed', defaultStrength: 1, strengthAdjustable: false, chosen: false, strength: 1, download: null },
              { id: 'pruna', name: 'Pruna 8-Step', kind: 'Speed', steps: 8, purpose: 'Fast.', license: 'QRL', totalBytes: 335606104,
                state: 'Installed', defaultStrength: 1, strengthAdjustable: false, chosen: true, strength: 1, download: null },
              { id: 'film', name: 'Film Stills', kind: 'Style', purpose: 'Film.', license: 'QRL', totalBytes: 79743888,
                state: 'Installed', defaultStrength: 0.7, strengthAdjustable: true, chosen: true, strength: 0.6, download: null },
              { id: 'grain', name: 'Grainscape', kind: 'Style', purpose: 'Grain.', license: 'QRL', totalBytes: 79743888,
                state: 'NotInstalled', defaultStrength: 0.7, strengthAdjustable: true, chosen: false, strength: 0.7, download: null } ] };
            R['/api/agent/loras'] = lorasState;
            R['/api/agent/loras/choice'] = function () { return { __status: 400, body: { error: 'Viggle Turbo is not downloaded yet.' } }; };
            """, """
            __page.byId['model'].dispatch('click');
            return settle(20).then(function () {
              var shown = __page.byId['open-loras'].style.display !== 'none';
              __page.byId['open-loras'].dispatch('click');
              return settle(20).then(function () {
                var rows = __page.byId['lora-list'].querySelectorAll('.lorarow');
                function row(id) { return rows.filter(function (r) { return r.getAttribute('data-lora') === id; })[0]; }
                function actions(r) { return r.querySelectorAll('BUTTON').map(function (b) { return b.textContent; }); }
                var box = row('viggle').querySelectorAll('INPUT')[0];
                box.checked = true;
                box.dispatch('change');
                return settle(20).then(function () {
                  return {
                    shown: shown,
                    hint: __page.byId['loras-hint'].textContent,
                    rows: rows.length,
                    installedActions: actions(row('viggle')),
                    missingActions: actions(row('grain')),
                    sliders: row('film').querySelectorAll('INPUT').length,
                    sent: __page.requests('/api/agent/loras/choice').map(function (c) { return c.body; }),
                    sheetError: __page.byId['lora-error'].textContent,
                    sheetErrorShown: __page.byId['lora-error'].style.display !== 'none',
                    errors: __page.errorNotices()
                  };
                });
              });
            });
            """);

        Assert.True(result.GetProperty("shown").GetBoolean());
        Assert.Equal("Pruna 8-Step + Film Stills", result.GetProperty("hint").GetString());
        Assert.Equal(4, result.GetProperty("rows").GetInt32());
        Assert.Equal(new[] { "Remove" }, Strings(result, "installedActions"));
        Assert.Equal(new[] { "Download" }, Strings(result, "missingActions"));
        // The switch and the strength slider of a style that is on.
        Assert.Equal(2, result.GetProperty("sliders").GetInt32());

        JsonElement sent = Assert.Single(result.GetProperty("sent").EnumerateArray());
        JsonElement[] loras = sent.GetProperty("loras").EnumerateArray().ToArray();
        Assert.Equal(new[] { "film", "viggle" }, loras.Select(l => l.GetProperty("id").GetString()));
        Assert.Equal(0.6, loras[0].GetProperty("strength").GetDouble(), 3);
        // A refusal is the host's own words, not a status code, said in the sheet beside the
        // switch that flipped back rather than in the chat underneath it.
        Assert.Equal("Viggle Turbo is not downloaded yet.", result.GetProperty("sheetError").GetString());
        Assert.True(result.GetProperty("sheetErrorShown").GetBoolean());
        Assert.DoesNotContain("Viggle Turbo is not downloaded yet.", Strings(result, "errors"));
    }

    /// <summary>A sheet of two installed styles, neither on, whose choice route answers with what it was sent.</summary>
    private const string TwoStyles = """
        function loraRows(chosen, extra) {
          function on(id) { return chosen.some(function (c) { return c.id === id; }); }
          return { loadedModel: 'qwen-image-2.1-q4km', loadedModelName: 'Qwen-Image 2.1', supportsLoras: true, minStrength: 0.1, maxStrength: 1.5, chosen: chosen, loras: [
            { id: 'film', name: 'Film Stills', kind: 'Style', purpose: 'Film.', license: 'QRL', totalBytes: 79743888,
              state: 'Installed', installedBytes: 79743888, defaultStrength: 0.7, strengthAdjustable: true, chosen: on('film'), strength: 0.7, download: null },
            { id: 'grain', name: 'Grainscape', kind: 'Style', purpose: 'Grain.', license: 'QRL', totalBytes: 79743888,
              state: 'Installed', installedBytes: 79743888, defaultStrength: 0.7, strengthAdjustable: true, chosen: on('grain'), strength: 0.7, download: null }
          ].concat(extra || []) };
        }
        R['/api/agent/loras'] = loraRows([]);
        R['/api/agent/loras/choice'] = function (call) { return loraRows(call.body.loras); };
        function openLoras() {
          __page.byId['model'].dispatch('click');
          return settle(20).then(function () {
            __page.byId['open-loras'].dispatch('click');
            return settle(20);
          });
        }
        function loraRow(id) {
          return __page.byId['lora-list'].querySelectorAll('.lorarow').filter(function (r) { return r.getAttribute('data-lora') === id; })[0];
        }
        """;

    /// <summary>
    /// Two switches flipped faster than the host answers: the second change builds on the
    /// first rather than on the list painted before it, so both plug-ins end up on. Each
    /// save replaces the whole choice, so a second built on the old list would undo the first.
    /// </summary>
    [Fact]
    public void TwoQuickChangesInTheLoraSheetBuildOnEachOther()
    {
        JsonElement result = Run(ImageModel + TwoStyles, """
            return openLoras().then(function () {
              var film = loraRow('film').querySelectorAll('INPUT')[0];
              var grain = loraRow('grain').querySelectorAll('INPUT')[0];
              film.checked = true;
              film.dispatch('change');
              grain.checked = true;
              grain.dispatch('change');
              return settle(30).then(function () {
                return {
                  sent: __page.requests('/api/agent/loras/choice').map(function (c) { return c.body.loras.map(function (l) { return l.id; }).join('+'); }),
                  names: __page.byId['lora-list'].querySelectorAll('.nm').map(function (n) { return n.textContent; })
                };
              });
            });
            """);

        Assert.Equal(new[] { "film", "film+grain" }, Strings(result, "sent"));
        Assert.Equal(new[] { "● Film Stills", "● Grainscape" }, Strings(result, "names"));
    }

    /// <summary>
    /// A plug-in that is on but whose files have gone keeps a switch, so it can be turned off
    /// as the refused picture tells the user to; a stopped download's part files can be
    /// removed without finishing it; and a removal the host fails is said in the sheet,
    /// not painted as an empty list and announced as done.
    /// </summary>
    [Fact]
    public void APlugInWhoseFilesAreGoneCanStillBeTurnedOffOrRemoved()
    {
        JsonElement result = Run(ImageModel + TwoStyles + """
            var gone = { id: 'viggle', name: 'Viggle Turbo', kind: 'Speed', steps: 6, purpose: 'Fast.', license: 'QRL', totalBytes: 679604800,
              state: 'NotInstalled', installedBytes: 0, defaultStrength: 1, strengthAdjustable: false, chosen: true, strength: 1, download: null };
            var stopped = { id: 'pruna', name: 'Pruna 8-Step', kind: 'Speed', steps: 8, purpose: 'Fast.', license: 'QRL', totalBytes: 335606104,
              state: 'Partial', installedBytes: 120000000, defaultStrength: 1, strengthAdjustable: false, chosen: false, strength: 1,
              download: { running: false, state: 'Cancelled' } };
            R['/api/agent/loras'] = loraRows([{ id: 'newer-build', strength: 0.8 }, { id: 'viggle', strength: 1 }], [gone, stopped]);
            R['/api/agent/loras/choice'] = function (call) {
              return loraRows(call.body.loras, [Object.assign({}, gone, { chosen: false }), stopped]);
            };
            R['/api/agent/loras/pruna'] = { __status: 500, body: { error: 'The server failed to handle the request.' } };
            """, """
            return openLoras().then(function () {
              function actions(r) { return r.querySelectorAll('BUTTON').map(function (b) { return b.textContent; }); }
              var viggle = loraRow('viggle');
              var box = viggle.querySelectorAll('INPUT')[0];
              var before = {
                viggleSwitchOn: !!(box && box.checked),
                viggleActions: actions(viggle),
                viggleNote: viggle.querySelectorAll('.err').map(function (e) { return e.textContent; }),
                prunaActions: actions(loraRow('pruna'))
              };
              box.checked = false;
              box.dispatch('change');
              return settle(20).then(function () {
                var remove = loraRow('pruna').querySelectorAll('BUTTON').filter(function (b) { return b.textContent === 'Remove'; })[0];
                remove.dispatch('click');
                return settle(30).then(function () {
                  before.sent = __page.requests('/api/agent/loras/choice').map(function (c) { return c.body.loras.map(function (l) { return l.id; }).join('+'); });
                  before.rowsAfterFailedRemove = __page.byId['lora-list'].querySelectorAll('.lorarow').length;
                  before.sheetError = __page.byId['lora-error'].textContent;
                  before.notices = __page.notices();
                  return before;
                });
              });
            });
            """);

        Assert.True(result.GetProperty("viggleSwitchOn").GetBoolean());
        Assert.Equal(new[] { "Download" }, Strings(result, "viggleActions"));
        Assert.Contains(Strings(result, "viggleNote"), n => n.Contains("files are missing", StringComparison.Ordinal));
        Assert.Equal(new[] { "Download", "Remove" }, Strings(result, "prunaActions"));
        // Turned off: the rest of the choice goes back, a newer build's plug-in included.
        Assert.Equal(new[] { "newer-build" }, Strings(result, "sent"));
        Assert.Equal(4, result.GetProperty("rowsAfterFailedRemove").GetInt32());
        Assert.Equal("Could not remove Pruna 8-Step: The server failed to handle the request.", result.GetProperty("sheetError").GetString());
        Assert.DoesNotContain(Strings(result, "notices"), n => n.Contains("was removed", StringComparison.Ordinal));
    }

    /// <summary>Requests a test answers itself, in any order: <c>hold(path)</c> makes that route wait.</summary>
    private const string Held = """
        var held = [];
        function hold(path) {
          R[path] = function (call) {
            return new Promise(function (answer) { held.push({ path: path, call: call, answer: answer }); });
          };
        }
        function heldFor(path) { return held.filter(function (h) { return h.path === path; }); }
        function ids(call) { return call.body.loras.map(function (l) { return l.id; }).join('+'); }
        """;

    /// <summary>
    /// Each save replaces the whole choice, so they go one at a time, in the order they were
    /// made: two in flight together could reach the host in the other order and leave the
    /// first change as the last word.
    /// </summary>
    [Fact]
    public void LoraSavesGoOneAtATimeInTheOrderTheyWereMade()
    {
        JsonElement result = Run(ImageModel + TwoStyles + Held, """
            return openLoras().then(function () {
              hold('/api/agent/loras/choice');
              var film = loraRow('film').querySelectorAll('INPUT')[0];
              var grain = loraRow('grain').querySelectorAll('INPUT')[0];
              film.checked = true;
              film.dispatch('change');
              grain.checked = true;
              grain.dispatch('change');
              return settle(20).then(function () {
                var out = { inFlight: heldFor('/api/agent/loras/choice').length };
                var first = heldFor('/api/agent/loras/choice')[0];
                out.first = ids(first.call);
                first.answer(loraRows(first.call.body.loras));
                return settle(20).then(function () {
                  var all = heldFor('/api/agent/loras/choice');
                  out.afterFirst = all.length;
                  out.second = all.length > 1 ? ids(all[1].call) : '';
                  if (all.length > 1) all[1].answer(loraRows(all[1].call.body.loras));
                  return settle(20).then(function () {
                    out.names = __page.byId['lora-list'].querySelectorAll('.nm').map(function (n) { return n.textContent; });
                    return out;
                  });
                });
              });
            });
            """);

        Assert.Equal(1, result.GetProperty("inFlight").GetInt32());
        Assert.Equal("film", result.GetProperty("first").GetString());
        Assert.Equal(2, result.GetProperty("afterFirst").GetInt32());
        Assert.Equal("film+grain", result.GetProperty("second").GetString());
        Assert.Equal(new[] { "● Film Stills", "● Grainscape" }, Strings(result, "names"));
    }

    /// <summary>
    /// A list the page asked for before a change was saved, and that arrives after it, is not
    /// painted: it would put the switch the user just turned on back off.
    /// </summary>
    [Fact]
    public void AListReadBeforeALoraChangeIsNotPaintedOverIt()
    {
        JsonElement result = Run(ImageModel + TwoStyles + Held + """
            var notYet = { id: 'pruna', name: 'Pruna 8-Step', kind: 'Speed', steps: 8, purpose: 'Fast.', license: 'QRL', totalBytes: 335606104,
              state: 'NotInstalled', installedBytes: 0, defaultStrength: 1, strengthAdjustable: false, chosen: false, strength: 1, download: null };
            R['/api/agent/loras'] = loraRows([], [notYet]);
            R['/api/agent/loras/choice'] = function (call) { return loraRows(call.body.loras, [notYet]); };
            """, """
            return openLoras().then(function () {
              hold('/api/agent/loras');
              // Starting a download reads the list again; that read is held.
              loraRow('pruna').querySelectorAll('BUTTON').filter(function (b) { return b.textContent === 'Download'; })[0].dispatch('click');
              return settle(20).then(function () {
                var film = loraRow('film').querySelectorAll('INPUT')[0];
                film.checked = true;
                film.dispatch('change');
                return settle(20).then(function () {
                  var stale = heldFor('/api/agent/loras');
                  stale.forEach(function (h) { h.answer(loraRows([], [notYet])); });
                  return settle(20).then(function () {
                    return {
                      staleReads: stale.length,
                      names: __page.byId['lora-list'].querySelectorAll('.nm').map(function (n) { return n.textContent; }),
                      filmOn: loraRow('film').querySelectorAll('INPUT').filter(function (i) { return i.type === 'checkbox'; })[0].checked
                    };
                  });
                });
              });
            });
            """);

        Assert.True(result.GetProperty("staleReads").GetInt32() >= 1);
        Assert.Contains("● Film Stills", Strings(result, "names"));
        Assert.True(result.GetProperty("filmOn").GetBoolean());
    }

    /// <summary>
    /// A change made while a removal is on its way builds on the choice without the removed
    /// plug-in. Built on the list painted before, it sent the plug-in back, and the host,
    /// which had just deleted its files, refused the whole change.
    /// </summary>
    [Fact]
    public void AChangeMadeWhileALoraIsBeingRemovedLeavesItOut()
    {
        JsonElement result = Run(ImageModel + TwoStyles + Held + """
            R['/api/agent/loras'] = loraRows([{ id: 'film', strength: 0.7 }, { id: 'grain', strength: 0.7 }]);
            """, """
            return openLoras().then(function () {
              hold('/api/agent/loras/film');
              loraRow('film').querySelectorAll('BUTTON').filter(function (b) { return b.textContent === 'Remove'; })[0].dispatch('click');
              return settle(20).then(function () {
                var slider = loraRow('grain').querySelectorAll('INPUT').filter(function (i) { return i.type === 'range'; })[0];
                slider.value = '0.5';
                slider.dispatch('change');
                return settle(20).then(function () {
                  var sentBeforeRemoval = __page.requests('/api/agent/loras/choice').length;
                  heldFor('/api/agent/loras/film')[0].answer(loraRows([{ id: 'grain', strength: 0.7 }]));
                  return settle(30).then(function () {
                    var sent = __page.requests('/api/agent/loras/choice');
                    return {
                      sentBeforeRemoval: sentBeforeRemoval,
                      sent: sent.map(function (c) { return c.body.loras.map(function (l) { return l.id + '@' + l.strength; }).join('+'); })
                    };
                  });
                });
              });
            });
            """);

        Assert.Equal(0, result.GetProperty("sentBeforeRemoval").GetInt32());
        Assert.Equal(new[] { "grain@0.5" }, Strings(result, "sent"));
    }

    /// <summary>
    /// An earlier save that fails, followed by a later one that saves the whole choice,
    /// leaves no error behind: the later save answered for everything the earlier one sent.
    /// </summary>
    [Fact]
    public void AnEarlierLoraSavesFailureIsNotLeftOnceALaterOneSucceeds()
    {
        JsonElement result = Run(ImageModel + TwoStyles + Held, """
            return openLoras().then(function () {
              hold('/api/agent/loras/choice');
              var film = loraRow('film').querySelectorAll('INPUT')[0];
              var grain = loraRow('grain').querySelectorAll('INPUT')[0];
              film.checked = true;
              film.dispatch('change');
              grain.checked = true;
              grain.dispatch('change');
              return settle(20).then(function () {
                heldFor('/api/agent/loras/choice')[0].answer({ __reject: 'Load failed' });
                return settle(20).then(function () {
                  var second = heldFor('/api/agent/loras/choice')[1];
                  second.answer(loraRows(second.call.body.loras));
                  return settle(20).then(function () {
                    return {
                      sheetError: __page.byId['lora-error'].textContent,
                      names: __page.byId['lora-list'].querySelectorAll('.nm').map(function (n) { return n.textContent; })
                    };
                  });
                });
              });
            });
            """);

        Assert.Equal("", result.GetProperty("sheetError").GetString());
        Assert.Equal(new[] { "● Film Stills", "● Grainscape" }, Strings(result, "names"));
    }

    /// <summary>
    /// A refused save puts the switches back where the host last had them at once, with the
    /// host's own words, even when the list cannot be read again: the switch the user flipped
    /// does not stay on, and the next change does not build on it.
    /// </summary>
    [Fact]
    public void ARefusedLoraSaveRepaintsTheHostsLastAnswerWhenTheReloadFails()
    {
        JsonElement result = Run(ImageModel + TwoStyles + Held, """
            return openLoras().then(function () {
              hold('/api/agent/loras/choice');
              var film = loraRow('film').querySelectorAll('INPUT')[0];
              var grain = loraRow('grain').querySelectorAll('INPUT')[0];
              film.checked = true;
              film.dispatch('change');
              grain.checked = true;
              grain.dispatch('change');
              return settle(20).then(function () {
                // Film Stills is saved, but its answer is not painted: a later change is on its way.
                heldFor('/api/agent/loras/choice')[0].answer(loraRows([{ id: 'film', strength: 0.7 }]));
                return settle(20).then(function () {
                  R['/api/agent/loras'] = { __reject: 'Load failed' };
                  heldFor('/api/agent/loras/choice')[1].answer({ __status: 400, body: { error: 'Grainscape is not downloaded yet.' } });
                  return settle(30).then(function () {
                    function box(id) { return loraRow(id).querySelectorAll('INPUT').filter(function (i) { return i.type === 'checkbox'; })[0]; }
                    return {
                      filmOn: box('film').checked,
                      grainOn: box('grain').checked,
                      names: __page.byId['lora-list'].querySelectorAll('.nm').map(function (n) { return n.textContent; }),
                      sheetError: __page.byId['lora-error'].textContent
                    };
                  });
                });
              });
            });
            """);

        Assert.True(result.GetProperty("filmOn").GetBoolean());
        Assert.False(result.GetProperty("grainOn").GetBoolean());
        Assert.Equal(new[] { "● Film Stills", "Grainscape" }, Strings(result, "names"));
        Assert.Equal("Grainscape is not downloaded yet.", result.GetProperty("sheetError").GetString());
    }

    /// <summary>
    /// While a plug-in downloads, the sheet polls every second. A tick that moves only the
    /// percentage leaves every row in place, so a strength slider being dragged on another
    /// row is not replaced under the finger; the finished download repaints the rows and
    /// the polling stops.
    /// </summary>
    [Fact]
    public void ADownloadsProgressLeavesTheRowsUnderTheFingerInPlace()
    {
        JsonElement result = Run(ImageModel + TwoStyles + """
            function downloading(fraction) {
              var running = { id: 'pruna', name: 'Pruna 8-Step', kind: 'Speed', steps: 8, purpose: 'Fast.', license: 'QRL', totalBytes: 335606104,
                state: 'Partial', installedBytes: 1000, defaultStrength: 1, strengthAdjustable: false, chosen: false, strength: 1,
                download: fraction === null ? null : { running: true, state: 'Running', progress: { fraction: fraction } } };
              if (fraction === null) { running.state = 'Installed'; running.installedBytes = 335606104; }
              return loraRows([{ id: 'film', strength: 0.6 }], [running]);
            }
            R['/api/agent/loras'] = downloading(0.25);
            """, """
            return openLoras().then(function () {
              var list = __page.byId['lora-list'];
              var film = loraRow('film');
              var slider = film.querySelectorAll('INPUT').filter(function (i) { return i.type === 'range'; })[0];
              var first = loraRow('pruna').querySelectorAll('.pct')[0].textContent;
              R['/api/agent/loras'] = downloading(0.5);
              return wait(1300).then(function () {
                var during = {
                  first: first,
                  second: loraRow('pruna').querySelectorAll('.pct')[0].textContent,
                  filmRowKept: film.parentNode === list,
                  sliderKept: slider.parentNode !== null && loraRow('film') === film
                };
                R['/api/agent/loras'] = downloading(null);
                return wait(1300).then(function () {
                  during.repaintedWhenDone = film.parentNode === null && loraRow('pruna').querySelectorAll('.pct').length === 0;
                  var polls = __page.requests('/api/agent/loras').length;
                  return wait(1300).then(function () {
                    during.pollsAfterDone = __page.requests('/api/agent/loras').length - polls;
                    return during;
                  });
                });
              });
            });
            """);

        Assert.Equal("25%", result.GetProperty("first").GetString());
        Assert.Equal("50%", result.GetProperty("second").GetString());
        Assert.True(result.GetProperty("filmRowKept").GetBoolean());
        Assert.True(result.GetProperty("sliderKept").GetBoolean());
        Assert.True(result.GetProperty("repaintedWhenDone").GetBoolean());
        Assert.Equal(0, result.GetProperty("pollsAfterDone").GetInt32());
    }

    /// <summary>
    /// The picture is not lost to the answer being painted: a turn that also writes text
    /// repaints the bubble, and the picture is put back under it.
    /// </summary>
    [WebJavaScriptFact]
    public void APictureSurvivesTextPaintedIntoTheSameBubble()
    {
        JsonElement result = Run(ImageModel + """
            R['/api/chat'] = { __sse: [
              { image_step: 1, image_steps: 1, preview: 'data:image/png;base64,AAAA' },
              { token: 'Here it is.' },
              { imageUrl: '/uploads/boat.png', width: 1024, height: 1024 },
              { done: true, sessionId: 's1', truncated: false }
            ] };
            """, """
            __page.byId['text'].value = 'a blue boat';
            __page.byId['send'].dispatch('click');
            return settle(30).then(function () {
              var turns = __page.transcript();
              var answer = turns[turns.length - 1];
              return {
                images: answer.media.filter(function (m) { return m.tag === 'IMG'; }).map(function (m) { return m.src; }),
                html: answer.html
              };
            });
            """);

        Assert.Equal(new[] { "/uploads/boat.png" }, Strings(result, "images"));
        Assert.Contains("Here it is.", result.GetProperty("html").GetString(), StringComparison.Ordinal);
    }

    /// <summary>
    /// A photo is an edit, and an edit needs to be told what to change: sent alone, its
    /// file name would become the instruction. The draft stays as it was.
    /// </summary>
    [WebJavaScriptFact]
    public void APhotoWithNoWordsIsNotSentToAnImageModel()
    {
        JsonElement result = Run(ImageModel, """
            window.TensorAgent.addAttachment({ ok: true, file: 'a1.png', fileName: 'street.png',
                                               mediaType: 'image', url: '/uploads/a1.png' });
            __page.byId['text'].value = '';
            __page.byId['send'].dispatch('click');
            return settle(10).then(function () {
              return {
                sent: __page.requests('/api/chat').length,
                attachments: window.TensorAgent.attachmentCount(),
                notices: __page.notices()
              };
            });
            """);

        Assert.False(result.TryGetProperty("error", out JsonElement failure), failure.ToString());
        Assert.Equal(0, result.GetProperty("sent").GetInt32());
        Assert.Equal(1, result.GetProperty("attachments").GetInt32());
        Assert.Contains(Strings(result, "notices"), n => n.Contains("Say what to change", StringComparison.Ordinal));
    }

    /// <summary>
    /// An image model edits the photo rather than looking at it, so the page does not
    /// stop the send to re-check for a vision file the way it does for a chat model —
    /// the host refuses an edit itself when the file is missing — and the photo goes
    /// out as a still, which is what the host edits.
    /// </summary>
    [WebJavaScriptFact]
    public void APhotoAndAnInstructionGoOutAsAnEdit()
    {
        JsonElement result = Run(ImageModel + """
            R['/api/chat'] = { __sse: [
              { imageUrl: '/uploads/edited.png', width: 1024, height: 768 },
              { done: true, sessionId: 's1', truncated: false }
            ] };
            """, """
            window.TensorAgent.addAttachment({ ok: true, file: 'a1.png', fileName: 'street.png',
                                               mediaType: 'image', url: '/uploads/a1.png' });
            __page.byId['text'].value = 'make it night';
            __page.byId['send'].dispatch('click');
            return settle(30).then(function () {
              return { sent: __page.requests('/api/chat').map(function (c) { return c.body; }) };
            });
            """);

        JsonElement message = Assert.Single(result.GetProperty("sent").EnumerateArray()).GetProperty("messages")[0];
        Assert.Equal("make it night", message.GetProperty("content").GetString());
        Assert.Equal(new[] { "a1.png" }, Strings(message, "stillImagePaths"));
    }

    // ---- a video model (VideoTurns on the host) -----------------------------------------

    // What /api/models reports for MiniMax-H3's two checkpoints. The keyframes one (FL2VA)
    // starts a clip from a photo and can end it on a second; the references one (Ref2VA)
    // puts what it is shown into a scene of its own. They share one architecture, so the
    // page decides everything from `video`.
    private const string KeyframesVideo = """
        { family: 'minimax-h3', supportsAudio: true, supportsImageConditioning: true,
          supportsEndImageConditioning: true, supportsReferenceConditioning: false, maxReferenceImages: 0 }
        """;

    private const string ReferencesVideo = """
        { family: 'minimax-h3', supportsAudio: true, supportsImageConditioning: true,
          supportsEndImageConditioning: false, supportsReferenceConditioning: true, maxReferenceImages: 9 }
        """;

    private static string VideoModel(string video) => $$"""
        R['/api/models'] = { loaded: 'minimax-h3-q4k.gguf', architecture: 'minimax-h3',
                             loadedBackend: 'ggml_metal', visionReady: false, video: {{video}} };
        """;

    private static string[] Tags(JsonElement turn) =>
        turn.GetProperty("media").EnumerateArray().Select(m => m.GetProperty("tag").GetString()!).ToArray();

    /// <summary>
    /// The composer says what the loaded checkpoint can be given, and the two differ: one
    /// starts a clip from a photo, the other features the photos, clips and sounds it is
    /// shown. A model that reports no video capability is a chat model, whatever its
    /// architecture is called.
    /// </summary>
    [WebJavaScriptTheory]
    [InlineData(KeyframesVideo, "Describe a video… or attach a photo to start it from")]
    [InlineData(ReferencesVideo, "Describe a video… attach photos, clips or sounds it should feature")]
    [InlineData("""
        { family: 'minimax-h3', supportsAudio: true, supportsImageConditioning: false,
          supportsEndImageConditioning: false, supportsReferenceConditioning: false, maxReferenceImages: 0 }
        """, "Describe a video…")]
    [InlineData("null", "Message… or hold to talk")]
    public void AVideoModelsComposerSaysWhatItsCheckpointCanBeGiven(string video, string expected)
    {
        JsonElement result = Run(VideoModel(video), """
            return { placeholder: __page.byId['text'].placeholder };
            """);

        Assert.Equal(expected, result.GetProperty("placeholder").GetString());
    }

    /// <summary>
    /// A video is filmed from its description; a photo beside it is only where it starts.
    /// Sent with no words, the file's name would be the whole script, so nothing is sent,
    /// the user is told what is missing, and the photo stays where it was.
    /// </summary>
    [WebJavaScriptTheory]
    [InlineData(true)]
    [InlineData(false)]
    public void AVideoIsNotSentWithoutADescription(bool withPhoto)
    {
        string attach = withPhoto
            ? "window.TensorAgent.addAttachment({ ok: true, file: 'a1.png', fileName: 'beach.png', mediaType: 'image', url: '/uploads/a1.png' });\n"
            : string.Empty;
        JsonElement result = Run(VideoModel(KeyframesVideo), attach + """
            __page.byId['text'].value = '';
            __page.byId['send'].dispatch('click');
            return settle(10).then(function () {
              return {
                sent: __page.requests('/api/chat').length,
                attachments: window.TensorAgent.attachmentCount(),
                notices: __page.notices(),
                generating: window.TensorAgent.isGenerating()
              };
            });
            """);

        Assert.False(result.TryGetProperty("error", out JsonElement failure), failure.ToString());
        Assert.Equal(0, result.GetProperty("sent").GetInt32());
        Assert.Equal(withPhoto ? 1 : 0, result.GetProperty("attachments").GetInt32());
        Assert.Equal(new[] { "Describe the video you want, then send." }, Strings(result, "notices"));
        Assert.False(result.GetProperty("generating").GetBoolean());
    }

    /// <summary>
    /// A photo is the clip's first frame, not something the model has to see, so the send
    /// is not held for the vision re-check a chat model gets. That check would refuse it:
    /// this model reports no vision file. The photo goes out as a still, which is what
    /// the host makes a keyframe of.
    /// </summary>
    [WebJavaScriptFact]
    public void APhotoAndADescriptionGoToAVideoModelWithoutAVisionCheck()
    {
        JsonElement result = Run(VideoModel(KeyframesVideo) + """
            R['/api/chat'] = { __sse: [{ done: true, sessionId: 's1', truncated: false }] };
            """, """
            window.TensorAgent.addAttachment({ ok: true, file: 'a1.png', fileName: 'beach.png',
                                               mediaType: 'image', url: '/uploads/a1.png' });
            var modelReads = __page.requests('/api/models').length;
            __page.byId['text'].value = 'the tide comes in over the sand';
            __page.byId['send'].dispatch('click');
            return settle(30).then(function () {
              return {
                sent: __page.requests('/api/chat').map(function (c) { return c.body; }),
                rereads: __page.requests('/api/models').length - modelReads,
                notices: __page.notices()
              };
            });
            """);

        JsonElement message = Assert.Single(result.GetProperty("sent").EnumerateArray()).GetProperty("messages")[0];
        Assert.Equal("the tide comes in over the sand", message.GetProperty("content").GetString());
        Assert.Equal(new[] { "a1.png" }, Strings(message, "stillImagePaths"));
        Assert.Equal(0, result.GetProperty("rereads").GetInt32());
        Assert.Empty(Strings(result, "notices"));
    }

    /// <summary>
    /// The references checkpoint can be shown all three kinds of thing at once. Each goes
    /// out under the list the host reads it from, and the clip's frames do not hold the
    /// send for a vision re-check either.
    /// </summary>
    [WebJavaScriptFact]
    public void PhotosClipsAndSoundsAllReachAReferencesVideoModel()
    {
        JsonElement result = Run(VideoModel(ReferencesVideo) + """
            R['/api/chat'] = { __sse: [{ done: true, sessionId: 's1', truncated: false }] };
            """, """
            window.TensorAgent.addAttachment({ ok: true, file: 'a1.png', fileName: 'dog.png', mediaType: 'image', url: '/uploads/a1.png' });
            window.TensorAgent.addAttachment({ ok: true, file: 'a2.mp4', fileName: 'walk.mp4', mediaType: 'video', url: '/uploads/a2.mp4',
                                               frames: ['a2_0001.png'] });
            window.TensorAgent.addAttachment({ ok: true, file: 'a3.wav', fileName: 'bark.wav', mediaType: 'audio', url: '/uploads/a3.wav' });
            __page.byId['text'].value = 'the dog runs along the beach, barking';
            __page.byId['send'].dispatch('click');
            return settle(30).then(function () {
              return { sent: __page.requests('/api/chat').map(function (c) { return c.body; }), notices: __page.notices() };
            });
            """);

        JsonElement message = Assert.Single(result.GetProperty("sent").EnumerateArray()).GetProperty("messages")[0];
        Assert.Equal("the dog runs along the beach, barking", message.GetProperty("content").GetString());
        Assert.Equal(new[] { "a1.png" }, Strings(message, "stillImagePaths"));
        Assert.Equal(new[] { "a2.mp4" }, Strings(message, "videoFilePaths"));
        Assert.Equal(new[] { "a3.wav" }, Strings(message, "audioPaths"));
        Assert.Empty(Strings(result, "notices"));
    }

    /// <summary>
    /// A clip takes minutes, in stages the host names: the description is read, the clip
    /// denoises step by step, then its frames, its sound and the MP4 are made. The strip
    /// says each in turn, with about how long is left while it films. The turn ends with
    /// one player in the bubble and the clip in the history, so the next request does not
    /// drop it from the saved chat. The sound is inside the MP4 here, so there is nothing
    /// else to play, whether the frame leaves audioUrl out or sends it as null. Null is
    /// what the host actually writes, because it keeps nulls in its frames.
    /// </summary>
    [WebJavaScriptTheory]
    [InlineData("")]
    [InlineData("audioUrl: null, ")]
    public void AVideoModelsTurnIsOneClipInTheBubbleAndTheHistory(string muxedSound)
    {
        JsonElement result = Run(VideoModel(KeyframesVideo) + $$"""
            R['/api/chat'] = { __sse: [
              { video_step: 0, video_steps: 3, video_phase: 'text-encode', elapsed: 2.1, eta: -1 },
              { video_step: 1, video_steps: 3, video_phase: 'denoise', elapsed: 41.2, eta: 210.5 },
              { video_step: 2, video_steps: 3, video_phase: 'denoise', elapsed: 80.3, eta: 42.4 },
              { video_step: 3, video_steps: 3, video_phase: 'denoise', elapsed: 119.0, eta: 0 },
              { video_step: 3, video_steps: 3, video_phase: 'vae-decode', elapsed: 125.6, eta: -1 },
              { video_step: 3, video_steps: 3, video_phase: 'audio-decode', elapsed: 131.0, eta: -1 },
              { video_step: 3, video_steps: 3, video_phase: 'encode', elapsed: 133.2, eta: -1 },
              { videoUrl: '/uploads/video-9b1e4c7d2a6f8e3b5c0d1a2f4e6b8c9d.mp4', {{muxedSound}}
                width: 640, height: 384, frames: 22, fps: 24, seed: 12345, hasAudio: true },
              { done: true, sessionId: 's1', tokenCount: 0, elapsed: 133.9, tokPerSec: 0.0, truncated: false }
            ] };
            """, """
            var shownBefore = __page.progress().length;
            __page.byId['text'].value = 'a lighthouse at dusk, waves breaking below';
            __page.byId['send'].dispatch('click');
            return settle(40).then(function () {
              var turns = __page.transcript();
              var answer = turns[turns.length - 1];
              return {
                sent: __page.requests('/api/chat').map(function (c) { return c.body; }),
                role: answer.role,
                media: answer.media,
                shown: __page.progress().slice(shownBefore),
                history: window.TensorAgent.history(),
                generating: window.TensorAgent.isGenerating(),
                errors: __page.errorNotices()
              };
            });
            """);

        JsonElement sent = Assert.Single(result.GetProperty("sent").EnumerateArray());
        Assert.Equal("a lighthouse at dusk, waves breaking below", sent.GetProperty("messages")[0].GetProperty("content").GetString());

        Assert.Equal("assistant", result.GetProperty("role").GetString());
        JsonElement clip = Assert.Single(result.GetProperty("media").EnumerateArray());
        Assert.Equal("VIDEO", clip.GetProperty("tag").GetString());
        Assert.Equal("/uploads/video-9b1e4c7d2a6f8e3b5c0d1a2f4e6b8c9d.mp4", clip.GetProperty("src").GetString());
        Assert.True(clip.GetProperty("controls").GetBoolean());
        Assert.True(clip.GetProperty("playsInline").GetBoolean());
        Assert.True(clip.GetProperty("loop").GetBoolean());
        Assert.Equal("metadata", clip.GetProperty("preload").GetString());

        Assert.Equal(new[]
        {
            "Filming…",
            "Reading the description…",
            "Filming… step 1 of 3 · about 4 min left",
            "Filming… step 2 of 3 · about 42 s left",
            "Filming… step 3 of 3",
            "Developing the frames…",
            "Adding the sound…",
            "Saving the video…",
        }, Strings(result, "shown"));

        JsonElement history = result.GetProperty("history");
        Assert.Equal(2, history.GetArrayLength());
        JsonElement made = history[1];
        Assert.Equal("assistant", made.GetProperty("role").GetString());
        Assert.Equal("", made.GetProperty("content").GetString());
        Assert.Equal("/uploads/video-9b1e4c7d2a6f8e3b5c0d1a2f4e6b8c9d.mp4", made.GetProperty("videoUrl").GetString());
        Assert.False(made.TryGetProperty("audioUrl", out _), "the sound is inside the MP4, so there is no second file to keep");
        Assert.False(result.GetProperty("generating").GetBoolean());
        Assert.Empty(Strings(result, "errors"));
    }

    /// <summary>
    /// The time left is said the way a person would say it: minutes past a minute and a
    /// half, seconds under it, and never "0 s". A stage this page has no name for still
    /// says it is filming, rather than leaving the last stage's words up.
    /// </summary>
    [WebJavaScriptFact]
    public void TheTimeLeftWhileFilmingReadsTheWayAPersonWouldSayIt()
    {
        JsonElement result = Run(VideoModel(KeyframesVideo) + """
            R['/api/chat'] = { __sse: [
              { video_step: 1, video_steps: 4, video_phase: 'denoise', elapsed: 30, eta: 91 },
              { video_step: 2, video_steps: 4, video_phase: 'denoise', elapsed: 60, eta: 90 },
              { video_step: 3, video_steps: 4, video_phase: 'denoise', elapsed: 90, eta: 0.3 },
              { video_step: 4, video_steps: 4, video_phase: 'interpolate', elapsed: 95, eta: -1 },
              { videoUrl: '/uploads/video-5e5e5e5e5e5e5e5e5e5e5e5e5e5e5e5e.mp4', hasAudio: true },
              { done: true, sessionId: 's1', tokenCount: 0, elapsed: 96, tokPerSec: 0.0, truncated: false }
            ] };
            """, """
            var shownBefore = __page.progress().length;
            __page.byId['text'].value = 'snow falling on a quiet street';
            __page.byId['send'].dispatch('click');
            return settle(30).then(function () {
              return { shown: __page.progress().slice(shownBefore), generating: window.TensorAgent.isGenerating() };
            });
            """);

        Assert.Equal(new[]
        {
            "Filming…",
            "Filming… step 1 of 4 · about 2 min left",
            "Filming… step 2 of 4 · about 90 s left",
            "Filming… step 3 of 4 · about 1 s left",
            "Filming…",
        }, Strings(result, "shown"));
        Assert.False(result.GetProperty("generating").GetBoolean());
    }

    /// <summary>
    /// When the host keeps the soundtrack as a file of its own instead of inside the MP4,
    /// the page plays it: one player for the sound, right under the clip, and both files
    /// in the history.
    /// </summary>
    [WebJavaScriptFact]
    public void ASoundtrackKeptAsItsOwnFilePlaysUnderTheClip()
    {
        JsonElement result = Run(VideoModel(KeyframesVideo) + """
            R['/api/chat'] = { __sse: [
              { video_step: 1, video_steps: 1, video_phase: 'denoise', elapsed: 40, eta: -1 },
              { videoUrl: '/uploads/video-0a1b2c3d4e5f60718293a4b5c6d7e8f9.mp4',
                audioUrl: '/uploads/video-0a1b2c3d4e5f60718293a4b5c6d7e8f9.wav',
                width: 640, height: 384, frames: 22, fps: 24, seed: 7, hasAudio: true },
              { done: true, sessionId: 's1', tokenCount: 0, elapsed: 61.5, tokPerSec: 0.0, truncated: false }
            ] };
            """, """
            __page.byId['text'].value = 'rain on a tin roof';
            __page.byId['send'].dispatch('click');
            return settle(30).then(function () {
              var turns = __page.transcript();
              return { answer: turns[turns.length - 1], history: window.TensorAgent.history() };
            });
            """);

        JsonElement answer = result.GetProperty("answer");
        Assert.Equal(new[] { "VIDEO", "AUDIO" }, Tags(answer));
        JsonElement media = answer.GetProperty("media");
        Assert.Equal("/uploads/video-0a1b2c3d4e5f60718293a4b5c6d7e8f9.mp4", media[0].GetProperty("src").GetString());
        Assert.Equal("/uploads/video-0a1b2c3d4e5f60718293a4b5c6d7e8f9.wav", media[1].GetProperty("src").GetString());
        Assert.True(media[1].GetProperty("controls").GetBoolean());
        Assert.Equal("metadata", media[1].GetProperty("preload").GetString());

        JsonElement made = result.GetProperty("history")[1];
        Assert.Equal("/uploads/video-0a1b2c3d4e5f60718293a4b5c6d7e8f9.mp4", made.GetProperty("videoUrl").GetString());
        Assert.Equal("/uploads/video-0a1b2c3d4e5f60718293a4b5c6d7e8f9.wav", made.GetProperty("audioUrl").GetString());
    }

    /// <summary>
    /// The page paints once per read, and a read can carry the words and the clip
    /// together. The clip goes under the words, so the words are painted first: painted
    /// after, they would replace what is in the bubble and take the player with them.
    /// </summary>
    [WebJavaScriptFact]
    public void AClipGoesUnderTheWordsOfTheSameTurn()
    {
        JsonElement result = Run(VideoModel(KeyframesVideo) + """
            R['/api/chat'] = { __sse: [{ __chunk: [
              { token: 'Here it is.' },
              { videoUrl: '/uploads/video-1f2e3d4c5b6a79881f2e3d4c5b6a7988.mp4', hasAudio: true },
              { done: true, sessionId: 's1', truncated: false }
            ] }] };
            """, """
            __page.byId['text'].value = 'a red kite over a field';
            __page.byId['send'].dispatch('click');
            return settle(30).then(function () {
              var turns = __page.transcript();
              var answer = turns[turns.length - 1];
              return { media: answer.media, html: answer.html };
            });
            """);

        JsonElement clip = Assert.Single(result.GetProperty("media").EnumerateArray());
        Assert.Equal("VIDEO", clip.GetProperty("tag").GetString());
        Assert.Equal("/uploads/video-1f2e3d4c5b6a79881f2e3d4c5b6a7988.mp4", clip.GetProperty("src").GetString());
        Assert.Contains("Here it is.", result.GetProperty("html").GetString(), StringComparison.Ordinal);
    }

    /// <summary>
    /// Leaving the app while a clip is made and coming back re-attaches to the turn, and
    /// the host replays it from its first frame, all at once, into the same bubble. The
    /// clip and its sound were already showing; the bubble still ends with one of each,
    /// and the history with one entry for the turn.
    /// </summary>
    [WebJavaScriptFact]
    public void ComingBackToAVideoTurnDoesNotStackASecondPlayer()
    {
        JsonElement result = Run(VideoModel(KeyframesVideo) + $$"""
            var frames = [
              { video_step: 0, video_steps: 2, video_phase: 'text-encode', elapsed: 2, eta: -1 },
              { video_step: 1, video_steps: 2, video_phase: 'denoise', elapsed: 40, eta: 38 },
              { video_step: 2, video_steps: 2, video_phase: 'encode', elapsed: 80, eta: -1 },
              { videoUrl: '/uploads/video-aa55aa55aa55aa55aa55aa55aa55aa55.mp4',
                audioUrl: '/uploads/video-aa55aa55aa55aa55aa55aa55aa55aa55.wav',
                width: 640, height: 384, frames: 22, fps: 24, seed: 9, hasAudio: true }
            ];
            R['/api/chat'] = { __status: 200, headers: { {{TurnHeader}} }, body: { __sse: frames, __then: 'hang' } };
            R['/api/agent/turns'] = { turn: { id: 't1', running: true } };
            R['/api/agent/turns/t1'] = { __sse: [{ __chunk: frames.concat([
              { done: true, sessionId: 's1', tokenCount: 0, elapsed: 81, tokPerSec: 0.0, truncated: false }
            ]) }] };
            """, """
            window.TensorAgent.__testing.streamTrustMs(0);
            __page.byId['text'].value = 'a paper boat in a gutter stream';
            __page.byId['send'].dispatch('click');
            return settle(20).then(function () {
              var before = __page.transcript();
              document.visibilityState = 'hidden';
              document.dispatch('visibilitychange', {});
              document.visibilityState = 'visible';
              document.dispatch('visibilitychange', {});
              return wait(500).then(function () {
                return {
                  before: before,
                  after: __page.transcript(),
                  history: window.TensorAgent.history(),
                  generating: window.TensorAgent.isGenerating(),
                  attaches: __page.requests('/api/agent/turns/t1').length
                };
              });
            });
            """);

        JsonElement before = Assert.Single(Bubbles(result.GetProperty("before")), t => t.GetProperty("role").GetString() == "assistant");
        Assert.Equal(new[] { "VIDEO", "AUDIO" }, Tags(before));
        Assert.Equal(1, result.GetProperty("attaches").GetInt32());

        JsonElement after = Assert.Single(Bubbles(result.GetProperty("after")), t => t.GetProperty("role").GetString() == "assistant");
        Assert.Equal(new[] { "VIDEO", "AUDIO" }, Tags(after));
        Assert.Equal("/uploads/video-aa55aa55aa55aa55aa55aa55aa55aa55.mp4", after.GetProperty("media")[0].GetProperty("src").GetString());
        Assert.Equal("/uploads/video-aa55aa55aa55aa55aa55aa55aa55aa55.wav", after.GetProperty("media")[1].GetProperty("src").GetString());

        var history = result.GetProperty("history").EnumerateArray().ToList();
        Assert.Equal(2, history.Count);
        Assert.Equal("/uploads/video-aa55aa55aa55aa55aa55aa55aa55aa55.mp4", history[1].GetProperty("videoUrl").GetString());
        Assert.Equal("/uploads/video-aa55aa55aa55aa55aa55aa55aa55aa55.wav", history[1].GetProperty("audioUrl").GetString());
        Assert.False(result.GetProperty("generating").GetBoolean());
    }

    /// <summary>
    /// A clip comes back with the chat it was made in: the player again, and a player
    /// for the sound only where the sound was a file of its own. The host sends a saved
    /// message with every field, null where there is nothing, so the first clip's
    /// soundtrack arrives as a null and must not become an empty player. The history the
    /// next request is built from still names both files, so that request does not
    /// delete them from the saved chat.
    /// </summary>
    [WebJavaScriptFact]
    public void AReopenedChatShowsItsClipsAgain()
    {
        JsonElement result = Run("""
            R['/api/agent/conversations'] = { conversations: [{ id: 'saved', title: 'Lighthouse', updatedAt: '2026-09-30T10:00:00Z', messageCount: 4 }] };
            R['/api/sessions?conversation=saved'] = {
              sessionId: 's9', conversationId: 'saved', think: false, skills: [],
              messages: [
                { role: 'user', content: 'a lighthouse at dusk' },
                { role: 'assistant', content: '', videoUrl: '/uploads/video-c0ffeec0ffeec0ffeec0ffeec0ffee00.mp4', audioUrl: null },
                { role: 'user', content: 'the same, with gulls calling' },
                { role: 'assistant', content: '', videoUrl: '/uploads/video-d00dd00dd00dd00dd00dd00dd00dd00d.mp4',
                  audioUrl: '/uploads/video-d00dd00dd00dd00dd00dd00dd00dd00d.wav' }
              ]
            };
            R['/api/chat'] = { __sse: [{ token: 'The second one.' }, { done: true, truncated: false }] };
            """, """
            var shown = __page.transcript();
            __page.byId['text'].value = 'Which one has the gulls?';
            __page.byId['send'].dispatch('click');
            return settle(20).then(function () {
              return { shown: shown, sent: __page.requests('/api/chat').map(function (c) { return c.body; }) };
            });
            """);

        var turns = Bubbles(result.GetProperty("shown"));
        Assert.Equal(4, turns.Count);
        Assert.Equal(new[] { "VIDEO" }, Tags(turns[1]));
        JsonElement first = turns[1].GetProperty("media")[0];
        Assert.Equal("/uploads/video-c0ffeec0ffeec0ffeec0ffeec0ffee00.mp4", first.GetProperty("src").GetString());
        Assert.True(first.GetProperty("controls").GetBoolean());
        Assert.True(first.GetProperty("playsInline").GetBoolean());
        Assert.True(first.GetProperty("loop").GetBoolean());
        Assert.Equal("metadata", first.GetProperty("preload").GetString());
        Assert.Equal(new[] { "VIDEO", "AUDIO" }, Tags(turns[3]));
        JsonElement sound = turns[3].GetProperty("media")[1];
        Assert.Equal("/uploads/video-d00dd00dd00dd00dd00dd00dd00dd00d.wav", sound.GetProperty("src").GetString());
        Assert.True(sound.GetProperty("controls").GetBoolean());
        Assert.Equal("metadata", sound.GetProperty("preload").GetString());

        JsonElement messages = Assert.Single(result.GetProperty("sent").EnumerateArray()).GetProperty("messages");
        Assert.Equal(5, messages.GetArrayLength());
        Assert.Equal("/uploads/video-c0ffeec0ffeec0ffeec0ffeec0ffee00.mp4", messages[1].GetProperty("videoUrl").GetString());
        Assert.True(!messages[1].TryGetProperty("audioUrl", out JsonElement noSound) || noSound.ValueKind == JsonValueKind.Null,
            "the first clip's sound is inside its MP4, so the next request names no soundtrack for it");
        Assert.Equal("/uploads/video-d00dd00dd00dd00dd00dd00dd00dd00d.mp4", messages[3].GetProperty("videoUrl").GetString());
        Assert.Equal("/uploads/video-d00dd00dd00dd00dd00dd00dd00dd00d.wav", messages[3].GetProperty("audioUrl").GetString());
    }
}
