using System.Text.Json;

namespace TensorAgent.Tests;

public sealed partial class WebUiPageTests
{
    [WebJavaScriptFact]
    public void TextOnlyTurnAfterIngredientsDoesNotReuseImageReferenceMode()
    {
        JsonElement result = Run(VideoModel(LtxKeyframesVideo.Replace("supportsReferenceConditioning: false", "supportsReferenceConditioning: true")) + """
            R['/api/chat'] = { __sse: [{ done: true, sessionId: 's1', truncated: false }] };
            """, """
            window.TensorAgent.addAttachment({ok:true,file:'a.png',mediaType:'image'});
            var choice=__page.byId['chips'].querySelector('.video-guidance-mode'); choice.value='ref'; choice.dispatch('change');
            __page.byId['text'].value='The woman waves.'; __page.byId['send'].dispatch('click');
            return settle(30).then(function () {
              __page.byId['text'].value='A red fox runs through a field.'; __page.byId['send'].dispatch('click');
              return settle(30);
            }).then(function () {return {sent:__page.requests('/api/chat').map(function(call){return call.body;}),notices:__page.notices()};});
            """);
        Assert.False(result.TryGetProperty("error", out JsonElement error), error.ToString());
        Assert.Empty(Strings(result, "notices"));
        var requests = result.GetProperty("sent").EnumerateArray().ToArray();
        Assert.Equal(2, requests.Length);
        Assert.Equal("ref", requests[0].GetProperty("messages")[0].GetProperty("videoMode").GetString());
        JsonElement next = requests[1].GetProperty("messages").EnumerateArray().Last();
        Assert.Equal("user", next.GetProperty("role").GetString());
        Assert.False(next.TryGetProperty("videoMode", out _));
        Assert.False(next.TryGetProperty("stillImagePaths", out _));
    }

    [WebJavaScriptTheory]
    [InlineData("same")]
    [InlineData("capability_removed")]
    [InlineData("changed")]
    public void CapabilityRefreshKeepsDraftVideoChoicesOnlyForTheSameCompatibleCheckpoint(string refresh)
    {
        JsonElement result = Run(VideoModel(LtxKeyframesVideo.Replace("supportsReferenceConditioning: false", "supportsReferenceConditioning: true")) + """
            R['/api/chat'] = { __sse: [{ done: true, sessionId: 's1', truncated: false }] };
            """, $$"""
            window.TensorAgent.addAttachment({ok:true,file:'a.png',mediaType:'image'});
            var box=__page.byId['chips'];
            var length=box.querySelector('.video-clip-length'); length.value='97'; length.dispatch('change');
            var choice=box.querySelector('.video-guidance-mode'); choice.value='ref'; choice.dispatch('change');
            R['/api/models']=JSON.parse(JSON.stringify(R['/api/models']));
            if ({{JsonSerializer.Serialize(refresh)}}==='changed') R['/api/models'].loaded='another-ltx.gguf';
            if ({{JsonSerializer.Serialize(refresh)}}==='capability_removed') R['/api/models'].video.supportsReferenceConditioning=false;
            window.TensorAgent.refreshModel();
            return settle(30).then(function () {
              var guidance=box.querySelector('.video-guidance-mode');
              var identity=guidance && guidance.value==='ref';
              if (guidance) {guidance.value='keyframes';guidance.dispatch('change');}
              var frames=box.querySelector('.video-clip-length').value;
              __page.byId['text'].value='The woman waves.'; __page.byId['send'].dispatch('click');
              return settle(30).then(function () {return {identity:!!identity,frames:frames,
                sent:__page.requests('/api/chat').map(function(call){return call.body;}),notices:__page.notices()};});
            });
            """);
        Assert.False(result.TryGetProperty("error", out JsonElement error), error.ToString());
        Assert.Empty(Strings(result, "notices"));
        Assert.Equal(refresh == "same", result.GetProperty("identity").GetBoolean());
        Assert.Equal(refresh == "changed" ? "17" : "97", result.GetProperty("frames").GetString());
        var message = Assert.Single(result.GetProperty("sent").EnumerateArray()).GetProperty("messages")[0];
        if (refresh == "changed") Assert.False(message.TryGetProperty("videoFrames", out _));
        else Assert.Equal(97, message.GetProperty("videoFrames").GetInt32());
        Assert.False(message.TryGetProperty("videoMode", out _));
    }

    [WebJavaScriptFact]
    public void LongerClipChoiceUpdatesAutomaticGuidePositionsAndTheSavedRequest()
    {
        JsonElement result = Run(VideoModel(LtxKeyframesVideo) + """
            R['/api/chat'] = { __sse: [{ done: true, sessionId: 's1', truncated: false }] };
            """, """
            ['a.png','b.png','c.png'].forEach(function (file) {
              window.TensorAgent.addAttachment({ok:true,file:file,mediaType:'image'});
            });
            var box=__page.byId['chips'];
            var length=box.querySelector('.video-clip-length'); length.value='97'; length.dispatch('change');
            // Changing strength makes the complete automatic timeline explicit.
            var strength=box.querySelector('.video-frame-strength'); strength.value='.8'; strength.dispatch('change');
            __page.byId['text'].value='The woman blows a kiss.'; __page.byId['send'].dispatch('click');
            return settle(30).then(function () { return {sent:__page.requests('/api/chat').map(function (call) {return call.body;}),notices:__page.notices()}; });
            """);
        Assert.False(result.TryGetProperty("error", out JsonElement error), error.ToString());
        Assert.Empty(Strings(result, "notices"));
        var message = Assert.Single(result.GetProperty("sent").EnumerateArray()).GetProperty("messages")[0];
        Assert.Equal(97, message.GetProperty("videoFrames").GetInt32());
        Assert.Equal(new[] { 0, 48, 96 }, message.GetProperty("imageConditions").EnumerateArray().Select(c => c.GetProperty("frameIndex").GetInt32()));
        Assert.Equal(.8, message.GetProperty("imageConditions")[0].GetProperty("strength").GetDouble());
    }

    [WebJavaScriptFact]
    public void IngredientsIdentityChoiceSendsExplicitModeAndHidesTemporalControls()
    {
        JsonElement result = Run(VideoModel(LtxKeyframesVideo.Replace("supportsReferenceConditioning: false", "supportsReferenceConditioning: true")) + """
            R['/api/chat'] = { __sse: [{ done: true, sessionId: 's1', truncated: false }] };
            """, """
            window.TensorAgent.addAttachment({ ok:true, file:'a.png', mediaType:'image' });
            window.TensorAgent.addAttachment({ ok:true, file:'b.png', mediaType:'image' });
            var box = __page.byId['chips'];
            var choice = box.querySelector('.video-guidance-mode'); choice.value = 'ref'; choice.dispatch('change');
            var temporal = box.querySelectorAll('.video-frame-position').length;
            var guidance = box.querySelector('.video-reference-help').textContent;
            __page.byId['text'].value = 'The woman waves.'; __page.byId['send'].dispatch('click');
            return settle(30).then(function () { return { temporal:temporal, guidance:guidance, placeholder:__page.byId['text'].placeholder,
                sent:__page.requests('/api/chat').map(function (call) { return call.body; }), notices:__page.notices() }; });
            """);
        Assert.False(result.TryGetProperty("error", out JsonElement error), error.ToString());
        Assert.Equal(0, result.GetProperty("temporal").GetInt32());
        const string guidance = "Describe the target scene and each reference image by its position; clean subject images without text work best. Multiple references may retain the sheet's panel layout; one image gives more predictable composition.";
        Assert.Equal(guidance, result.GetProperty("placeholder").GetString());
        Assert.Equal(guidance, result.GetProperty("guidance").GetString());
        Assert.Empty(Strings(result, "notices"));
        var message = Assert.Single(result.GetProperty("sent").EnumerateArray()).GetProperty("messages")[0];
        Assert.Equal("ref", message.GetProperty("videoMode").GetString());
        Assert.False(message.TryGetProperty("imageConditions", out _));
        Assert.Equal(2, message.GetProperty("stillImagePaths").GetArrayLength());
    }

    private const string LtxKeyframesVideo = """
        { family: 'ltx-2.5', supportsAudio: true, supportsImageConditioning: true,
          supportsEndImageConditioning: true, supportsKeyframeConditioning: true,
          supportsReferenceConditioning: false, maxReferenceImages: 16, defaultPreviewFrames: 17 }
        """;

    [WebJavaScriptTheory]
    [InlineData(true)]
    [InlineData(false)]
    public void LtxImageFrameAndStrengthControlsReachChatWithoutAVisionProjector(bool lastOnly)
    {
        JsonElement result = Run(VideoModel(LtxKeyframesVideo) + """
            R['/api/chat'] = { __sse: [{ done: true, sessionId: 's1', truncated: false }] };
            """, $$"""
            window.TensorAgent.addAttachment({ ok: true, file: 'a.png', fileName: 'portrait.png', mediaType: 'image' });
            if (!{{(lastOnly ? "true" : "false")}}) {
              window.TensorAgent.addAttachment({ ok: true, file: 'b.png', mediaType: 'image' });
              window.TensorAgent.addAttachment({ ok: true, file: 'c.png', mediaType: 'image' });
            }
            var box = __page.byId['chips'];
            var select = box.querySelector('.video-frame-position');
            select.value = {{(lastOnly ? "'-1'" : "'custom'")}};
            select.dispatch('change');
            if (!{{(lastOnly ? "true" : "false")}}) {
              var number = box.querySelector('.video-frame-number'); number.value = '5'; number.dispatch('change');
            }
            var strength = box.querySelector('.video-frame-strength'); strength.value = '.65'; strength.dispatch('change');
            var masks = box.querySelectorAll('.mask-select').length;
            var placeholder = __page.byId['text'].placeholder;
            var reads = __page.requests('/api/models').length;
            __page.byId['text'].value = '请让图中的女人来个飞吻';
            __page.byId['send'].dispatch('click');
            return settle(30).then(function () { return {
              sent: __page.requests('/api/chat').map(function (call) { return call.body; }),
              reads: __page.requests('/api/models').length - reads,
              masks: masks, placeholder: placeholder, notices: __page.notices()
            }; });
            """);
        Assert.False(result.TryGetProperty("error", out JsonElement error), error.ToString());
        Assert.Equal("Describe a video… attach a video to continue from its last frame, or a photo for frame guidance", result.GetProperty("placeholder").GetString());
        Assert.Equal(0, result.GetProperty("masks").GetInt32());
        Assert.Equal(0, result.GetProperty("reads").GetInt32());
        Assert.Empty(Strings(result, "notices"));
        JsonElement message = Assert.Single(result.GetProperty("sent").EnumerateArray()).GetProperty("messages")[0];
        var conditions = message.GetProperty("imageConditions").EnumerateArray().ToArray();
        Assert.Equal(lastOnly ? new[] { -1 } : new[] { 5, 8, 16 }, conditions.Select(c => c.GetProperty("frameIndex").GetInt32()));
        Assert.Equal(.65, conditions[0].GetProperty("strength").GetDouble());
        Assert.Equal("a.png", conditions[0].GetProperty("imagePath").GetString());
        JsonElement attachment = message.GetProperty("attachments")[0];
        Assert.Equal(lastOnly ? -1 : 5, attachment.GetProperty("videoFrameIndex").GetInt32());
        Assert.Equal(.65, attachment.GetProperty("videoStrength").GetDouble());
    }
}
