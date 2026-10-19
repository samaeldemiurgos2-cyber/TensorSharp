# Video Studio

Video Studio is a shared editor in **WebUX Chat** and **TensorAgent**. Open
**Video Studio** from the chat interface/menu, or **Open in Studio** beside a
video served by the current host. Media stays on that host. No cloud account or
AI model is required for manual editing.

## Start locally

Use the existing .NET 10 development setup. Desktop media import requires
operator-installed `ffprobe`. Studio generation and rendering additionally require
`ffmpeg` with an available H.264 encoder and AAC. These output capabilities are
checked before native generation can enter the queue. Studio does not download
these tools or install models. Review the
[dependency notes](video-studio-design.md#dependency-and-licensing-decisions)
before redistributing a media-tool build.

The TensorAgent Mac Catalyst app supports the same local media pipeline as the
desktop server. Finder launches also discover installed tools in
`/opt/homebrew/bin` and `/usr/local/bin`; `TS_FFMPEG` and `TS_FFPROBE` can select
explicit executables. Studio verifies a small H.264/AAC MP4 before enabling
generation and export. On Mac desktop it prefers VideoToolbox and falls back to
libx264 when the installed hardware encoder cannot start. Successful checks are
cached; failed checks can recover after the tools become available.

```sh
dotnet build TensorSharp.Server.Host/TensorSharp.Server.Host.csproj
TS_STUDIO_DIRECTORY="$PWD/artifacts/my-studio" \
  dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll \
  --host 127.0.0.1 --port 8187
```

Open `http://127.0.0.1:8187`. `TS_STUDIO_DIRECTORY` selects server project/job
storage; otherwise Studio uses its directory beside the server output. Do not
point two active hosts at the same storage directory. Use project packages to
move work between hosts.

Studio server access is local by default, even if other server APIs listen on a
network interface. Both the connection peer and Host header must be loopback.
For a deliberately remote deployment, protect the whole interface/API with an
authenticated proxy first, then set `TS_STUDIO_ALLOW_REMOTE=1` on startup. The
standalone server does not provide authentication or per-user project isolation;
that opt-in only removes Studio's loopback restriction. TensorAgent keeps its
existing local session-token protection.

TensorAgent includes the same editor in its existing MAUI interface. Its
in-process `AgentAppHost` stores Studio projects under its private data root,
in `studio-projects`. For desktop development without launching MAUI, the
existing real loopback-host harness runs that application core:

```sh
dotnet run --project eng/validation/TensorAgentHost -- \
  --root artifacts/studio-agent --skills TensorAgent/skills --port 8188
```

Use the authenticated `entryUrl` in the harness's local `connection.json`.
That file contains a session credential: do not publish it or add it to Git.
The harness exercises the real application core and shared web page; it is not
a physical iOS/Windows-device test.

## Edit a project

1. Open Studio, create a project, and set its dimensions and rational frame
   rate. Import video, still images or audio, or open an existing chat upload.
2. Add assets to video/audio tracks. Library assets remain available when their
   timeline clips are removed. A clip's own audio moves with its picture;
   explicitly linked clips share move/trim/split operations.
3. Select clips in the timeline or shot strip. Drag to move or use the inspector
   for exact frame positions, source in-points and lengths. Split at the cursor,
   duplicate, remove, change fit, adjust volume/fades, and mute, hide or remove
   tracks.
   Snapping and zoom help with longer sequences; exact controls work without
   dragging and on touch layouts. Split outside active fades, or clear a fade
   first; splitting inside its envelope is refused to preserve its appearance.
   For a dissolve, first overlap two clips on the same video track, with the
   incoming clip starting and ending later. Apply the dissolve to that pair.
   It changes fades without moving either clip: the incoming picture fades in
   over the outgoing picture, while their audio gains crossfade. A third clip
   overlapping that interval or conflicting existing fades must be resolved first.
4. Add/edit timed captions. The initial portable caption renderer supports
   Basic Latin text and renders it uppercase; unsupported characters are refused
   for export. Automatic transcription and word-based cuts are not exposed.
5. Render a preview. This is a playable composition of a **captured project
   revision**, using the same rendering path as export. An edit marks an earlier
   preview stale. Source preview is separately labeled and does not claim to
   show the complete timeline.
6. Export the rendered MP4. Export is a background job; keep editing while it
   runs. The result belongs to the captured revision, not later changes.

Each successful command autosaves atomically. Undo/redo survives reopening.
An atomic multi-command proposal is one undo step. Import and generation results
survive undo. A failed transaction leaves the previous document intact.

### Remove a track

In either host, click **×** among the track's buttons in the timeline header.
The confirmation shows how many clips the track holds; confirm to remove the
track together with those clips, or choose **Cancel** (or press Escape) to keep
it. Clips on other tracks stay, including clips linked to the removed ones, and
every asset stays in the library. Use **Undo** to restore the track in its
place with its clips; **Redo** removes it again. The Studio assistant can
propose the same edit, stating how many clips it removes; it takes effect only
when you apply the proposal.

### Delete an asset

In either host, open the **Asset library** and click **Delete asset** on the
asset's card. Confirm the dialog, or choose **Cancel** to keep it. If the asset
is used on the timeline, the warning shows how many clips use it. Confirming
removes the library entry and those clips together; other assets and their
clips remain in place. Use **Undo** to restore the asset and its removed clips
in one step; **Redo** removes them again. These edits autosave.

Deletion removes the asset from the current project. Studio retains its managed
media file for undo and renders already captured from earlier revisions, so this
action does not reclaim disk space. Your original imported file is preserved.

Thumbnails are bounded 320×180 derivatives. Waveforms contain 512 peak buckets,
not an entire soundtrack in browser memory. Codec playback still depends on the
browser/WebView; export errors report missing encoders rather than quietly
substituting another output format. Probing and derivative creation are bounded,
asynchronous and cancellable; unlike generation/export they are not durable
queued jobs. Completed thumbnails and waveforms are cached.

Default limits are 32 tracks, 2,000 clips/assets, 5,000 captions and 86,400 timeline
frames (one hour at 24 fps). A render supports at most 120 combined clips and
captions. Media imports and expanded project packages are limited to 2 GiB;
there is a 256 MiB free-space reserve. The most recent 40 edit transactions are
retained for undo. Host options can lower or adjust the configurable limits.

## Generate and refine

Load a native video model using the application's existing model controls (or
the server's existing `--model ... --backend ...` startup arguments). Studio
queries the loaded model. Wan, MiniMax-H3 and LTX 2.5 keep their own frame grids,
sampling recipes and conditioning rules. Unsupported options fail before a job
starts. No hidden model or remote-provider substitution occurs. A loaded model
is marked unavailable in Studio when this host cannot probe, validate and render
its output. Installing a model alone therefore does not enable generation on a
host without the required local media tools or subprocess support. Project
review and manual edits remain available.

Studio re-reads what the loaded model can make by itself: when the Generate or
Assistant tab opens, when you come back to Studio (window focus or the page
becoming visible), when TensorAgent reports that the model or its plug-ins
changed, and every few seconds while Generate waits for a video model. A host
that cannot generate at all (no media subprocesses, as in the TensorAgent app)
says so once and is not polled.

### Make a take

The **Generate** tab reads top to bottom:

1. **Video model.** The loaded model's family, what it can do, and its file
   name on a muted line. With no video model loaded the tab says so, and keeps
   your description and photos until you load one.
2. **Describe this shot.** The placeholder changes with the photos you chose.
3. **Start or end on a photo (optional).** Two slots, **First frame** and
   **Last frame**. Leave both empty to make the take from the description
   alone. Fill a slot in any of these ways:
   - tap it and choose a photo from the library's thumbnails, or **Import a
     photo…** (PNG, JPEG or WebP) to add one and use it at once;
   - drag a library photo onto it (desktop);
   - choose **As first frame** or **As last frame** on a library photo. This
     opens Generate with the photo in place;
   - for the first frame only, continue a video: choose it under **Or continue
     a video from its last frame** in the picker, or drop it on the slot
     (desktop). See [Make a long video as a chain](#make-a-long-video-as-a-chain).

   A filled slot shows the photo cropped to the take's shape, as the model will
   crop it, with **Change** and **Remove**. A frame taken from a video says
   which ("Last frame of “shot.mp4”"). **Swap first and last frames**
   exchanges them. A line under the slots says what the take will do ("The take
   ends on the last frame."). A slot the loaded model cannot use stays visible
   and says why (Wan, for example, cannot pin a last frame).
4. **Keep a character the same (optional).** Reference photos of who or what
   should look the same, each with **Remove**. **+ Add reference photo** adds
   one, as does **As character reference** on a library photo. With reference
   photos the take is a `ref` take: a new scene with that character. Starting on
   a first frame as well needs a model that keeps references with one, which no
   installed model does; the section says why for the loaded one. A last frame
   cannot join reference photos. See
   [Keep a character across segments](#keep-a-character-across-segments).
5. **Other ways to use photos** (only when the model offers more than one):
   **Hit photos at set moments** (timed keyframes) and **Keep someone or
   something the same** (identity references, the same list of photos as
   step 4). Photos chosen for one use are kept when you switch to another, but
   only the active use's photos are sent.
6. **Length** (in seconds, only the lengths the model's frame grid makes),
   **Size**, **Generate audio**, and under **Advanced settings** the negative
   prompt (when the model's guidance runs one), steps, guidance, seed and frame
   rate.
7. **Reset to model defaults.** It re-reads the loaded model and its plug-ins
   from the host, then puts length, size, frame rate, steps, guidance and sound
   back to that model's defaults; a chain segment keeps the size of the video it
   continues. Your description, photos and seed stay. This
   one button replaces the earlier **Refresh model capabilities** and **Use
   model defaults** buttons: the first only re-read the host and redrew the
   form, the second only reset the fields from the values the form already
   held, so it could apply stale defaults.
8. **Generate take**, with a one-line summary above it ("Ends on beach.jpg ·
   0.9 s (22 frames) · 768 × 1152 · with sound"). A problem shows next to the
   button in plain words; the host's own message is under **Details**. A refusal
   about the model (it changed, or a plug-in did) also re-reads the model.

The form is kept per project, so editing the timeline, a finished job or a model
refresh never clears it. After you generate, Jobs shows the progress. The finished
take appears in the library named after its prompt ("Waves roll in at sunrise.mp4",
then "… · take 2.mp4" for the next take of the same shot, and "… · part 2.mp4"
for a segment that continues a video), marked, and announced.

### Generation modes

The mode says what the photos *mean*. Studio derives it from the slots and
reference photos you fill and sends only the photos that mode uses. The host refuses a request whose
images do not match its mode, so an image is never quietly reinterpreted.

| You fill | Mode sent | What happens |
|---|---|---|
| Nothing | **t2v** | The description alone makes the shot. |
| First frame | **i2v** | The photo *is* the first frame; the description drives the motion. |
| Last frame only | **fl2v** | The clip ends on the photo; the description says how it gets there. |
| First and last frame | **fl2v** | The clip starts on one photo and ends on the other. |
| Timed keyframes | **keyframes** | Each photo guides a chosen output frame at a chosen strength; first and last frames stay optional. |
| Reference photos | **ref** | Identity/appearance references for a new scene. They are not frames; the first frame need not resemble them. |
| Reference photos and a first frame | **ref** | The take starts on the first frame and keeps who or what the references show. Only a model that keeps references with a first frame runs it. A last frame or timed keyframes never join references. |

First and last frames are each optional, as on MiniMax-H3's FL2VA checkpoint:
its model card accepts zero, one or two images (one image is the first *or* the
last frame), and its prompt guide calls the last-only case L2VA. The serving
stacks MiniMax points to agree: sglang's `fl2va` accepts the frame signatures
`[0]`, `[-1]` and `[0, -1]`, vllm-omni takes `frame_indices` `[-1]`, diffusers
takes `image` and/or `last_image`, and stable-diffusion.cpp accepts `--end-img`
alone. None of them, and not Studio, puts a blank image where a frame is
missing: the frame is simply not pinned. With only a last frame, MiniMax-H3
places that one keyframe block at the clip's end and sizes the canvas from it;
LTX 2.5 appends it as an image guide at the final frame. A last frame alone,
or with a first, is `fl2v`; a first frame alone stays `i2v`.

Use **i2v** to animate a photo and **ref** to put the person or product from a
photo somewhere new. **ref** may also carry a first frame on a checkpoint that
keeps references while starting on one (none installed here; see
[Keep a character across segments](#keep-a-character-across-segments)). Studio keeps references in the order you select them and
shows that order. MiniMax-H3 names them `<Picture 1>`, `<Picture 2>`… in the
prompt, and LTX lays them out in that order.

What each loaded checkpoint runs:

| Model | Text only | First frame | Last frame only | First + last | Keyframes | References | References + first frame |
|---|---|---|---|---|---|---|---|
| MiniMax-H3 FL2VA (`minimax_h3_fl2va_*`) | Yes | Yes | Yes | Yes | No | Needs the Ref2VA file | No |
| MiniMax-H3 Ref2VA (`minimax_h3_ref2va_*`) | Yes | Needs the FL2VA file | Needs the FL2VA file | Needs the FL2VA file | No | Yes, up to 9 images | No |
| LTX 2.5 Distilled | Yes | Yes | Yes | Yes | Yes, up to 16 images | With the Ingredients IC-LoRA: up to 16 images, at least 121 frames | No (the pipeline composes it, but validation cut to the reference photos) |
| Wan2.2 TI2V-5B | Yes | Yes | No | No | No | No | No |
| Wan2.2 I2V-A14B | No (a first frame is required) | Yes | No | No | No | No | No |
| Wan 2.1 / Wan2.2 T2V | Yes | No | No | No | No | No | No |

The slots and uses come from the loaded checkpoint. One it cannot use stays
visible with the reason underneath. MiniMax-H3's FL2VA and Ref2VA are separate
files rather than settings, so load the other file through the model controls;
a Ref2VA checkpoint opens Generate on reference photos. LTX finds the
Ingredients adapter at `loras/ltx-2.5-22b-ic-lora-ingredients-0.9.safetensors`
beside the checkpoint, or through `TS_LTX25_INGREDIENTS_LORA`. LTX's image modes
also need the desktop FFmpeg/libx264 image transform. Wan never reads a last
frame: through the server, the CLI and the API, an end image on Wan is refused
instead of being dropped.

Recipes depend on the mode. LTX's ref mode, with or without a first frame, is
one full-size pass rather than the two-stage recipe, and choosing references
raises the length to 121 frames when it is lower, saying so. Studio references
and frames are still images. A frame taken from a video is one too: Studio puts
it in the library as a still that records the video and frame it came from.
MiniMax-H3 reference video clips and soundtracks remain available through chat,
the CLI and the HTTP API.

Takes made before modes were explicit reopen in the mode their inputs imply and
run unchanged. Shot plans saved by the Studio assistant may name a `mode`. Studio
checks that the mode matches the plan's images when the plan is saved, and checks
it against the loaded checkpoint when you start the plan.

New generations inherit the host's chat video defaults, including the configured
output dimensions, frame rate and clip length, resolved to the loaded model's
supported frame grid. With `config/ltx-2.5-distilled.json`, the default output is
1024×1024. `--video-frames 144` becomes 145 frames at 24 fps (about six seconds),
as in chat. LTX's first sampling pass runs at half the output size: lowering the
output to 512×512 also lowers that pass to 256×256 and can distort faces and
hands. Higher resolution requires more memory and generation time.

MiniMax-H3 keeps a keyframe's framing, so with a first or last frame its
**Size** defaults to **Match the photo (sharpest faces)**: the canvas takes the
first frame's aspect, or the last frame's when it is the only one, at MiniMax's
768-pixel short edge (1152×768 for a 3:2 photo, 768×1152 for a 2:3 portrait).
The form shows that size and its cost before you generate, and the request sends
0×0, so the host reads the image upright and records the size it used in the
take. At 640×384 a wide shot's faces are only a few latent cells across and come
out blurred and distorted however many steps run; the recipe keeps them, at about
2.4× (square) to 4.2× (16:9) the video tokens per frame. Choose **Custom** for a
smaller, faster preview. A host started with both `--video-width` and
`--video-height` keeps that size for every mode, as chat does, and the TensorAgent
app keeps 640×384 until the recipe's memory is measured on its 32 GB tier. A
second keyframe whose shape differs from the canvas is cropped to it; the slot
says when edges are cut off.

**More → Make another version** on a generated take reopens Generate with that
take's description, photos, seed and, on the same model, its size, length and
sampling. **Reset to model defaults** then applies the current model's defaults
while keeping its description and photos. Existing footage is not regenerated
automatically. Studio and chat use the same native generation and encoding path;
compare them with the same image, prompt, seed and settings.

A successful job adds a managed asset with prompt, effective settings, seed,
model identity and output metadata. **Add to timeline**, or open **More →
Replace the selected clip with this** to swap a selected clip's take while
keeping its placement. Original assets stay in the library for comparison and
undo; a replacement must fit the requested source range.

### Make a long video as a chain

A take is short, so a longer video is made as a chain of takes: each segment
starts on the frame the one before it ends on, and the segments play one after
another on the timeline. Each of these opens Generate with that frame as the
**First frame**:

- **Continue from last frame** on a video's library card.
- **Continue after this clip** in **Properties** for a selected video clip. The
  take starts on the frame the clip ends on: its out point when the clip is
  trimmed, or the video's last frame when the clip reaches the end.
- **Continue from this take** on a finished generation in **Jobs**.
- In the first-frame picker, a video under **Or continue a video from its last
  frame** (its tile shows that frame), or a video dropped on the first slot.

Your description and reference photos stay, so a character carries on, and a
last frame chosen for the previous shot is cleared. Describe what happens next.
The frame becomes a still in the library ("shot · last frame.png" or "shot ·
frame 48.png"), whose card and slot say which video and frame it came from.
Taking the same frame again reuses that still. Like an import, adding it is not
an undo step; delete it as any other asset.

For a take generated on this host, the last frame is the take's own last frame,
saved losslessly before its video was encoded, so the next segment is
conditioned on exactly those pixels rather than on a decoded copy. Its own first
frame is the model's reconstruction of them: close, not bit-identical. Any other
frame is decoded from the video file: an imported video, a take from a project
package (packages omit job history), or a trimmed clip's out point. The slot
then adds "from the encoded video". A frame number is counted on the project
clock, and the frame is the one the timeline shows there. Decoding refuses HDR
(PQ or HLG) and interlaced video rather than tone-mapping or deinterlacing it;
export an SDR or progressive version and continue from that. Non-square pixels
are widened to square, as a player shows them.

A segment is made at the size of the video it continues when the loaded model
can make that size ("Size matches the video this take continues"), so the
segments join without a crop or a jump. Otherwise the hint says the frame is
fitted to the take's size. The size is set once, when the frame is placed; you
can change it afterwards, and **Reset to model defaults** keeps it. Sound is
made per segment: it does not carry on across the join. For one continuous
soundtrack, put it on an audio track under the segments.

A segment's take is named after its prompt with " · part N". Continuing an
imported video gives part 2, and continuing a part 2 take gives part 3.

When the take is ready, **Add after “shot.mp4”** on its Jobs card or library
card puts it right after the clip it continues, on the same track. The button
appears when a clip of that video ending on that frame is on a video track. The
new clip starts one frame in (its **Source in**), because the take's first frame
repeats the last frame of the clip before it, which would otherwise show twice.
When the project frame rate is higher than the take's, it skips the matching
number of project frames: two at 48 fps for a 24 fps take. The status line says
how many, and warns when the new clip overlaps another clip on the track. It is
an ordinary undoable edit. **Add take to timeline** still adds the whole take.

Taking a frame needs the desktop media decoder (FFmpeg), as generation does.
Where Studio runs without media subprocesses (iOS, and the TensorAgent app on
Mac Catalyst), the continue actions are unavailable with generation, and a frame
request is refused with that reason. The assistant can also
[plan a segment](#the-assistant) that continues a video.

### Keep a character across segments

A first frame carries a character into a segment, but the look can drift over
many segments. Reference photos under **Keep a character the same (optional)**
ask the model to keep that person or object the same. They are listed in the
order sent to the model (MiniMax-H3 calls them Picture 1, 2, …). Reference photos
cannot be combined with a last frame.

With no first frame, the photos make a new shot that keeps the character
(MiniMax-H3 Ref2VA, or LTX 2.5 with the Ingredients IC-LoRA): a fresh framing, so
the join to the previous segment is a cut. Starting on the previous segment's
frame **and** keeping reference photos needs a checkpoint that does both in one
take (`referencesWithFirstFrame`). None of the installed families does, and the
section says why for the loaded one instead of dropping either input:

- **MiniMax-H3.** FL2VA pins frames but takes no reference photos; Ref2VA keeps a
  character but pins no frames. MiniMax's own pipeline, stable-diffusion.cpp and
  vllm-omni refuse keyframes and references in one request. On FL2VA, continue
  from the frame alone: it already shows the character. On Ref2VA, MiniMax's
  prompt guide lets a reference photo describe "the first frame of [Shot 1]". In
  validation that kept the character but framed a new shot rather than continuing
  the frame.
- **LTX 2.5.** Its pipeline composes a frame-0 image guide with the Ingredients
  sheet as the publisher's IC-LoRA pipeline does
  ([LTX 2.5 card](models/ltx-2.5.md#starting-on-a-first-frame)), but the
  publisher trained Ingredients without a pinned first frame, and in validation
  every such take cut from the frame to the reference photos at its second frame.
  Studio therefore does not offer it.
- **Wan** has no identity references. Its image-to-video checkpoints (TI2V-5B,
  I2V-A14B) continue from the frame alone.

A model that does advertise `referencesWithFirstFrame` gets the same section
enabled, and its **Keep someone or something the same** view shows an optional
first-frame slot. Ingredients is trained at 768 × 448 and 121 frames: choosing
references raises a shorter length to 121 frames, saying so, and when the size
differs the section offers **Use 768 × 448** (unless the host pins one size for
every mode).

### The assistant

The **Assistant** tab is a chat agent scoped to this project. It runs on the
loaded **chat** model, not the video model, through the host's normal chat
pipeline, with a set of Studio-only tools: `studio_inspect` (tracks, clips,
assets, captions, the selection, revision and frame rate), `studio_assets`,
`studio_jobs`, `studio_propose`, `studio_generate`, `studio_render` and
`studio_import_chat`. It only sees the project through those tools and up to
eight recent messages.

What it does with them:

- **Edits you review.** `studio_propose` turns a plain request into one
  validated proposal of timeline commands. The tab lists the changes in words
  (the exact commands are under **Inspect exact commands**); **Apply proposal**
  applies them as one undo step. If you change the project first, the proposal
  can no longer apply as is, and **Ask again** re-sends your request against the
  current project.
- **Shot plans.** `studio_generate` saves a plan (prompt, mode, photos, size,
  length); it never makes footage. Open a plan in Generate to review it, or run
  it with a loaded video model.
- **Chain segments.** A plan can continue a video: `continueFromAssetId` names
  the video, and the optional `continueFromFrame` names `last` (the default) or
  a frame number on the project clock. The plan stores this as `firstFrameFrom`
  and its card reads "Continues “shot.mp4”". The frame is taken when you start
  the plan or choose **Open in Generate**, so planning never changes the
  library. `studio_inspect` and `studio_assets` show which video and frame a
  still came from (`derivedFrom`) and a short summary of each take (mode, first
  frame, reference count, length and shot), so the assistant can follow a chain.
- **Exports and imports.** `studio_render` starts a local preview/export job and
  `studio_import_chat` imports a chat upload into the library. Neither waits for
  an Apply step.

For example, with Gemma-4-E4B (Q8_0) loaded as the chat model, a 4-second clip
on the timeline and selected, the request "Trim the selected clip to its first 2
seconds and lower its volume to 25 percent. Keep its position." produced a proposal
listed as "Trim “assistant-clip.mp4” to 2 s" and "Change “assistant-clip.mp4”:
volume 25%" (`trimClip` to 48 frames plus `updateClip` with `volume` 0.25).
**Apply proposal** made the clip 2.00 s long as one undoable edit. The lines are
computed from the commands themselves, not from the model's summary, so read them
before applying: a second run of the same request summarized "first 2 seconds"
while its command trimmed to 1 s, and the list showed "to 1 s". The example chips
("Fade out the last clip over 1 second", "Add the title “Day one” over the first 3
seconds", "Plan a 5-second opening shot…") fill the box without sending.

The existing host has **one active model slot**, so the assistant and Generate
are not available at the same time: while a video model is loaded, the tab says
so and **Send** is disabled. Conversational editing needs a text model that
supports structured tools; video models are rejected as planners. Model switching
is not automatic. The project remains editable throughout; generated footage is
added through the same command layer as imported media.

### Sampling recipes

A step-distillation plug-in fixes how the model samples. With MiniMax-H3 FL2VA and a
[LightX2V Turbo plug-in](models/minimax-h3.md#turbo-plug-ins), started on the server
with `--lora config/lora/minimax-h3-fl2v-turbo-8step.json` or switched on in
TensorAgent's LoRA sheet, the model's capability names its sampling recipe (for
example `minimax-h3-lora:minimax-h3-fl2v-turbo-8step`) and its display name gains
" · LightX2V Turbo 8-step (FL2V v1.0, 768p)". The form shows the recipe outside
Advanced, and the step count is fixed at the recipe's 8 (or 4); a request with another
count is refused with "This model samples exactly 8 steps (LightX2V Turbo 8-step (FL2V
v1.0, 768p)); leave steps 0 for its default." A model without a plug-in has the
recipe `native`.

Admission binds each job to the recipe the model has when the job is accepted. The
binding happens after the idempotency fingerprint, so jobs saved before recipes keep
their fingerprints. When the loaded model's recipe has changed since the shot was
planned (for example, the plug-in was switched or removed), the
request is refused with **409** `sampling_recipe_conflict` and no job is created:

> This shot was planned for minimax-h3-lora:minimax-h3-fl2v-turbo-8step but the loaded
> model now samples with its own schedule. Refresh the model settings and review the steps.

The host checks the bound recipe again under the model's GPU lock, just before the
clip runs, so a change while the job waited fails that job instead of sampling it on
another schedule. Reopening a take made under another recipe shows a note, and the
steps start from the current model's default rather than the take's; takes from before
recipes count as `native`. A take records the plug-in file, its strength and SHA-256 in
its model identity, and the recipe, both flow shifts and the sigma grid in its
effective settings.

A server started with a `--video-steps` that contradicts the plug-in does not start:
the refusal is printed before anything downloads. As a backstop, Studio also marks a
model unavailable, with the reason, when the host's step default contradicts its
recipe, and when a plug-in is set on the Ref2VA checkpoint.

The server's Studio uses the plug-ins the server started with. In TensorAgent, Studio
follows the app's saved plug-in choice: the plug-in the LoRA sheet has on for the
loaded model applies to the next clip in chat and in Studio alike, and changing it
between planning and running a shot gives the 409 above.

## Portable projects and recovery

Export a `.tsstudio` project package to continue on another installation. Packages
contain project settings, tracks, clips, captions, asset metadata/provenance and
relative media files. They omit conversation history, host paths, credentials,
undo history and job history. Import creates a new project ID. Hash verification,
path restrictions and compressed/expanded size limits apply before publishing
the imported project.

Missing files are reported, not silently discarded. Relinking accepts the same
original bytes, verified by SHA-256, so existing source ranges and captured
renders retain their meaning. A new version of footage should be imported as a
new take. Schema 1 is the first Studio format; unknown future schemas are refused
without rewriting them. There is no historic Studio format to migrate yet.

The **Jobs** tab shows what each job is doing. Its tab label counts the
unfinished jobs. A generation card names the model, mode, canvas, frame count,
steps and seed, plus the start of the prompt. While a job is waiting it shows how
many jobs are ahead and how long it has been queued. While it runs it shows the
current stage (encoding the prompt, denoising, decoding video frames, decoding
audio, encoding the MP4, adding the take to the library), the denoising steps done
and left, the start time, the elapsed time, and the model's latest detail, such as
a VAE tile or a Wan pass, with the time spent in that stage. An estimate of the
time left appears once a denoising step has finished. Without an earlier run, the
estimate covers only the denoising steps and says that decoding follows. After a
generation with the same model, canvas, frames, steps, CFG, mode, audio and
sampling recipe has completed, the estimate also covers the stages around
denoising, and a queued job shows how long that earlier run took. Finished jobs
show their total time and a per-stage breakdown. Jobs that stop early name the
stage they stopped in and the steps they reached. The host computes elapsed and
remaining durations, so a client clock that disagrees with the host does not
skew them.

Generation/export jobs survive panel closure and browser disconnection. Cancelling
a queued job stops it before execution. A running job first becomes **cancelling**
and only becomes **cancelled** after its worker/process cleanup finishes. On
restart, a generation result already committed to the asset library is reconciled
with its job as completed. Other unfinished jobs become **interrupted**. An
explicit retry uses a new request identifier and restarts the operation;
mid-denoising checkpoint resume is not available. Failed jobs retain their error. Free disk space, correct missing media/model
settings, then retry explicitly. The queue is bounded; Studio does not silently
lower quality after out-of-memory or disk errors.

Mobile can review and edit imported project packages. Desktop subprocess-based
probing, thumbnails, waveforms, frame stills and export are unavailable on
iOS/Android. Studio
generation is also disabled there because the required output validation and
rendering pipeline is unavailable; a loaded native model does not bypass this
gate. No desktop-equivalent mobile validation is claimed. There is no automatic remote
fallback. An explicitly configured remote host has its own storage and compute
boundary.

## Reproduce validation

```sh
dotnet test TensorSharp.Studio.Tests/TensorSharp.Studio.Tests.csproj
node --test eng/validation/tests/studio-ui.test.mjs

# Against the server command above; uses real synthetic media and actual FFmpeg.
python3 eng/studio/validate.py --url http://127.0.0.1:8187 \
  --output artifacts/studio-check

# Against the real TensorAgent loopback harness.
python3 eng/studio/validate.py --connection artifacts/studio-agent/connection.json \
  --output artifacts/studio-agent-check

# With a real video model already loaded; no test double or download is used.
python3 eng/studio/validate.py --url http://127.0.0.1:8187 \
  --generate --output artifacts/studio-native-check

# Against an actual Mac Catalyst app launched with TENSORAGENT_VALIDATION_ROOT,
# with the pinned LTX catalog assets in its isolated Cache/models directory.
# Exercises native model selection, Studio import, real 1024x1024 joint video/audio
# generation, thumbnails/waveforms, timeline insertion and fully decoded export.
python3 eng/studio/validate_maui.py \
  --connection artifacts/studio-mac-app/connection.json \
  --app TensorAgent/src/TensorAgent.Maui/bin/Release/net10.0-maccatalyst/maccatalyst-arm64/TensorAgent.app \
  --output artifacts/studio-mac-check
```

After stopping/restarting the same host against the same data root, pass
`--reopen <projectId>` to the validation script. It reopens the document and
renders/decodes it again. Generated evidence belongs in ignored `artifacts/` or
`docs/validation/`, never force-added to Git. See
[the validation record](video-studio-validation.md) for actual environments,
results and remaining limitations.
