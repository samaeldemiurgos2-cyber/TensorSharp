# Video Studio architecture and reference audit

## Scope and ownership

Video Studio extends WebUX Chat and TensorAgent's existing WebView/loopback host.
`TensorSharp.Studio` owns the project, assets, validated commands, history,
proposals, jobs, persistence and composition. `TensorSharp.Chat/Studio` adapts the
existing `ModelService` and agent runtime. Both hosts serve the same
`TensorSharp.Chat/WebUi/studio.js` and `studio.css`. There is no Electron, Node,
Python, ComfyUI, remote inference service or separately downloaded UI runtime in
the product path.

The host's Studio service is the authoritative writer. The page displays returned
snapshots and submits commands with explicit project IDs and revisions. Media is
outside the JSON document. Asset insertion and generation are independent: undo
removes an insertion, never the generated source and never reruns inference.
Explicit library deletion uses the shared `deleteAsset` command with an asset
ID and optional `removeTimelineClips` confirmation. Assets still used by clips
are refused with `asset_in_use` unless that confirmation is true. The command
removes the library entry and only clips using that asset as one revision-checked,
idempotent, undoable transaction. Linked clips using other assets remain intact.
Deletion retains managed media for undo and captured jobs; it does not delete
the imported original or reclaim disk space. Removed assets are no longer served
through the current project's asset endpoint until restored with undo.
Project edits use integral frames at a rational project frame rate; source
positions use the same clock and rendering samples source presentation timestamps.
Seconds round to the nearest frame, with midpoint ties away from zero. Variable
frame rate input is sampled onto the output clock rather than assuming each
source frame has equal duration.

Preview is an actual composition render from a captured revision, played in the
browser media element. Export uses the same renderer. Editing invalidates the
preview visibly; source playback is separate. This deliberately favors agreement
over pretending that a browser approximation supports an effect the renderer
cannot reproduce. Live composited preview is a future optimization.

Cuts, per-clip fades and explicitly applied dissolves use that same composition
path. A dissolve requires an existing partial overlap between two clips on one
video track. It changes no placement: incoming alpha increases over the opaque
outgoing picture, while complementary linear audio fades keep the gain envelope
consistent. Aligned linked audio companions receive the same fades. Triple
overlap and incompatible fades are rejected. Splits inside an active fade are
also rejected; splitting outside fades preserves their existing envelopes.

## Baseline audit

Inspected the working tree at `main`,
`6bdc5c7b799fdc63e061b8dbd3ec9a64bb46cf48`; implementation branch:
`codex/video-studio`. Existing uncommitted TensorAgent localization, web-link and
MAUI navigation work was present and is unrelated to Studio.

| Area | Existing reusable implementation | Gap addressed / remaining boundary |
|---|---|---|
| Shared chat | `TensorSharp.Chat/WebUiChatService.cs`, `ChatGenerationPipeline.cs`, `Skills/SkillChatLoop.cs` | Explicit Studio context and tools need the shared command service; media bytes/history must not enter planner context. |
| Native generation | `Models/Video/IVideoGenerationModel.cs`, `VideoGenerationParams.cs`, `VideoGenerationProgress.cs`; Wan, MiniMax-H3, LTX 2.5 implementations | Existing API permits model-specific defaults/ignored parameters; Studio admission must reject unsupported requests and record effective settings. |
| Host generation | `TensorAgent.Core/Hosting/VideoTurns.cs`; server video endpoints and model service | Existing chat results are artifacts, not timeline projects. Studio jobs retain results independently of the HTTP connection. |
| Model lifetime | Shared `ModelService`; TensorAgent `AgentAppHost`, `EngineMemoryPolicy`, catalog/store | Reuse existing loaded model and serialized access; do not maintain another model pool. A video diffusion model cannot plan conversational edits. |
| Media | `Models/Media/MediaCodecs`, desktop/Apple providers, `WanVideo/VideoIO.cs` | Existing generated video encoding is not a multitrack compositor. Desktop composition uses a separately installed FFmpeg. Apple mobile subprocess export is unavailable. |
| Uploads | `Hosting/UploadPathGuard.cs`, `UploadContentPolicy.cs`, `UploadStoragePolicy.cs`, host uploads/media routes | Restrict artifact import to known local upload references; managed project assets must survive upload cleanup. |
| Agent persistence | TensorAgent sessions, conversations, image/video turns; AgentHost tools/permissions | Timeline proposals, revision conflicts, atomic undo and durable render jobs are new. No arbitrary shell edits to project documents. |
| Frontend | Server `wwwroot/index.html`; TensorAgent MAUI `wwwroot/index.html`; embedded shared mask editor | Shared Studio assets fit both existing pages. MAUI framework and platform bridge remain in place. |

## Inspected references

Checked on 2026-10-15 (America/Los_Angeles). Clones used for inspection are ignored
under `artifacts/studio/references/`; no reference source is shipped.

| Repository | Branch | Commit | Repository license |
|---|---|---|---|
| [LTX Desktop](https://github.com/Lightricks/LTX-Desktop) | `main` (`v1.3.0`) | `30ed2e7bc3c117c9f2fce59d7d9dbded6c7e65ca` | Apache-2.0, `LICENSE.txt`; separate `NOTICES.md` |
| [OpenChatCut](https://github.com/0xsline/OpenChatCut) | `main` (`v0.2.17`) | `72ddb3300cda4ed31cbd0bccbfc94fb29b8cffca` | AGPL-3.0, `LICENSE` |
| [TensorSharp](https://github.com/zhongkaifu/TensorSharp) | local `main` baseline | `6bdc5c7b799fdc63e061b8dbd3ec9a64bb46cf48` | BSD-3-Clause, root `LICENSE`; existing component notices still apply |

The [shared discussion](https://chatgpt.com/share/6ad19241-62b8-83e8-a3a5-9fed4d3c84fd)
returned a title but no readable discussion body. It was **not read**; the user's
supplied discussion context is the basis for requirements.

### Reference-to-implementation mapping

These are architectural observations from source and tests, not copied algorithms
or implementation code. Reference tests were inspected, not executed.

| Reference source path | Useful principle observed | TensorSharp implementation direction | Verification |
|---|---|---|---|
| LTX `frontend/hooks/use-generation.ts`, `backend/_routes/generation.py` | Capture generation identity, progress, cancellation and recovery separately from the active panel | Host-independent durable Studio jobs; native model adapter | Disconnect polling, cancel, restart, generated asset/provenance checks |
| LTX `frontend/views/editor/useRegeneration.ts`, `frontend/contexts/ProjectContext.tsx` | Keep takes and conditioning settings; resolve the selected clip's take | Generated assets keep requests; explicit replacement preserves clip position and original asset | Replace/undo tests; real generation metadata |
| LTX `frontend/lib/project-storage.ts`, `frontend/types/project-model.ts` | Validate and migrate persistent projects; timeline and asset library belong to one project | Versioned host-owned project document and atomic saves | Round-trip/migration/future-version rejection/restart |
| LTX `backend/tests/test_keyframe_request.py`, `test_generate_video_request.py` | Reject incompatible conditioning combinations before expensive generation | Capability-driven Studio request validation | Invalid frame grid, reference bounds, unsupported mode tests |
| LTX `frontend/views/editor/usePlaybackEngine.ts`, `usePlaybackAudioSync.ts` | Playback clock and media synchronization are distinct from UI refresh cadence | Browser transport over a rendered revision, with frame-aware timeline cursor | Decode/frame/audio checks against captured project |
| LTX `electron/export/export-handler.ts`, `timeline.ts`, `audio-mix.ts` | Visual ordering, audio composition and output validation need explicit semantics | TensorSharp-owned FFmpeg plan; track order, trim, fit, fades, volume, captions | Deterministic colored clips and tones, full output decode |
| OpenChatCut `src/editor/storeCommands.ts`, `storeCommandBuilder.ts`, `reduce.ts` | Humans and agents need the same bounded editing operations and atomic history | Studio command batches and persistent undo/redo | Atomic rollback, linked edits, one-step agent undo |
| OpenChatCut `src/editor/proposal.verify.ts`, `src/agent/proposal.ts` | Proposals edit a draft; stale bases must be rejected; generated assets survive rejection | Validated proposals with revision checks; separate asset ingestion | Concurrent manual edit invalidates pending proposal |
| OpenChatCut `src/agent/tools/project-tools.ts`, `src/agent/external-tool-policy.ts` | Project context must be explicit and tools separated by authority | Existing TensorSharp agent tools bound to project/revision | Tool isolation and planner/video-model separation |
| OpenChatCut `server/plugins/project-store-durable.ts`, `server/project-store-http-auth.ts` | Atomic persistence and authenticated transport are separate responsibilities | Studio atomic document store plus host auth/origin checks | Interrupted writes, path traversal and cross-origin requests |
| OpenChatCut `remotion/render-contract.mjs`, `src/captions/exportCaptions.verify.ts` | A render must resolve its own captured media/caption state | Immutable Studio render revision; same preview/export composition | Render while project changes; captions in decoded output |
| OpenChatCut `src/transcript/edit.ts`, `projection.verify.ts`, `transcribe-jobs.ts` | Transcript words require stable source timing and projection through cuts | Do not expose automatic transcription until a compatible local adapter is connected; manual captions use timeline frames | No nonfunctional transcription button; caption timing tests |

## Dependency and licensing decisions

TensorSharp's license is unchanged. No LTX Desktop or OpenChatCut implementation,
UI assets, models or fonts are copied, translated or adapted. Reading a reference
and writing C# is not described as a clean-room process. OpenChatCut's AGPL code
is not incorporated. LTX Desktop's Apache application license does not license its
model weights or dependencies.

The new core uses the .NET base class library and the existing repository's test
packages. The UI uses existing browser primitives and system fonts; no new UI
library, web font or stock media is downloaded. Exported captions use a small
TensorSharp-owned bitmap glyph renderer with an explicit uppercase Basic Latin
character boundary. Test media is generated locally
from solid colors and sine waves. Remotion is not introduced: its
[license](https://github.com/remotion-dev/remotion/blob/main/LICENSE.md) includes
commercial licensing conditions, and it would add a separate rendering runtime.

FFmpeg is an **operator-provided executable**, not bundled or downloaded by
Studio. Its [official license notes](https://ffmpeg.org/legal.html) explain that
optional GPL components change the build's license. Validation on this machine
uses Homebrew FFmpeg 9.0.2, configured with `--enable-gpl --enable-version3` and
libx264. That fact does not establish redistribution clearance for another build.
Packaging FFmpeg, codec libraries or fonts requires a separate distribution
review including build flags, corresponding sources/notices and applicable
codec patent obligations. Studio must report encoder availability and never
silently substitute an unavailable codec.

Model weights are separately installed through existing paths. Wan checkpoints,
MiniMax-H3, LTX 2.5, Gemma text encoders and optional LoRA adapters retain their
individual publisher terms. TensorAgent's existing catalog records these
requirements; Studio neither accepts terms nor downloads weights for the user.
LTX 2.5 catalog companions are pinned to publisher revision
`49b093a39c19169a8a81721dcc541e07e945232f`. Existing local weights may be used for
validation; no model files are added to this change.

## Security and operational boundaries

Projects belong to one host workspace and are shared among its admitted clients.
There is no per-session project ACL or multi-tenant collaboration service. An
exclusive storage lease prevents two active Studio service instances from
writing the same root. TensorAgent retains its existing loopback session-token
authentication. The standalone server has no built-in authentication: Studio
requires both a loopback connection peer and a loopback/localhost Host header by
default, including Studio-bound chat requests. This also rejects DNS rebinding
through an arbitrary Host header. `TS_STUDIO_ALLOW_REMOTE=1` is an explicit
operator opt-in only for a deployment protected by an external authentication
and authorization boundary. It does not authenticate clients, isolate users,
upload media or configure a remote provider.

Studio adds same-origin/custom-header checks for writes and bound chat requests.
Native/API callers may omit Origin; this is not proof of authentication. IDs are validated before paths
are constructed; media routes serve only project-managed assets/results.
Imported text and metadata are data. Rendering uses an argument list, never a
model-generated shell command. Package import must reject unsafe paths and bound
entry counts and expanded bytes. No remote URL import, automatic upload,
publishing, telemetry or paid provider is added.

There is one bounded generation/render queue per Studio service. Queued jobs can
cancel without starting; cancellation of running work remains pending until
inference/subprocess work has returned. On startup, a generated asset committed
before its terminal job record is reconciled through the durable request ID.
Other unfinished persisted jobs become interrupted and can be retried explicitly;
no mid-denoising resume is claimed. Model execution uses existing host resource
ownership.

Studio generation admission also requires enabled subprocesses, usable ffprobe
and an FFmpeg H.264/AAC encoder before entering that queue. Capability responses
retain each model's family/settings metadata while marking it unavailable with
an explanation when the output pipeline is absent. This avoids running native
inference on a mobile or incompletely configured desktop host only to fail while
ingesting the result. Imported-media editing and project-package review remain
available according to the independent host capabilities.

Generation requests carry an explicit mode (`t2v`, `i2v`, `fl2v`, `keyframes` or
`ref`). The mode defines what each image means. `StudioGenerationModes` checks that
a request supplies exactly that mode's inputs. First and last frames are each
optional: `fl2v` needs a last frame and takes a first frame too when one is given,
while a first frame alone is `i2v`. This follows MiniMax-H3's FL2VA contract (zero,
one or two keyframes; sglang's `fl2va` signatures `[0]`, `[-1]` and `[0, -1]`), and
nothing stands in for a missing frame. The page derives the mode from two optional
slots and the reference photos beside them (none → `t2v`, first → `i2v`, last or
both → `fl2v`, any reference photo → `ref`). It runs for every client: the page,
saved assistant plans and the HTTP API. Each capability derives its `modes` from
the checkpoint's conditioning flags. The native adapter adds `unavailableModes`,
which explain the family modes the loaded file or host cannot run. Admission
refuses those modes with that same reason. The mode names are the native mode
names, so `MiniMaxH3ModeResolver` and `Ltx25GenerationSettings` stay the final
authority on each family's combination rules. A request saved without a mode is
inferred from its inputs. Its serialized form omits the field, so idempotency
fingerprints of earlier jobs do not change.

`ref` may also carry a first frame (`conditioningAssetId`), to keep a character
while a chain continues from the previous segment's last frame. It never takes a
last frame or temporal guides. Few checkpoints run the combination, so it is a
capability flag rather than another mode: `referencesWithFirstFrame`, with
`referencesWithFirstFrameUnavailableReason` when it is false. Admission
(`RequireReferencesWithFirstFrame`) and the native adapter's validation refuse
ref plus a first frame on a checkpoint without the flag as
`unsupported_conditioning`, with that reason, before the mode check, and never
drop either input; the reason names what the family can do instead. No installed
family sets the flag. MiniMax-H3 FL2VA pins frames but takes no references,
Ref2VA takes references but pins no frames, and MiniMax's pipeline,
stable-diffusion.cpp and vllm-omni refuse keyframes with references in one
request; Wan has no identity references. LTX 2.5's pipeline does compose a frame-0
image guide with the Ingredients sheet, equivalent to upstream `ICLoraPipeline`
called with `images` and `video_conditioning` together and checked bit for bit
against the unchanged upstream functions; for a ref request with a first frame the
adapter would send that frame as a frame-0 `VideoImageCondition` at strength 1, not
as `ImagePath`, which LTX's ref mode reads as an identity image. LTX still leaves
the flag false: Ingredients was trained without a pinned first frame, and in
validation every such take cut to the reference sheet at its second frame (see the
[LTX 2.5 card](models/ltx-2.5.md#starting-on-a-first-frame)). The flag, the page's
first-frame slot in the reference view and the adapter mapping are ready for a
checkpoint that does both. When the host pins no output size,
LTX's capability also names `referenceDefaultWidth`/`referenceDefaultHeight`
(768 × 448, the Ingredients trained bucket), which the page offers as a size.

A capability may also list `imageCanvasModes`, with the recipe that sizes them
(`imageCanvasShortEdge`, `imageCanvasMaxPixels`, `imageCanvasMaxAspect`). In
those modes a request with width and height both 0 takes its canvas from the
first-frame image, or the last-frame image when there is no first: admission resolves the mode first, leaves 0×0 unfilled, and
the native adapter replaces it with the size it reads from the decoded, upright
image before the job is queued, so the job and the take record a concrete size.
The fingerprint is still computed on 0×0, so a retry matches. MiniMax-H3 FL2VA
lists i2v and fl2v, with MiniMax's 768p recipe, only on a host that pins no
video size and follows the model recipe; those modes keep the first frame's
framing, and at 640×384 a wide shot's faces do not survive the video VAE.

A capability also carries `samplingRecipe` (default `native`) and
`samplingRecipeName`. A step-distillation plug-in, today MiniMax-H3 FL2VA with a
LightX2V Turbo plug-in, sets them from the model's `VideoSamplingContract` and fixes
the capability's steps and guidance to the contract. Admission refuses a request
that names another recipe (`sampling_recipe_conflict`, which `StudioHostBoundary`
maps to 409) and a step count the recipe does not run, then binds
`request.SamplingRecipe ??= capability.SamplingRecipe` after validation. The binding
comes after the idempotency fingerprint, and the field is omitted from JSON when
null, so earlier requests and their fingerprints are unchanged. The native adapter
passes the bound recipe to `WebUiChatService.GenerateVideoAsync`, which applies the
clip's plug-ins and re-checks the recipe under the model's GPU lock before the clip
runs. In TensorAgent the adapter takes a plug-in provider: capability and validation
queries plan the clip's plug-ins without logging, and the run
applies the same saved choice the chat path uses. A host `--video-steps` that
contradicts the recipe, or a plug-in on the Ref2VA checkpoint, leaves the model
unavailable as a backstop; the server already refuses the former at startup.

Media probing, thumbnails and waveforms are asynchronous, cancellable and bounded,
but are not durable jobs. The default timeline is limited to 86,400 frames and
renders to 120 combined clips/captions; imports/packages have a 2 GiB limit and
256 MiB disk reserve. Forty undo transactions are retained. These limits bound
individual work; they are not a general multi-user quota system. See the user
guide and validation report for exact implemented and unverified features.

## Frame stills and video chains

A long video is made as a chain of takes, each starting on the frame the previous
segment ends on. `StudioFrames.cs` materializes that frame as an ordinary image
asset. Its `derivedFrom` records the source: the video's `assetId`, its `name` and
`sha256` when the frame was taken, `frame` (-1 for the last frame, else N on the
project clock), `frameRate` (only for N, the rate that counts it) and `method`
(`generation` or `decode`). The generation request then names a photo, as every
other first frame does. Putting a video ID and frame in the request instead would
have changed the request shape and with it the idempotency fingerprint, the job
snapshot, package contents and every model adapter, each of which would then need
to decode video. A still changes none of them: adapters receive an image path,
the job snapshot captures the still, and a package carries it as an image with
its record. Because the record keeps the name and hash, a still keeps its meaning
after its video is renamed or deleted. Package metadata is untrusted, so
`StudioCommands.Validate` refuses a record Studio could not have written: one on
a non-image or generated asset, an out-of-range frame, a frame rate present for
the last frame or missing for N, an unknown method, or a malformed name or hash.

Host action `assets/extract-frame` takes `{projectId, assetId, frame: "last" | N}`
and returns `{project, assetId, reused}`. Stills are deduplicated by source and
frame (source asset ID and hash, frame and frame rate, with the file present),
never by content hash. Two frames with equal pixels stay two stills, and an
ordinary import of byte-identical content never stands in for one, since it
would lose the record. A repeated request, such as a double tap or a second page,
returns the existing still with `reused: true`, even on a host that cannot
decode. A new still enters the library as an import does: the project moves to a
new revision without an undo step, and the still can be deleted like any asset.
Photos and audio are refused with `invalid_frame_source`. Without external
processes the action is refused with `unsupported_platform`: taking a frame needs
FFmpeg, which is off where Studio media subprocesses are (iOS and Mac Catalyst).

Frames are decoded one at a time, apart from thumbnails and waveforms, with
Studio's protocol and format allow-lists and a timeout scaled to the video's
length (2 to 30 minutes). Pixels are copied, never remapped. PQ/HLG transfers
and interlaced field orders are refused with `unsupported_frame_source` rather
than tone-mapped or deinterlaced. Non-square sample aspect ratios are widened to
square pixels, because the decoders that read the still ignore aspect metadata.
The PNG is upright RGB24 at full size. The last frame is the last frame the
decoder presents, with edit lists honored. Decoding starts 2 s before the video
stream's own end (a longer soundtrack moves the container's end past the
picture), then 2 s before the container's end, then at the start, and each
decoded frame overwrites the output. Frame N is sampled exactly as the timeline
renderer samples the source at the project rate: an unseeked `fps` grid from 0
with nearest rounding, then that one frame. It is therefore the frame that a clip
ending on N shows.

A generated take's last frame needs no decode. `WebUiChatService.GenerateVideoAsync`
takes an optional `lastFramePath` and writes the clip's final frame there as a
lossless PNG of the pixels the MP4 was encoded from. The native adapter asks for
`last-frame.png` in the job's work directory and returns it as
`StudioGenerationResult.LastFramePath`. The work directory is deleted when the
job ends, so before the take is imported the service checks that the file is a
PNG inside that directory, not a reparse point, and that it probes as an image of
the request's exact size. It then moves the file to `jobs/<id>/last-frame.png`
and records `LastFrameFileName` in the persisted job record; a frame of another
size is not kept. When a restart reconciles a take that was committed before its
job record finished, the file is re-linked as well. The last frame of such a take
comes from this file
(`method: generation`), so a chain does not go through an H.264 round trip at
each join. Frame N, imported videos and takes whose job record is absent are
decoded (`method: decode`); packages omit job history.

The thumbnail route `GET /api/studio/thumbnail/{projectId}/{assetId}?frame=last`
on both hosts shows a video's last frame through the same decode, cached as a
separate derivative. A photo is its own last frame, and any other `frame` value
is refused with `invalid_request`. The page uses it for the first-frame picker's
video tiles.

A shot plan continues a video by reference. A generation draft may store
`firstFrameFrom` (`{assetId, frame}`) and no `conditioningAssetId`. Saving checks
that the video belongs to the project and resolves the mode as if the frame were
the first frame. The still is taken or reused when the plan starts, or when the
page opens the plan in Generate, so planning never changes the library. The
planner's asset summaries add `derivedFrom` and a bounded take summary (mode,
first-frame asset, reference count, frame count, fps, shot) without returning
whole requests. `ChainPart` names a take " · part N" by following first frames
back through stills and the takes they came from; an imported video is part 1.

Placing a segment is a page action over existing commands. **Add after** sends
one `insertClip` right after the clip the take continues, on the same track. Its
`sourceStartFrame` is max(1, round(project fps / take fps)), because the take's
first frame repeats the previous clip's last frame, and its `durationFrames` is
the rest of the take.
