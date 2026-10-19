#!/usr/bin/env python3
"""Make a two-segment chain through a real Studio host and measure the join.

Segment 1 is generated (i2v from --image, or t2v without one). Its last frame is
taken into the library with assets/extract-frame, and segment 2 starts on that
still: i2v, or with --character a ref take that keeps that character while
starting on the frame (only on a model whose capability says
referencesWithFirstFrame; elsewhere the request must be refused before a job
exists, and segment 2 then starts on the frame alone). Both segments go on the timeline, the
second right after the first and one frame in (its first frame repeats the
first segment's last), and the timeline is rendered.

Reported, not asserted beyond the structural checks:
  * still_vs_take_last_psnr: the still against the take's decoded last frame
    (a "generation" still is the take's lossless last frame; expect ~40 dB+)
  * seg2_first_vs_still_psnr / seg2_first_vs_seg1_first_psnr: continuity of
    segment 2's first frame with the frame it started on, against an unrelated
    frame of the same scene
  * seam_psnr: the two frames either side of the join in the rendered timeline,
    and the median frame-to-frame PSNR inside each segment for comparison

Requires Python 3 stdlib, ffmpeg/ffprobe and a host with one video model loaded.
Evidence belongs below ignored artifacts/ or docs/validation/. Example:

  python3 eng/studio/validate_chain.py --url http://127.0.0.1:8191 --image imgs/banner_1.png \\
      --width 640 --height 384 --frames 22 --output artifacts/studio-chain/h3

On a checkpoint that keeps references but pins no frames (MiniMax-H3 Ref2VA), --segment1 imports an existing video
as segment 1 and --frame-as-reference sends the continued frame as reference Picture 1 beside the character
(MiniMax's prompt guide: "<Picture N> is the first frame of [Shot 1]"); the take is guided to start like the frame,
not pinned to it, and the continuity numbers measure how closely it does.
"""
import argparse
import json
import pathlib
import re
import statistics
import subprocess

from validate import Host, run, uid, verify_media


def frame_png(video, index, path):
    """Decode frame `index` of a video (every frame the decoder shows, in order) to a PNG."""
    run("ffmpeg", "-v", "error", "-y", "-i", video, "-map", "0:V:0", "-fps_mode", "passthrough",
        "-vf", f"select=eq(n\\,{index})", "-frames:v", 1, path)
    return path


def frame_count(video):
    out = run("ffprobe", "-v", "error", "-count_frames", "-select_streams", "V:0", "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", video)
    return int(out.strip())


def psnr(a, b):
    """Average PSNR (dB) of two same-size images; inf when identical."""
    log = subprocess.run(["ffmpeg", "-v", "info", "-i", str(a), "-i", str(b), "-lavfi", "[0][1]psnr", "-f", "null", "-"],
                         check=True, capture_output=True).stderr.decode()
    match = re.search(r"average:(inf|[0-9.]+)", log)
    assert match, log[-800:]
    return float("inf") if match.group(1) == "inf" else float(match.group(1))


def adjacent_psnrs(video, first, last, work):
    values = []
    for i in range(first, last):
        a = frame_png(video, i, work / f"adj-{i}.png")
        b = frame_png(video, i + 1, work / f"adj-{i + 1}.png")
        values.append(psnr(a, b))
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://127.0.0.1:8187")
    parser.add_argument("--connection", type=pathlib.Path, help="TensorAgentHost connection.json (token is never reported)")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--image", type=pathlib.Path, help="First frame of segment 1 (i2v); omitted: segment 1 is t2v")
    parser.add_argument("--segment1", type=pathlib.Path, help="Import this video as segment 1 instead of generating it")
    parser.add_argument("--frame-as-reference", action="store_true", help="Send the continued frame as reference Picture 1 (ref, nothing pinned)")
    parser.add_argument("--character", type=pathlib.Path, help="Reference photo segment 2 keeps (ref with a first frame)")
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--frames", type=int, help="Frames per segment (default: the model's)")
    parser.add_argument("--frames2", type=int, help="Frames of segment 2 (default: --frames)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt1", default="The woman sits on a glowing ribbon of light in space, smiling, her hair drifting as she slowly turns her head toward the camera.")
    parser.add_argument("--prompt2", default="She keeps smiling, lifts her hand from her knee and waves gently at the camera while the glowing cubes float around her.")
    parser.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    work = args.output / "frames"; work.mkdir(exist_ok=True)
    connection = json.loads(args.connection.read_text()) if args.connection else {}
    host = Host(connection.get("baseUrl", args.url), connection.get("cookie"))

    caps = host.api("capabilities")
    models = [m for m in caps["models"] if m["available"]]
    assert models, "No available video model: " + json.dumps(caps.get("generationUnavailableReason"))
    model = models[0]
    report = {"model": model["modelId"], "family": model["family"], "modes": model["modes"],
              "referencesWithFirstFrame": model.get("referencesWithFirstFrame", False),
              "referencesWithFirstFrameUnavailableReason": model.get("referencesWithFirstFrameUnavailableReason"),
              "width": args.width, "height": args.height}
    project = host.api("projects/create", {"name": "Chain validation " + uid()[:6], "width": args.width, "height": args.height})
    pid = project["id"]
    fps = project["frameRate"]["numerator"] / project["frameRate"]["denominator"]

    def generate(request, label):
        request = {"projectId": pid, "requestId": uid(), "modelId": model["modelId"], "seed": args.seed, "fps": 24,
                   "width": args.width, "height": args.height, "temporalConditions": [], "referenceAssetIds": [], **request}
        if args.frames:
            request.setdefault("frameCount", args.frames)
        job = host.api("jobs/generate", request)
        job = host.poll(pid, job["id"], timeout=args.timeout)
        take = next(a for a in host.api("projects/get", {"projectId": pid})["assets"] if a["id"] == job["outputAssetId"])
        path = args.output / f"{label}.mp4"
        status, body, _ = host.request(f"/api/studio/projects/{pid}/assets/{take['id']}")
        assert status == 200, status
        path.write_bytes(body)
        verify_media(path)
        timing = job.get("timing") or {}
        report[label] = {"assetId": take["id"], "name": take["name"], "frames": frame_count(path), "elapsedSeconds": timing.get("elapsedSeconds"),
                         "mode": take["generation"]["effectiveParameters"].get("mode")}
        return take, path

    # Segment 1.
    if args.segment1:
        seg1 = host.upload(pid, args.segment1)["assets"][-1]
        seg1_path = args.output / "segment1.mp4"; seg1_path.write_bytes(args.segment1.read_bytes())
        report["segment1"] = {"assetId": seg1["id"], "name": seg1["name"], "frames": frame_count(seg1_path), "imported": True}
    else:
        first_id = None
        if args.image:
            first_id = host.upload(pid, args.image)["assets"][-1]["id"]
        seg1, seg1_path = generate({"mode": "i2v" if first_id else "t2v", "prompt": args.prompt1, "conditioningAssetId": first_id}, "segment1")

    # Its last frame, taken into the library.
    taken = host.api("assets/extract-frame", {"projectId": pid, "assetId": seg1["id"], "frame": "last"})
    still = next(a for a in taken["project"]["assets"] if a["id"] == taken["assetId"])
    still_path = work / "still.png"
    status, body, _ = host.request(f"/api/studio/projects/{pid}/assets/{still['id']}")
    assert status == 200; still_path.write_bytes(body)
    seg1_last = frame_png(seg1_path, report["segment1"]["frames"] - 1, work / "seg1-last.png")
    again = host.api("assets/extract-frame", {"projectId": pid, "assetId": seg1["id"], "frame": "last"})
    assert again["reused"] and again["assetId"] == still["id"], "taking the frame again must reuse the still"
    report["still"] = {"name": still["name"], "method": still["derivedFrom"]["method"], "size": [still["width"], still["height"]],
                       "still_vs_take_last_psnr": psnr(still_path, seg1_last)}

    # Segment 2 starts on the still; with a character it keeps that character too, where the model can.
    seg2_request = {"mode": "i2v", "prompt": args.prompt2, "conditioningAssetId": still["id"], "width": still["width"], "height": still["height"]}
    if args.frames2:
        seg2_request["frameCount"] = args.frames2
    if args.character:
        character_id = host.upload(pid, args.character)["assets"][-1]["id"]
        seg2_request.update({"mode": "ref", "referenceAssetIds": [character_id]})
        if args.frame_as_reference:
            seg2_request.update({"conditioningAssetId": None, "referenceAssetIds": [still["id"], character_id]})
        elif not model.get("referencesWithFirstFrame"):
            refused = host.api("jobs/generate", {"projectId": pid, "requestId": uid(), "modelId": model["modelId"], "fps": 24,
                                                 "temporalConditions": [], **seg2_request}, status=400)
            assert refused.get("code") == "unsupported_conditioning", refused
            jobs = host.api("jobs/list", {"projectId": pid})
            assert len(jobs if isinstance(jobs, list) else jobs["jobs"]) == 1, "a refused chain segment must not queue a job"
            report["characterRefused"] = refused.get("error")
            # The chain itself still runs: segment 2 starts on the frame alone.
            seg2_request.update({"mode": "i2v", "referenceAssetIds": []})
    seg2, seg2_path = generate(seg2_request, "segment2")
    seg2_first = frame_png(seg2_path, 0, work / "seg2-first.png")
    seg1_first = frame_png(seg1_path, 0, work / "seg1-first.png")
    report["continuity"] = {"seg2_first_vs_still_psnr": psnr(seg2_first, still_path),
                            "seg2_first_vs_seg1_last_psnr": psnr(seg2_first, seg1_last),
                            "seg2_first_vs_seg1_first_psnr": psnr(seg2_first, seg1_first)}

    # The timeline: segment 2 right after segment 1, one frame in (its first frame repeats segment 1's last).
    project = host.api("projects/get", {"projectId": pid})
    track = next(t for t in project["tracks"] if t["kind"] == "video")["id"]
    n1, n2 = report["segment1"]["frames"], report["segment2"]["frames"]
    skip = max(1, round(fps / 24))
    project = host.api("commands/apply", {"projectId": pid, "expectedRevision": project["revision"], "commandId": uid(), "commands": [
        {"type": "insertClip", "clipId": uid(), "assetId": seg1["id"], "trackId": track, "startFrame": 0, "durationFrames": n1, "sourceStartFrame": 0},
        {"type": "insertClip", "clipId": uid(), "assetId": seg2["id"], "trackId": track, "startFrame": n1, "durationFrames": n2 - skip, "sourceStartFrame": skip}]})
    render = host.api("jobs/render", {"projectId": pid, "expectedRevision": project["revision"], "requestId": uid()})
    render = host.poll(pid, render["id"], timeout=args.timeout)
    render_path = args.output / "chain.mp4"
    status, body, _ = host.request(f"/api/studio/projects/{pid}/jobs/{render['id']}/output")
    assert status == 200; render_path.write_bytes(body)
    rendered = frame_count(render_path)
    assert rendered == n1 + n2 - skip, (rendered, n1, n2, skip)
    before = frame_png(render_path, n1 - 1, work / "seam-before.png")
    after = frame_png(render_path, n1, work / "seam-after.png")
    inside1 = adjacent_psnrs(render_path, max(0, n1 - 6), n1 - 1, work)
    inside2 = adjacent_psnrs(render_path, n1, min(rendered - 1, n1 + 5), work)
    report["timeline"] = {"frames": rendered, "skip": skip, "seam_psnr": psnr(before, after),
                          "segment1_tail_adjacent_psnr_median": statistics.median(inside1),
                          "segment2_head_adjacent_psnr_median": statistics.median(inside2)}
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
