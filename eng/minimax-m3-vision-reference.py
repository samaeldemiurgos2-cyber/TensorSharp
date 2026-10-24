#!/usr/bin/env python3
# ruff: noqa: E501
"""
MiniMax-M3.1 vision oracle: HuggingFace transformers (minimax_m3_vl, >= 5.20) image
processor + vision tower + projector, run on synthetic test images, dumped as raw
little-endian F32 files for InferenceWeb.Tests/MiniMaxM3VisionOracleTests.cs.

Only the vision weights are read: transformers' own MiniMaxM3VLVisionModel and
MiniMaxM3VLMultiModalProjector are built from the checkpoint config and filled from
the 523 safetensors tensors with the vision_tower. / multi_modal_projector. /
patch_merge_mlp. prefixes (shards 13 and 48 of the full checkpoint; a directory that
holds the config, the filtered index and those two shards is enough). The text model
is never constructed.

Per image it writes, under OUT_DIR:
    images/<name>.png                 the synthetic input (RGB, lossless)
    <name>.pixels.f32                 HF pixel_values, temporal slice 0: [P, 3*14*14],
                                      merge-block patch order (HF duplicates the slice;
                                      the script asserts both copies are equal)
    <name>.tower.f32                  vision tower output (F32 run) [P, 1280]
    <name>.emb_f32.f32                projector output, weights BF16 -> F32 compute [N, 6144]
    <name>.emb_bf16.f32               projector output, BF16 compute (HF's checkpoint dtype)
    manifest.json                     sizes, grids, token counts, timings, HF f32-vs-bf16 gap

Usage (VM):
    /root/convvenv/bin/python eng/minimax-m3-vision-reference.py OUT_DIR \
        --hf-dir /workspace/models/MiniMax-M3.1-Flash-hf-vision --device cuda:0
Then:
    TS_TEST_MINIMAX_M3_MMPROJ=<mmproj-MiniMax-M3.1-Flash-BF16.gguf> \
    TS_TEST_MINIMAX_M3_VISION_ORACLE=OUT_DIR TS_TEST_GGML_BACKEND=cuda \
    dotnet test InferenceWeb.Tests --filter MiniMaxM3VisionOracleTests
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from safetensors import safe_open

from transformers.models.minimax_m3_vl.configuration_minimax_m3_vl import MiniMaxM3VLConfig
from transformers.models.minimax_m3_vl.image_processing_minimax_m3_vl import MiniMaxM3VLImageProcessor
from transformers.models.minimax_m3_vl.modeling_minimax_m3_vl import (
    MiniMaxM3VLMultiModalProjector,
    MiniMaxM3VLVisionModel,
)

# name, width, height, kind
IMAGES = [
    ("scene_1024x768", 1024, 768, "scene"),
    ("shapes_640x480", 640, 480, "shapes"),
    ("text_256x256", 256, 256, "text"),
    ("scene_1920x1080", 1920, 1080, "scene"),
    ("tiny_56x56", 56, 56, "shapes"),
    ("square_672x672", 672, 672, "scene"),
    ("round_350x350", 350, 350, "text"),        # Python round(12.5) = 12: 336x336, not 364
    ("tall_300x900", 300, 900, "shapes"),
    ("noise_500x333", 500, 333, "noise"),
]


def font(size):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                 "/usr/share/fonts/dejavu/DejaVuSans.ttf",
                 "/System/Library/Fonts/Supplemental/Arial.ttf"):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def make_image(width, height, kind, seed):
    rng = np.random.RandomState(seed)
    if kind == "noise":
        return Image.fromarray(rng.randint(0, 256, size=(height, width, 3), dtype=np.uint8), "RGB")
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    base = np.stack([255 * xx / max(width - 1, 1),
                     255 * yy / max(height - 1, 1),
                     127.5 + 127.5 * np.sin((xx + yy) / max(width, height) * 6.283)], axis=-1)
    if kind == "text":
        base = np.full((height, width, 3), 245.0, dtype=np.float32)
    img = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), "RGB")
    draw = ImageDraw.Draw(img)
    unit = max(4, min(width, height) // 8)
    for i in range(6 if kind != "text" else 2):
        x0, y0 = rng.randint(0, max(1, width - unit)), rng.randint(0, max(1, height - unit))
        x1, y1 = x0 + rng.randint(unit // 2, unit * 3), y0 + rng.randint(unit // 2, unit * 3)
        color = tuple(int(c) for c in rng.randint(0, 256, size=3))
        if i % 3 == 0:
            draw.ellipse([x0, y0, x1, y1], fill=color, outline=(0, 0, 0))
        elif i % 3 == 1:
            draw.rectangle([x0, y0, x1, y1], fill=color)
        else:
            draw.line([x0, y0, x1, y1], fill=color, width=max(1, unit // 6))
    lines = ["MiniMax M3.1", "vision oracle", f"{width}x{height}"]
    f = font(max(8, min(width, height) // 10))
    for i, line in enumerate(lines):
        draw.text((max(2, width // 20), max(2, height // 20) + i * (min(width, height) // 6)), line,
                  fill=(10, 10, 10), font=f)
    return img


def load_vision_state(hf_dir):
    index = json.load(open(os.path.join(hf_dir, "model.safetensors.index.json")))["weight_map"]
    tower, proj = {}, {}
    shards = {}
    for name, shard in index.items():
        if name.startswith("vision_tower.vision_model."):
            key = name[len("vision_tower.vision_model."):]
            key = key.replace("embeddings.patch_embedding.", "embeddings.proj.").replace("encoder.layers.", "layers.")
            shards.setdefault(shard, []).append((name, tower, key))
        elif name.startswith("multi_modal_projector."):
            shards.setdefault(shard, []).append((name, proj, name[len("multi_modal_projector."):]))
        elif name.startswith("patch_merge_mlp."):
            key = name[len("patch_merge_mlp."):].replace("linear_", "merge_linear_")
            shards.setdefault(shard, []).append((name, proj, key))
    for shard, items in shards.items():
        with safe_open(os.path.join(hf_dir, shard), framework="pt") as f:
            for name, dst, key in items:
                dst[key] = f.get_tensor(name)
    return tower, proj


def cos_rel(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    num = (a * b).sum(-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-30
    cos = num / den
    rel = np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30)
    return float(cos.min()), float(cos.mean()), float(rel)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir")
    ap.add_argument("--hf-dir", default="/workspace/models/MiniMax-M3.1-Flash-hf-vision")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "eager"])
    ap.add_argument("--only", default=None, help="comma-separated image names")
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out_dir, "images"), exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    cfg = MiniMaxM3VLConfig.from_pretrained(args.hf_dir)
    vcfg = cfg.vision_config
    vcfg._attn_implementation = args.attn
    t0 = time.time()
    tower_sd, proj_sd = load_vision_state(args.hf_dir)
    print(f"loaded {len(tower_sd)} tower + {len(proj_sd)} projector tensors in {time.time() - t0:.1f}s", flush=True)
    dtypes = {str(v.dtype) for v in list(tower_sd.values()) + list(proj_sd.values())}
    print("checkpoint dtypes:", dtypes)

    with torch.device(args.device):
        tower = MiniMaxM3VLVisionModel(vcfg).float().eval()
        projector = MiniMaxM3VLMultiModalProjector(cfg).float().eval()
    missing, unexpected = tower.load_state_dict(tower_sd, strict=False)
    missing = [k for k in missing if "inv_freq" not in k]
    if missing or unexpected:
        sys.exit(f"tower state mismatch: missing {missing[:5]} unexpected {unexpected[:5]}")
    projector.load_state_dict(proj_sd, strict=True)
    # Module.to(bf16) would also round the rotary inv_freq buffer; a checkpoint loaded
    # in bf16 keeps it F32 (it is built with dtype=torch.float), so restore it per cast.
    inv_freq = tower.rotary_emb.inv_freq.detach().clone().float()

    processor = MiniMaxM3VLImageProcessor()
    only = set(args.only.split(",")) if args.only else None

    entries = []
    for seed, (name, width, height, kind) in enumerate(IMAGES):
        if only and name not in only:
            continue
        img = make_image(width, height, kind, seed + 1)
        png = os.path.join("images", name + ".png")
        img.save(os.path.join(args.out_dir, png))
        img = Image.open(os.path.join(args.out_dir, png)).convert("RGB")

        feats = processor(images=[img], return_tensors="pt")
        pv = feats["pixel_values"]
        t, gh, gw = (int(x) for x in feats["image_grid_thw"][0])
        assert t == 1, t
        p = gh * gw
        pv5 = pv.reshape(p, 3, 2, 14, 14)
        assert torch.equal(pv5[:, :, 0], pv5[:, :, 1]), "temporal copies differ"
        pixels = pv5[:, :, 0].reshape(p, 3 * 14 * 14).float().numpy()
        pixels.tofile(os.path.join(args.out_dir, name + ".pixels.f32"))

        thw = feats["image_grid_thw"].to(args.device)
        entry = {"name": name, "png": png, "width": width, "height": height, "grid_h": gh, "grid_w": gw,
                 "resized_h": gh * 14, "resized_w": gw * 14, "patches": p, "tokens": p // 4,
                 "pixels": name + ".pixels.f32", "tower": name + ".tower.f32"}
        outs = {}
        for tag, dtype in (("f32", torch.float32), ("bf16", torch.bfloat16)):
            tower.to(dtype)
            projector.to(dtype)
            tower.rotary_emb.inv_freq = inv_freq.clone()
            x = pv.to(args.device, dtype)
            with torch.no_grad():
                for _ in range(2 if tag == "f32" and not entries else 1):   # first call warms CUDA
                    torch.cuda.synchronize() if args.device.startswith("cuda") else None
                    t1 = time.time()
                    hidden = tower(pixel_values=x, grid_thw=thw).last_hidden_state.squeeze(0)
                    emb = projector(hidden)
                    torch.cuda.synchronize() if args.device.startswith("cuda") else None
                    ms = (time.time() - t1) * 1000
            emb_np = emb.float().cpu().numpy()
            assert emb_np.shape == (p // 4, cfg.text_config.hidden_size), emb_np.shape
            emb_np.tofile(os.path.join(args.out_dir, f"{name}.emb_{tag}.f32"))
            if tag == "f32":
                hidden.float().cpu().numpy().tofile(os.path.join(args.out_dir, entry["tower"]))
            entry[f"emb_{tag}"] = f"{name}.emb_{tag}.f32"
            entry[f"hf_{tag}_ms"] = round(ms, 1)
            outs[tag] = emb_np
        mn, mean, rel = cos_rel(outs["bf16"], outs["f32"])
        entry["hf_bf16_vs_f32"] = {"min_cos": mn, "mean_cos": mean, "rel_l2": rel}
        entries.append(entry)
        print(f"{name}: {width}x{height} -> {gh * 14}x{gw * 14}, {p} patches, {p // 4} tokens; "
              f"HF f32 {entry['hf_f32_ms']} ms, bf16 {entry['hf_bf16_ms']} ms; "
              f"bf16 vs f32 min cos {mn:.6f} mean {mean:.6f} rel-L2 {rel:.4e}", flush=True)
        tower.float()
        projector.float()

    manifest = {
        "transformers": __import__("transformers").__version__,
        "torch": torch.__version__,
        "torchvision": __import__("torchvision").__version__,
        "attn": args.attn,
        "device": args.device,
        "patch_dim": 3 * 14 * 14,
        "tower_dim": vcfg.hidden_size,
        "embedding_dim": cfg.text_config.hidden_size,
        "images": entries,
    }
    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print("wrote", os.path.join(args.out_dir, "manifest.json"))


if __name__ == "__main__":
    main()
