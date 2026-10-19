#!/usr/bin/env python3
"""Generate IC-LoRA reference-video plus frame-0 image-guide fixtures with an unchanged official LTX checkout.

Same tiny geometry as generate-ltx25-reference-conditioning-oracle.py (64x32 output, 121 frames, latent 16x1x2).
Each stage of the official ICLoraPipeline conditions on its image guides first (combined_image_conditionings:
frame_idx 0 -> VideoConditionByLatentIndex at latent_idx 0) and then on the IC-LoRA reference video
(VideoConditionByReferenceLatent), and create_noised_state then runs the noiser. This oracle runs those original
functions in that order for a full and a fractional guide strength and records the conditioned state and two
masked Euler steps with fixed velocities.
"""
import argparse
import ast
import hashlib
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

REVISION = "be6d6d0a69d11a0a399c462ed92606692e5f8697"
STRENGTHS = (1.0, 0.6)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert subprocess.check_output(["git", "-C", str(args.reference), "rev-parse", "HEAD"], text=True).strip() == REVISION
    assert not subprocess.check_output(["git", "-C", str(args.reference), "status", "--porcelain"], text=True).strip()
    packages = args.reference.resolve() / "packages"
    sys.path[:0] = [str(packages / "ltx-core/src"), str(packages / "ltx-pipelines/src")]
    import torch
    from ltx_core.components.diffusion_steps import EulerDiffusionStep
    from ltx_core.components.noisers import GaussianNoiser
    from ltx_core.components.patchifiers import VideoLatentPatchifier
    from ltx_core.conditioning import VideoConditionByKeyframeIndex, VideoConditionByLatentIndex
    from ltx_core.conditioning.types.reference_video_cond import VideoConditionByReferenceLatent
    from ltx_core.tools import VideoLatentTools
    from ltx_core.types import VideoLatentShape
    from ltx_core.utils import to_denoised
    from ltx_pipelines.utils.types import ImageConditioningInput

    # The per-stage order: ICLoraPipeline._video_conditionings lists the image items before the reference items.
    ic_lora = (packages / "ltx-pipelines/src/ltx_pipelines/ic_lora.py").read_text()
    assert 0 < ic_lora.index("*image_conditionings_for_chunk(") < ic_lora.index("*reference_video_conditionings_for_chunk(")

    # Load the pure helper functions from their original AST (with the module's own future import), avoiding
    # unrelated pipeline/media optional dependencies. No source is modified. The image loader and VAE that
    # combined_image_conditionings calls are replaced by the synthetic encoded image below.
    path = packages / "ltx-pipelines/src/ltx_pipelines/utils/helpers.py"
    tree = ast.parse(path.read_text())
    names = ("combined_image_conditionings", "create_noised_state", "state_with_conditionings",
             "post_process_latent", "timesteps_from_mask")
    future = [node for node in tree.body if isinstance(node, ast.ImportFrom) and node.module == "__future__"]
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(future) == 1 and len(selected) == len(names)
    torch.set_num_threads(1)
    image = torch.randn(1, 128, 1, 1, 2, generator=torch.Generator().manual_seed(20261017))
    loaded = object()

    def load_image_and_preprocess(**_kwargs):
        return loaded

    def video_encoder(pixels):
        assert pixels is loaded
        return image

    helpers = {"torch": torch, "VideoConditionByLatentIndex": VideoConditionByLatentIndex,
               "VideoConditionByKeyframeIndex": VideoConditionByKeyframeIndex,
               "load_image_and_preprocess": load_image_and_preprocess}
    exec(compile(ast.Module(body=future + selected, type_ignores=[]), str(path), "exec"), helpers)

    tools = VideoLatentTools(VideoLatentPatchifier(1), VideoLatentShape(1, 128, 16, 1, 2), 24)
    reference = torch.linspace(-.7, .9, 128 * 16 * 1 * 2).reshape(1, 128, 16, 1, 2)
    canvas_tokens = tools.target_shape.token_count()
    noise = torch.linspace(-1.3, 1.7, 2 * canvas_tokens * 128).reshape(1, 2 * canvas_tokens, 128)

    class FixedNoiser(GaussianNoiser):
        def _sample_noise(self, latent_state):
            assert latent_state.latent.shape == noise.shape
            return noise

    flat = lambda tensor: tensor.detach().flatten().tolist()
    report = {"source_revision": REVISION, "width": 64, "height": 32, "frames": 121, "fps": 24,
              "latent_shape": [128, 16, 1, 2], "reference_shape": [128, 16, 1, 2], "image_shape": [128, 1, 1, 2],
              "reference_latent": flat(reference), "image_latent": flat(image), "noise": flat(noise), "cases": []}
    sigmas = torch.tensor([1, .4, 0], dtype=torch.float32)
    report["sigmas"] = flat(sigmas)
    for strength in STRENGTHS:
        images = helpers["combined_image_conditionings"](
            images=[ImageConditioningInput(path="first-frame.png", frame_idx=0, strength=strength)],
            height=32, width=64, video_encoder=video_encoder, dtype=torch.float32, device=torch.device("cpu"))
        assert len(images) == 1 and type(images[0]) is VideoConditionByLatentIndex and images[0].latent_idx == 0
        # append_ic_lora_reference_video_conditionings with scale 1, temporal scale 1, no prefix, strength 1 and full
        # conditioning attention adds this item unwrapped.
        conditionings = [*images, VideoConditionByReferenceLatent(reference, downscale_factor=1, strength=1.0)]
        state = helpers["create_noised_state"](tools=tools, conditionings=conditionings, noiser=FixedNoiser(torch.Generator()),
                                               dtype=torch.float32, device=torch.device("cpu"), noise_scale=sigmas[0].item())
        # Disjoint tokens: references first, then frame 0, gives the identical state (TensorSharp's order).
        swapped = helpers["create_noised_state"](tools=tools, conditionings=conditionings[::-1], noiser=FixedNoiser(torch.Generator()),
                                                 dtype=torch.float32, device=torch.device("cpu"), noise_scale=sigmas[0].item())
        reverse_identical = all(torch.equal(getattr(state, name), getattr(swapped, name))
                                for name in ("latent", "clean_latent", "denoise_mask", "keyframes_mask", "positions"))
        case = {"strength": strength, "order": [type(item).__name__ for item in conditionings],
                "reverse_order_identical": reverse_identical,
                "initial": flat(state.latent), "clean": flat(state.clean_latent), "mask": flat(state.denoise_mask),
                "keyframe_mask": flat(state.keyframes_mask), "positions": flat(state.positions.transpose(1, 2)), "steps": []}
        for step in range(len(sigmas) - 1):
            timesteps = helpers["timesteps_from_mask"](state.denoise_mask, sigmas[step])
            velocity = torch.sin(state.latent * .1 + step * .25) * .2
            denoised = to_denoised(state.latent, velocity, timesteps)
            denoised = helpers["post_process_latent"](denoised, state.denoise_mask, state.clean_latent)
            state = replace(state, latent=EulerDiffusionStep().step(state.latent, denoised, sigmas, step))
            case["steps"].append({"timesteps": flat(timesteps), "velocity": flat(velocity), "output": flat(state.latent)})
        report["cases"].append(case)
    core = packages / "ltx-core/src/ltx_core"
    pipelines = packages / "ltx-pipelines/src/ltx_pipelines"
    sources = {f"ltx_core/{name}": core / name for name in [
        "conditioning/types/latent_cond.py", "conditioning/types/reference_video_cond.py", "components/patchifiers.py",
        "tools.py", "components/noisers.py", "components/diffusion_steps.py"]}
    sources |= {f"ltx_pipelines/{name}": pipelines / name for name in ["utils/helpers.py", "ic_lora.py", "iclora_utils.py"]}
    report["source_sha256"] = {name: hashlib.sha256(file.read_bytes()).hexdigest() for name, file in sources.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
