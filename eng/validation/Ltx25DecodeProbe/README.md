# LTX convolutional VAE decode probe

Exercise the real MLX convolutional video decoder without running the denoiser.
The checkpoint is read without modification. Synthetic, bounded latents make the
run repeatable; they do not establish video quality or complete generation speed.

From the repository root, with the normal MLX native libraries already built:

```sh
dotnet build eng/validation/Ltx25DecodeProbe -c Release \
  -p:TensorSharpSkipGgmlNative=true -p:TensorSharpSkipMlxNative=true -m:1
TENSORSHARP_MLX_LIBRARY_DIR="$PWD/TensorSharp.Backends.MLX/Native/dist" \
  dotnet eng/validation/Ltx25DecodeProbe/bin/Release/net10.0/Ltx25DecodeProbe.dll \
  /absolute/path/ltx-2.5-video-vae-conv-bf16.safetensors \
  257 1024 1024 docs/validation/ltx-decode/report.json
```

Arguments are the checkpoint, output frame count, width, height and fresh report
path. Frames must follow `8n+1`; spatial dimensions must be multiples of 32. The
257-frame, 1024-square case exceeds 2³¹ elements in a decoder activation and needs
substantial unified memory. The report records geometry, finite RGB checks,
elapsed time and the process peak working set. Native dependency revisions and
device conditions should be recorded separately with the validation evidence.
Store reports and logs in ignored `docs/validation/` or `artifacts/` directories.
