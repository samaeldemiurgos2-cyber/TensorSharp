namespace TensorAgent.Core.Catalog;

public static partial class ModelCatalog
{
    // Exact files used by the native distilled text/image-to-video/audio pipeline. The five
    // base files are from unsloth/LTX-2.5-GGUF at ac3c540c (read on 2026-10-18), which is not
    // gated. The optional Ingredients IC-LoRA is the official gated Lightricks release pinned
    // to its 2026-09-10 commit; HF_TOKEN or an accepted Hugging Face token file is required.
    // The denoiser is Unsloth's Dynamic 2.0 mixed-precision 4-bit file from the distilled/ folder
    // (the repository root holds the dev checkpoint, which the engine refuses); companions are
    // publisher files mirrored byte for byte in the Unsloth repository.
    private const string Ltx25Repository = "https://huggingface.co/unsloth/LTX-2.5-GGUF/resolve/ac3c540c6025903784336e99b2dc179385c4c3b7/";

    private static CatalogModel Ltx25Distilled() => new()
    {
        Id = "ltx-2.5-distilled-q4kxl",
        DisplayName = "LTX 2.5 Distilled",
        Family = CatalogFamily.Ltx25,
        Kind = CatalogArchitectureKind.Diffusion,
        // GGUF tensor inventory includes ~21B parameters with its AV connectors;
        // the authentic custom Gemma4 encoder is a separate 12B checkpoint.
        Parameters = "21B + 12B",
        Quantization = "UD-Q4_K_XL + BF16",
        Files = new[]
        {
            new CatalogFile(CatalogFileRole.Weights, "ltx-2.5-22b-distilled-UD-Q4_K_XL.gguf",
                Ltx25Repository + "distilled/ltx-2.5-22b-distilled-UD-Q4_K_XL.gguf",
                14_000_234_880, "55442750eb0013d94fffaa4dadb577c5f4dc07107ec04389d41444f6e517d980"),
            new CatalogFile(CatalogFileRole.TextEncoder, "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
                Ltx25Repository + "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
                26_263_858_182, "ef7243612fdae7a75cb4d5cee9433e81380675fb6c213bd98ae74a9cd16561d1"),
            new CatalogFile(CatalogFileRole.Vae, "ltx-2.5-video-vae-conv-bf16.safetensors",
                Ltx25Repository + "vae/ltx-2.5-video-vae-conv-bf16.safetensors",
                1_452_269_922, "685b06ee3d9b2039647698fc4ea33175112462fc374e2777312c907897dfce8d"),
            new CatalogFile(CatalogFileRole.AudioVae, "ltx-2.5-audio-vae-bf16.safetensors",
                Ltx25Repository + "vae/ltx-2.5-audio-vae-bf16.safetensors",
                364_866_540, "c52733d37f6a7fb7949c3dc0fb468c6cb2169e4d836983a73babb9f0d54837a5"),
            // The publisher's two-stage recipe: half-size pass, x2 latent upscale, refinement.
            new CatalogFile(CatalogFileRole.LatentUpscaler, "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
                Ltx25Repository + "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
                995_778_752, "eb5a71fe4068ee87ccdb1c3aa635e547ca76bd2d30ae20ae889f2c325c0677e8"),
            // Keep the gated optional file after all base assets: if access is denied,
            // the downloaded model remains usable for ordinary video generation.
            new CatalogFile(CatalogFileRole.IngredientsLora, "ltx-2.5-22b-ic-lora-ingredients-0.9.safetensors",
                "https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Ingredients/resolve/12040e4091ac2008d3906a594e31a7fb1ab9d546/ltx-2.5-22b-ic-lora-ingredients-0.9.safetensors",
                1_308_787_472, "ff873a5beada3c579a8137c7c53343916f78bcc9a529ba910143073fe8715e95", Optional: true),
        },
        Modalities = CatalogModalities.Image | CatalogModalities.VideoOutput | CatalogModalities.AudioOutput,
        // Conservative device tier; initial full-generation validation uses short clips.
        MinDeviceMemoryGB = 48,
        ContextLength = 0,
        KvCacheDtype = "f16",
        Sampling = new CatalogSampling(1f, 0, 1f, 0f),
        LeanCaches = true,
        Experimental = true,
        Notes = "catalog.model.ltx25.notes",
        License = "catalog.license.ltx25",
    };
}
