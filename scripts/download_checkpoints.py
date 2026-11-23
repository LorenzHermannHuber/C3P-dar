"""Download the pretrained weights from Hugging Face into checkpoints/<repository name>/.

python scripts/download_checkpoints.py              # all
python scripts/download_checkpoints.py radargen dcae

radargen: the original RadarGen (TruckScenes) diffusion model, the baseline of scripts/generate.py
dcae:     DC-AE, the latent autoencoder of RadarGen and of the DiT model
sana:     SANA-600M, the initialization of the DiT model
dinov3:   DINOv3 ViT-S/16, the initialization of the DINO models

The empty-prompt Gemma token (checkpoints/empty_prompt_gemma-2-2b-it.safetensors) is committed, so no
text encoder is needed.
"""

import argparse

from huggingface_hub import hf_hub_download

from c3pdar import CHECKPOINTS

FILES = {
    "radargen": ("TomerBo/RadarGen_600M_512px_TS", ["RadarGen_600M_512px_TS.safetensors"]),
    "dcae": ("mit-han-lab/dc-ae-f32c32-sana-1.1", ["config.json", "model.safetensors"]),
    "sana": ("Efficient-Large-Model/Sana_600M_512px", ["checkpoints/Sana_600M_512px_MultiLing.pth"]),
    "dinov3": ("timm/vit_small_patch16_dinov3.lvd1689m", ["model.safetensors"]),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("models", nargs="*", default=list(FILES), help=f"any of {', '.join(FILES)}")
    args = parser.parse_args()
    if unknown := set(args.models) - set(FILES):
        parser.error(f"unknown models {sorted(unknown)}")

    for name in args.models:
        repository, files = FILES[name]
        for file in files:
            print(f"{name}: {repository}/{file}", flush=True)
            path = hf_hub_download(repository, file, local_dir=CHECKPOINTS / repository.split("/")[1])
            print(f"  -> {path}", flush=True)


if __name__ == "__main__":
    main()
