import numpy as np
from pathlib import Path

MODEL_PATH = Path("models/vil-encoder-v1.2.pt")
EMBED_DIM = 768

class VILEncoder:
    def __init__(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"[VIL] Missing model: {MODEL_PATH}")

        try:
            import torch
        except Exception as e:
            raise RuntimeError("[VIL] Real PyTorch is required to load .pt checkpoint") from e

        if not hasattr(torch, "load"):
            raise RuntimeError(
                "[VIL] Imported torch is not real PyTorch: missing torch.load. "
                "Run this on Linux/HF Space with torch installed."
            )

        ckpt = torch.load(MODEL_PATH, map_location="cpu")

        if "vision_encoder" not in ckpt or "temporal_head" not in ckpt:
            raise KeyError(f"[VIL] Checkpoint keys are {list(ckpt.keys())}")

        self.vision = ckpt["vision_encoder"]
        self.temporal = ckpt["temporal_head"]

        self.vision.eval()
        self.temporal.eval()
        self.torch = torch

    def encode(self, glyph_ids):
        x = self.torch.tensor(glyph_ids, dtype=self.torch.long).unsqueeze(0)

        with self.torch.no_grad():
            v = self.vision(x)
            out = self.temporal(v)

        return out.squeeze(0).detach().cpu().numpy()

vil_model = None

def get_vil():
    global vil_model
    if vil_model is None:
        vil_model = VILEncoder()
    return vil_model
