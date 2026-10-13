"""Load trained weights + config. Pure model path: no data/ imports (SPEC 0.1)."""
import json
import os

import torch

from model.gravity import MUMGravityModel

DEFAULT_CKPT = os.path.join("checkpoints", "mum_gravity_pilot_stage3.pt")
DEFAULT_CFG = os.path.join("checkpoints", "config.json")


def load_model(ckpt_path=DEFAULT_CKPT, config_path=DEFAULT_CFG):
    """Returns (model, meta dict). Weights via torch.save .pt (safetensors
    unavailable offline; recorded in RESULTS.md)."""
    meta = {}
    if config_path and os.path.exists(config_path):
        meta = json.load(open(config_path))
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    mcfg = meta.get("model") or blob["config"]["model"]
    model = MUMGravityModel(**mcfg)
    model.load_state_dict(blob["model_state"])
    model.eval()
    meta.update({"stage": blob.get("stage"), "val_loss": blob.get("val_loss"),
                 "ckpt_path": ckpt_path})
    return model, meta
