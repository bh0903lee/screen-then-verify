"""Result paths, CSV writing and checkpoint I/O."""
import json, subprocess, time
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"; RES.mkdir(exist_ok=True)
CKPT = RES / "checkpoints"; CKPT.mkdir(exist_ok=True)
MANI = RES / "manifests"; MANI.mkdir(exist_ok=True)


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT)).decode().strip()
    except Exception:
        return "nogit"


def run_id(exp, tag=""):
    return f"{exp}_{tag}" if tag else exp


def save_manifest(exp, tag, params):
    m = {"exp": exp, "tag": tag or "(default)", "git": git_commit(),
         "time": time.strftime("%Y-%m-%d %H:%M:%S"), "params": {k: (str(v) if not isinstance(v, (int, float, str, bool, list)) else v)
                                                                 for k, v in params.items()}}
    p = MANI / f"{run_id(exp, tag)}.json"
    try:
        p.write_text(json.dumps(m, indent=2, default=str))
    except Exception as e:
        print(f"[io] manifest write failed: {e}")
    return p


def meta_cols(exp, tag, seed, **knobs):
    d = {"exp": exp, "tag": tag or "", "seed": seed, "git": git_commit()}
    d.update(knobs)
    return d


def ckpt_path(exp, tag, seed, band):
    return CKPT / f"{run_id(exp, tag)}_seed{seed}_band{band}.pt"


def save_checkpoint(model, exp, tag, seed, band, meta=None):
    p = ckpt_path(exp, tag, seed, band)
    try:
        sd = model.state_dict() if hasattr(model, "state_dict") else model
        torch.save({"state_dict": sd, "exp": exp, "tag": tag, "seed": seed, "band": band,
                    "git": git_commit(), "time": time.strftime("%Y-%m-%d %H:%M:%S"), "meta": meta or {}}, p)
    except Exception as e:
        print(f"[io] checkpoint save failed ({p.name}): {e}")
    return p


def load_checkpoint(exp, tag, seed, band, map_location="cpu"):
    p = ckpt_path(exp, tag, seed, band)
    return torch.load(p, map_location=map_location) if p.exists() else None
