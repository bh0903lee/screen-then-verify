"""Integrity checks applied to every result file."""

import numpy as np
import pandas as pd

BANDS = ["u1.0", "u0.8", "u0.6", "u0.5", "u0.45", "u0.4", "u0.35", "u0.3", "u0.25", "u0.2", "u0.15", "u0.1"]
BEAM_POLICIES = ("beam", "beam_vrel", "beam_sketch")
DEFAULT_BEAM_BANDS = ("u0.4", "u0.3")


def rows_ok(d, policy, n_targets, beam_bands=DEFAULT_BEAM_BANDS, bands=BANDS):
    need = {"sat", "target", "band"}
    if not need <= set(d.columns):
        return False, f"missing columns {sorted(need - set(d.columns))}"
    if len(d) != n_targets * len(bands):
        return False, f"rows {len(d)} != {n_targets} targets × {len(bands)} bands"
    tgt = pd.to_numeric(d["target"], errors="coerce"); band = d["band"].astype(str)
    keys = list(zip(tgt, band))
    if len(set(keys)) != len(keys) or set(keys) != {(t, b) for t in range(n_targets) for b in bands}:
        return False, "target × band grid incomplete or duplicated"
    sat = pd.to_numeric(d["sat"], errors="coerce")
    scored = band.isin(list(beam_bands)) if policy in BEAM_POLICIES else pd.Series(True, index=d.index)
    s = sat[scored]
    if s.isna().any() or not np.isin(s.values, (0.0, 1.0)).all():
        bad = s[~np.isin(s.fillna(-1).values, (0.0, 1.0))]
        return False, f"{len(bad)} scored rows with sat not in {{0, 1}} (e.g. {bad.head(3).tolist()})"
    if policy in BEAM_POLICIES and sat[~scored].notna().any():
        return False, f"{int(sat[~scored].notna().sum())} unscored beam rows carry a sat value"
    return True, ""
