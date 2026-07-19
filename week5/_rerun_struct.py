"""Monkey-patch DATA_DIR so detectors load structural injected data,
then run all 4 methods via the official `run()` entry points.

Outputs scores as _structural npy files in cache/.
Restores original files on exit.
"""
import os, sys, shutil, traceback
from pathlib import Path

DATA_DIR = Path("/home/ubuntu/amazon_repo/week5/data")
CACHE_DIR = Path("/home/ubuntu/amazon_repo/week5/cache")

TEST_BAK   = DATA_DIR / "flow_test_injected.npy.bak_struct"
VAL_BAK    = DATA_DIR / "flow_val_injected.npy.bak_struct"


def swap_files():
    """Replace V3 injected files with structural variant for this run."""
    if TEST_BAK.exists():
        TEST_BAK.unlink()
    if VAL_BAK.exists():
        VAL_BAK.unlink()
    shutil.move(str(DATA_DIR / "flow_test_injected.npy"), str(TEST_BAK))
    shutil.move(str(DATA_DIR / "flow_val_injected.npy"),  str(VAL_BAK))
    shutil.copy(str(DATA_DIR / "flow_test_injected_structural.npy"),
                str(DATA_DIR / "flow_test_injected.npy"))
    shutil.copy(str(DATA_DIR / "flow_val_injected_structural.npy"),
                str(DATA_DIR / "flow_val_injected.npy"))
    print("[swap] injected files replaced with structural variant")


def restore_files():
    if TEST_BAK.exists():
        shutil.move(str(TEST_BAK), str(DATA_DIR / "flow_test_injected.npy"))
    if VAL_BAK.exists():
        shutil.move(str(VAL_BAK),  str(DATA_DIR / "flow_val_injected.npy"))
    print("[restore] original V3 injected files restored")


def main():
    sys.path.insert(0, "/home/ubuntu/amazon_repo")
    sys.path.insert(0, "/home/ubuntu/amazon_repo/week5")
    os.chdir("/home/ubuntu/amazon_repo/week5")

    swap_files()
    try:
        # 1. Statistical
        print("\n[1/4] Running statistical.run()")
        from anomaly.statistical import run as stat_run
        sv, st = stat_run("taxi_flow_total")
        np.save(str(CACHE_DIR / "stat_scores_val_structural.npy"),  sv)
        np.save(str(CACHE_DIR / "stat_scores_test_structural.npy"), st)
        print(f"  saved stat val/test shapes {sv.shape} / {st.shape}")

        # 2. Prediction (STF)
        print("\n[2/4] Running prediction.run()")
        from anomaly.prediction import run as pred_run
        result = pred_run("taxi_flow_total")
        if isinstance(result, tuple) and len(result) == 3:
            pv, pt, _ = result
        else:
            pv, pt = result[0], result[1]
        np.save(str(CACHE_DIR / "pred_scores_val_structural.npy"),  pv)
        np.save(str(CACHE_DIR / "pred_scores_test_structural.npy"), pt)
        print(f"  saved pred val/test shapes {pv.shape} / {pt.shape}")

        # 3. VAE
        print("\n[3/4] Running vae_v3.run()")
        from anomaly.vae_v3 import run as vae_run
        vv, vt = vae_run("taxi_flow_total")
        np.save(str(CACHE_DIR / "vae_scores_val_structural.npy"),  vv)
        np.save(str(CACHE_DIR / "vae_scores_test_structural.npy"), vt)
        print(f"  saved vae val/test shapes {vv.shape} / {vt.shape}")

        # 4. Transformer AE
        print("\n[4/4] Running transformer_ae_v3.run()")
        from anomaly.transformer_ae_v3 import run as tae_run
        tv, tt = tae_run("taxi_flow_total")
        np.save(str(CACHE_DIR / "tae_scores_val_structural.npy"),  tv)
        np.save(str(CACHE_DIR / "tae_scores_test_structural.npy"), tt)
        print(f"  saved tae val/test shapes {tv.shape} / {tt.shape}")

        print("\nALL DONE")
    finally:
        restore_files()


import numpy as np
if __name__ == "__main__":
    main()
