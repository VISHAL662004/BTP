# data/

LUNA16 data is **not** stored in git (~66 GB). Download it from https://luna16.grand-challenge.org/ (data licence: CC BY 4.0) and place these files here:

| File | Needed for |
| ---- | ---------- |
| `subset0.zip` … `subset9.zip` | CT scans (read directly from the zips, never extracted) |
| `seg-lungs-LUNA16.zip` | lung segmentation masks |
| `annotations.csv` (tracked) | nodule annotations |
| `candidates_V2.zip`, `candidates.csv` | candidate lists |
| `evaluationScript.zip` | official FROC/CPM evaluation inputs |

Then run `python scripts/validate_dataset.py`, `python scripts/build_candidate_index.py` and `python scripts/build_patch_cache.py`.

Tracked, derived files: `metadata/` (scan statistics, validation outputs) and `splits/` (scan-level train/val/test).
Git-ignored, regenerated locally: `processed/` (patch cache, ~4 GB), `candidates/` (candidate index), `annotations/` (official excluded findings).
