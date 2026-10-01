"""Run a detector over a PatchStore and score it with the LUNA16 FROC protocol."""
import numpy as np
import pandas as pd
import torch

from src.evaluation.froc import FrocResult, evaluate


@torch.no_grad()
def predict_proba(model, store, device, batch_size: int = 1024) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(store), batch_size):
        x = store.x[i:i + batch_size].to(device).float() / 255.0
        logit, _ = model(x)
        out.append(torch.sigmoid(logit).float().cpu())
    return torch.cat(out).numpy()


def score(store, prob, seriesuids, annotations, excluded, max_marks_per_scan=100):
    """Returns (FrocResult, predictions DataFrame in the official submission format)."""
    m = store.meta
    pred = pd.DataFrame({"seriesuid": m.seriesuid, "coordX": m.coordX, "coordY": m.coordY,
                         "coordZ": m.coordZ, "probability": prob})
    return evaluate(pred, seriesuids, annotations, excluded, max_marks_per_scan), pred


def evaluate_store(model, store, device, seriesuids, annotations, excluded, max_marks_per_scan=100):
    return score(store, predict_proba(model, store, device), seriesuids, annotations, excluded, max_marks_per_scan)
