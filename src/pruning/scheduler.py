"""Pruning schedules (what to prune at each progressive level)."""

# Cumulative global sparsity of prunable weights after each level (unstructured).
UNSTRUCTURED_SPARSITY = (0.30, 0.50, 0.70, 0.85, 0.95)
# Fraction of prunable channels removed (relative to the ORIGINAL widths) after each level (structured).
STRUCTURED_RATIO = (0.20, 0.40, 0.55, 0.70, 0.80)


def levels(kind: str, custom=None) -> tuple:
    if custom:
        return tuple(custom)
    if kind == "unstructured":
        return UNSTRUCTURED_SPARSITY
    if kind == "structured":
        return STRUCTURED_RATIO
    raise ValueError(f"Unknown pruning type {kind!r}")


def select_final_level(rows: list, ref_val_cpm: float, max_drop: float = 0.02):
    """Pre-registered selection rule (fixed before any pruning results were seen):
    the highest level (most pruned) whose VALIDATION CPM >= reference validation CPM - max_drop.
    Test results are never used. Returns the chosen row or None."""
    ok = [r for r in rows if r["val_cpm"] >= ref_val_cpm - max_drop]
    return max(ok, key=lambda r: r["level"]) if ok else None
