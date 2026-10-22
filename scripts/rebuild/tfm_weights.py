"""Released TFM checkpoint paths; no model or training implementation."""
import os
from paclock_bench.paths import vendored

WEIGHTS = os.path.join(vendored("tfm"), "pretrained_weigths")
SINGLE_DIRS = {
    "tuab": "TUAB_tfm_tokenizer_2x2x8",
    "tuev": "TUEV_tfm_tokenizer_2x2x8",
    "chbmit": "CHBMIT_tfm_tokenizer_2x2x8",
}

def weight_paths(dataset: str, setting: str = "single") -> tuple[str, str | None]:
    """Locate the tokenizer and encoder checkpoints for a dataset."""
    if setting == "single":
        d = SINGLE_DIRS.get(dataset)
        if d is None:
            raise KeyError(
                f"no single-corpus TFM weights for {dataset!r}; "
                f"available: {sorted(SINGLE_DIRS)}")
        base = os.path.join(WEIGHTS, "single_dataset_settings", d)
        tok = os.path.join(base, "tfm_tokenizer_last.pth")
        enc = os.path.join(base, "tfm_encoder_best_model.pth")
    else:
        base = os.path.join(WEIGHTS, "multiple_dataset_settings")
        tok = os.path.join(base, "Pretrained_tfm_tokenizer_2x2x8",
                           "tfm_tokenizer_last.pth")
        enc = os.path.join(base, "MTP_Pretrained_tfm_encoder_64x4",
                           "tfm_encoder_mtp_last.pth")
    if not os.path.exists(tok):
        raise FileNotFoundError(tok)
    return tok, enc
