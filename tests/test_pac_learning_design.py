"""Scientific controls for the selective cross-band pairing task."""
import importlib.util
from pathlib import Path

import numpy as np
import torch


def test_selective_pairing_preserves_band_waveforms_and_fourier_magnitudes():
    # Import must not parse CLI arguments, create outputs, or launch training.
    path = Path(__file__).resolve().parents[1] / "scripts/experiments/pac_tokenizer_learning.py"
    spec = importlib.util.spec_from_file_location("pac_learning_design", path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    for seed in (162001, 172001, 182001):
        x, y, construction_error = script.dataset(seed, 4, (4.3, 5.0), "cross_pair")
        repeat_x, repeat_y, _ = script.dataset(seed, 4, (4.3, 5.0), "cross_pair")
        torch.testing.assert_close(x, repeat_x, rtol=0, atol=0)
        torch.testing.assert_close(y, repeat_y, rtol=0, atol=0)
        assert x.shape == (8, 1, 1000) and construction_error < 1e-9
        np.testing.assert_array_equal(y.reshape(-1, 4), [[1, 1, 0, 0]] * 2)

        spectrum = np.fft.rfft(x[:, 0].numpy().astype(np.float64), axis=-1)
        frequency = np.fft.rfftfreq(1000, 1 / 200)
        low = np.fft.irfft(spectrum * (frequency <= 12), n=1000).reshape(-1, 4, 1000)
        high = np.fft.irfft(spectrum * (frequency >= 16), n=1000).reshape(-1, 4, 1000)
        # A class cannot be identified from its LF or HF component alone:
        # each component occurs once in either class within every quadruplet.
        np.testing.assert_allclose(low[:, 0], low[:, 2], rtol=0, atol=1e-5)
        np.testing.assert_allclose(low[:, 1], low[:, 3], rtol=0, atol=1e-5)
        np.testing.assert_allclose(high[:, 0], high[:, 3], rtol=0, atol=1e-5)
        np.testing.assert_allclose(high[:, 1], high[:, 2], rtol=0, atol=1e-5)
        magnitude = abs(spectrum).reshape(-1, 4, 501)
        assert abs(magnitude - magnitude[:, :1]).max() / magnitude.max() < 1e-6
