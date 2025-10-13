# Notebooks

- `fftboost_demo.ipynb`: Colab-friendly demo that shows the full EEB pipeline with visuals and determinism checks.

## Open in Google Colab

Use the badge below after pushing to GitHub. Replace `your-org` and `your-repo` with your namespace.

```
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your-org/your-repo/blob/main/notebooks/fftboost_demo.ipynb)
```

## Notes

- In Colab, uncomment the `pip install` cell at the top to install `fftboost` directly from GitHub.
- The demo uses:
  - Huber loss + early stopping (contiguous holdout)
  - Experts: fft_bin (always) + sk_band (when bands provided)
  - Deterministic artifact save/load with stable SHA256
