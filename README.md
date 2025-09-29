# FFTBoost

[![Acceptance Gate](https://github.com/pinballsurgeon/fftboost/actions/workflows/acceptance.yml/badge.svg)](https://github.com/pinballsurgeon/fftboost/actions/workflows/acceptance.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

FFTBoost is a time-series analysis library designed to build robust and interpretable models by leveraging frequency-domain features within a boosting-like framework.

### The Problem

Standard machine learning models applied to raw time-series data often overfit to high-frequency noise or learn patterns that are not physically meaningful. This can lead to "twitchy" or unstable predictions that are difficult to trust, especially in control system applications where stability and efficiency are paramount.

The goal of this project is to move beyond simple measurement accuracy and enable more stable, predictive insights.

### The FFTBoost Approach

FFTBoost addresses this problem with a **two-branch architecture** that separates the feature engineering and selection process:

1.  **The FFT Branch:** A specialized, iterative algorithm hunts for a sparse set of the most impactful Fast Fourier Transform (FFT) frequency bins. It greedily selects features based on their correlation with the model's residual error, prioritizing physically relevant frequency bands while penalizing noise.

2.  **The Auxiliary Branch:** A curated set of low-dimensional, stable features (such as wavelet energies and Hilbert phase statistics) provides the model with broad contextual information about the signal.

These two branches are combined in a simple, regularized final model (e.g., Ridge regression). This approach is designed to produce models that are more robust to noise, faster to run at inference, and more interpretable than "black-box" alternatives.

## Getting Started

### Prerequisites
*   Python 3.9+

### Installation
Install the package directly from GitHub:
```bash
pip install git+https://github.com/pinballsurgeon/fftboost.git
