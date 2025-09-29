
# fftboost

stagewise spectral feature selector + small linear head. built for time-series windows where the signal is sparse in the fft magnitude domain.

## install

```bash
pip install git+https://github.com/pinballsurgeon/fftboost.git --quiet
```

## what it does

* picks fft “atoms” by residual correlation with physics-aware priors (band quotas, min spacing, hf/coherence penalties)
* fits a compact ridge head on the selected atoms (+ optional aux features like wavelet energies, hilbert stats)
* blocked time-series cross-validation with paired Δr² reporting
* fast inference suitable for real-time windows

> note: not tree boosting; not generic gradient boosting. this is guided, additive atom pursuit in the frequency domain.

## when to use

* fundamentals/harmonics/interharmonics drive the target
* spectra are wide (10³–10⁴ bins) but the answer is sparse
* you want inspectable frequency picks and low latency

## quick start

```python
import numpy as np
from fftboost import FFTBoost          # stagewise atom selection + ridge head

fs = 1000
window_s = 0.5
x = np.random.randn(int(60*fs))        # your 1-D signal
y = np.random.randn(int(60/window_s))  # your per-window target

model = FFTBoost(
    fs=fs,
    window_s=window_s,
    hop_s=0.25,
    atoms=16,
    band_quotas={"lf":1, "fund":2, "ih1":2, "ih2":1, "hf":1},
    min_sep_bins=3,
    lambda_hf=0.10,
    lambda_coh=3.0
)

model.fit(x, y)
yhat = model.predict(x)

print("selected bins (hz):", model.selected_freqs_hz_)
print("cv summary:", model.cv_report_)  # includes paired Δr² vs baseline
```

## design notes

* residual-driven, stagewise atom selection (greedy, with priors)
* periodic ridge refits to stabilize coefficients
* auxiliary features are optional and do not drive selection (keeps frequency picks honest)

## roadmap

* shrinkage + per-stage contributions (toward true boosting)
* pluggable losses (huber/quantile) and line search
* early stop on blocked validation
* minimal model artifact export/import

## license

mit

## maintainer

dan ehlers — [github: pinballsurgeon](https://github.com/pinballsurgeon)
linkedin: [https://www.linkedin.com/in/dan-ehlers-32953444/](https://www.linkedin.com/in/dan-ehlers-32953444/)
email: [pinballsurgeon@gmail.com](mailto:pinballsurgeon@gmail.com)

---
