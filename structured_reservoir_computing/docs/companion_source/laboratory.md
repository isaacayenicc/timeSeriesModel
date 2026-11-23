## Appendix A. Worked laboratory sessions

The six sessions below connect the mechanics, software and numerical claims. Run the Python cells from the project root with the environment activated. They use the installed package and the saved result files. Each example has an assertion with a clear target. The synthetic checks are small and use deterministic data. No example imports prior research directories, opens blind records, or retrains study checkpoints.

### A.1. Shaft speed, angular phase and the Nyquist margin

Confusing an order with frequency leads to an incorrect feature bank. Consider a speed ramp from 120 to 160 RPM over four seconds. Its angular rate is the ramp divided by 60, in revolutions per second. At the start, the input shaft rotates at 2.0 Hz and mesh frequency is 74 Hz. At the end, the input shaft rotates at 2.667 Hz and mesh frequency is 98.667 Hz. The same nominal gear pair with 20 teeth on the pinion gives a pinion rotation rate of 3.7 Hz initially and approximately 4.933 Hz at the end. Multiplying the pinion rate by 20 recovers the same mesh frequency. Both counts describe the same contacts from different shafts.

The angle reached by a linear RPM ramp has a closed form. Set speed at time $t$ to $n(t)=n_0+at$, with $a$ in RPM per second. Shaft turns are

$$
u(t)=\frac{n_0t}{60}+\frac{at^2}{120}. \tag{A.1}
$$

The whole-record trapezoidal diagnostic phase matches this integral at each available timestamp because the integrand is affine in time. The one-step prediction phase uses a different final increment. At target $j$, it first integrates the measured data through $j-1$, then advances one step using $n_{j-1}$. Its error relative to the full trapezoidal integral is $-a\Delta t^2/120$ turns for this ramp. That discrepancy is small and deterministic. It buys a meaningful availability property. Corrupting $n_j$ must leave the target-$j$ phase intact.

```python
import numpy as np
from src_gearbox.diagnostics.order_tracking import revolutions
from src_gearbox.src_model import phase_turns

fs = 3200
interval = 1 / fs
t = np.arange(fs * 4 + 1) / fs
n0, ramp = 120., 10.
rpm = n0 + ramp * t
exact = (n0 * t + ramp * t**2 / 2) / 60
observed, causal = phase_turns(rpm, fs)
np.testing.assert_allclose(revolutions(rpm, t), exact, atol=5e-13, rtol=0)
np.testing.assert_allclose(observed, exact, atol=5e-13, rtol=0)
np.testing.assert_allclose(causal[1:] - exact[1:], -ramp * interval**2 / 120,
                           atol=5e-13, rtol=0)
print("Final turns:", exact[-1])
print("Mesh range (Hz):", (37 * rpm[[0, -1]] / 60).round(3))
```

The example gives $9\frac{1}{3}$ turns. A 37-tooth input shaft therefore passes around 345.33 tooth spaces. Under an order-synchronous signal $\cos(2\pi\cdot37 u(t))$, the instantaneous frequency uses the mesh range above. If a code cell returns a constant frequency, it has replaced the integrated angle with fixed time spacing. If it returns a ramp scaled by 60, inspect the RPM conversion.

For a strictly sampled signal, order $q$ requires $(q n/60)< f_s/2$. At the highest observed training RPM, the temporal Nyquist order is about 574. A feature library evaluated under synthetic RPM bias needs additional margin since the measured RPM may differ from the nominal trajectory. The selected highest bank order, 259, remains comfortably below this limit. A diagnostic grid of 2,048 samples per revolution has a numerical Nyquist order of 1,024 after interpolation. Interpolation supplies no independent vibration information above the original temporal Nyquist boundary. A denser angular grid improves numerical indexing and representation. It cannot recover analogue content already removed by downsampling or confounded by aliasing.

**Check yourself.** At fixed speed, compare time lag 100 and angular lag $100n/(60f_s)$ revolutions. For a ramp, a 100-sample lag accumulates a changing number of turns. An angular lag keeps turns constant while its support in time changes. This explains why a single fixed delay can represent impacts poorly at one speed and reasonably well at another. It also explains why an angle map can be fragile under RPM sensor bias. Both points can hold in the same experiment.

### A.2. Finite oscillator, scaling and loss of synchronization

Choose one order $q$, memory $M$ samples and decay $\tau$ samples. Write $z_j=x_j e^{-i\phi_j}$ with $\phi_j=2\pi q u_j$. The held state before target $t$ is

$$
s_t=\sum_{m=1}^{M}r^{m-1}z_{t-m},\qquad r=e^{-1/\tau}. \tag{A.2}
$$

Expanding $s_{t+1}$ yields $z_t+r z_{t-1}+\cdots+r^{M-1}z_{t-M+1}$. Subtracting $r^M z_{t-M}$ from $z_t+r s_t$ produces the same finite sequence. The normalization $2(1-r)/(1-r^M)$ follows from the geometric sum. One factor of two compensates for using the coherent positive-frequency part of a real sinusoid. At exact phase lock with many effective samples, the other, oscillating conjugate term largely cancels. Short memory and low carrier rates can leave interference. This is expected behaviour from a finite filter, rather than a calibration error.

```python
import numpy as np
from src_gearbox.src_model import oscillator_states, phase_turns

fs, order = 3200, 148
rpm = np.linspace(124., 167., 2300)
measured_turns, target_turns = phase_turns(rpm, fs)
x = .6 * np.cos(2 * np.pi * order * measured_turns)
memory, tau = 170, 40.
response = oscillator_states(x, fs, rpm, order=order, memory=memory, tau=tau)
target = 1400
lags = np.arange(1, memory + 1)
weights = np.exp(-(lags-1) / tau)
rotation = np.exp(2j * np.pi * order * (target_turns[target] - measured_turns[target-lags]))
reference = 2 * ((weights*x[target-lags]) @ rotation) / weights.sum()
np.testing.assert_allclose(response[target], reference, rtol=0, atol=3e-12)
print("Finite normalized state (real, imaginary):", (response[target].real, response[target].imag))
```

A fixed relative speed error $\eta$ yields a phase error that grows with the elapsed angular separation in a window. At fixed true speed, write the error per sample as $\delta=2\pi q\eta n/(60 f_s)$ radians. The retained coherent amplitude fraction of a complex sinusoid, ignoring its real-valued conjugate term, becomes the magnitude of the weighted geometric sum

$$
\mathcal A(\delta)=
\left|\frac{1-(r e^{i\delta})^M}{1-r e^{i\delta}}\right|
\frac{1-r}{1-r^M}. \tag{A.3}
$$

At $\delta=0$, the fraction is one. The longer the state memory scales, the larger the phase mismatch across the weighted support for a given sensor error. Equation (A.3) describes ideal coherent content. It cannot determine the fitted readout MSE under an actual replay, where angular delays, order mixtures and optional residual correction interact. It nevertheless gives a physical hypothesis for why +2% measured RPM has a large effect on some models. In real data the direction and size of the final performance change can depend on which modes dominate and on covariance during training.

**Check yourself.** Replace \(\tau=40\) by \(\tau=8\) and \(\tau=128\) in the example. Compare the magnitudes of the finite complex response. Test masks at the beginning and end of your simulated record rather than dropping them silently. Use the exact same time span across settings. A larger quadrature magnitude alone does not choose the oscillator for a forecasting task. The held-fold prediction error, state dimension, and robustness evidence answer separate questions.

### A.3. Regularized fit, redundant degrees and the selection ledger

A readout model may contain nearly identical columns. Consider two delays that coincide in a narrow speed interval. In a simplified algebraic example, the two training columns are exactly identical. Holding the combined weight $s=w_1+w_2$ fixed gives identical fitted predictions. If both weights carry the same ridge penalty, the minimum penalty occurs at $w_1=w_2=s/2$. If the penalties differ, let their positive values be $\lambda_1$ and $\lambda_2$. A constrained first derivative gives

$$
w_1=\frac{\lambda_2}{\lambda_1+\lambda_2}s,\qquad
w_2=\frac{\lambda_1}{\lambda_1+\lambda_2}s. \tag{A.4}
$$

The less penalized feature receives more of the allocation. This may reflect a numerical prior rather than independent physical explanatory value. It clarifies why coefficient magnitude is not sufficient for attribution, especially when oscillator, angular and proxy channels overlap. Changing whether a feature is scaled before ridge also changes its effective penalty in physical units. The checkpoints preserve scaling so inference evaluates the same coordinate system.

The next example reconstructs the outer objective. The repetitions within a candidate are held folds from the training recording. Their means enter the saved bounded selection. Omit the development CSV when reconstructing selection. The positive-part stress term does not reward a noise intervention for accidentally improving a fold.

```python
from pathlib import Path
import json
import numpy as np
import pandas as pd

folder = Path("results/optimization")
trials = pd.read_csv(folder / "trials.csv")
trials = trials.loc[trials.status.eq("DONE")].copy()
selection = json.loads((folder / "selected_configs.json").read_text())
cfg = selection["protocol"]
trials["reduced_cost"] = (
    trials.fold_mse
    + cfg["stress_excess_weight"] * (trials.fold_stress_mse - trials.fold_mse).clip(lower=0)
    + cfg["pulse_distortion_weight"] * trials.fold_pulse_distortion
    + cfg["per_coefficient_penalty"] * trials.parameters
)
np.testing.assert_allclose(trials.reduced_cost, trials.utility, atol=1e-12, rtol=0)
for row in selection["selection"]:
    trial = trials.set_index("trial_id").loc[row["trial_id"]]
    print(row["model"], row["trial_id"], round(trial.reduced_cost, 6))
```

The computed objective winner in the designated eligible group, the clean-only challenger, and a compact group are separate choices. Once these are frozen, keeping multiple deliberate outcomes gives a useful comparison of design priorities. One must avoid reading their development MSEs and promoting the nominal winner as if that alternative were prespecified. The measured difference between selected and clean-only outcomes is evidence of the practical trade-off on one record, with no information about another machine's operating distribution.

**Check yourself.** Increase the pulse penalty once after reconstructing the recorded objective. State that you are computing a hypothetical retrospective ranking of saved training guard scores. Keep the official rank and checkpoints unchanged. If this changes the selected trial, explain what it would buy on the scored training folds and at what nominal or computational cost. Do not infer improved real-fault preservation from the objective alone. Synthetic pulse families probe their tested shapes and conditions.

### A.4. Same mask, paired differences and useful precision

Taking a dot product of raw sample differences requires more than equal array lengths. The arrays must describe the same record, normalization, target indices and time stamps. Comparing signal history through one sample using different RPM support also changes the information available to the estimator. The project retains separate provenance for such reproductions and the archive. The calculation below checks the full alignment of the selected checkpoint, the strict baseline bundle, and the archived bundle. It independently identifies the sign of the paired effect on their actual scored targets.

```python
from pathlib import Path
import numpy as np
from src_gearbox.comparison import blocked, verify_axes
from src_gearbox.study_protocol import settings

p = Path("results/predictions")
with (
    np.load(p / "src_sigvalid.npz", allow_pickle=False) as new,
    np.load(p / "baselines_sigvalid.npz", allow_pickle=False) as conventional,
    np.load(p / "archived_src_sigvalid.npz", allow_pickle=False) as old,
):
    verify_axes(new, conventional)
    verify_axes(new, old)
    actual = new["actual"]
    squared = (actual - new["primary_pi_src"])**2
    anchor = (actual - old["archived_src_corrected"])**2
    strict = (actual - conventional["sparse_lpv_arma"])**2
    assert actual.size == 22208
    assert squared.mean() > anchor.mean()
    assert squared.mean() < strict.mean()

e = settings("evaluation")
for name, other in [("archive", anchor), ("strict sparse LPV–ARMA", strict)]:
    difference = other - squared
    intervals = [blocked(difference, size, e["bootstrap_replicates"], e["random_seed"])
                 for size in e["bootstrap_blocks"]]
    print(name, "MSE difference", round(difference.mean(), 6), intervals)
```

Under the shown sign convention a positive mean indicates lower error for the selected PI-SRC. Signs for the archive and strict conventional reference differ. State both. The archive's advantage does not invalidate the strict-reference comparison. The strict-reference comparison does not supersede the matched archival evidence.

The empirical squared-error process is correlated, for example because a burst of underprediction occupies several adjacent samples. If sample indices are resampled independently, a single transient is broken into many apparently unrelated observations. Moving blocks preserve correlated patterns within each drawn piece. The implemented helper samples from all contiguous noncircular pieces of the configured width and handles the remaining length so the bootstrap draw has the original number of target samples. That method neither creates independent records nor solves differences in speed or health conditions absent from the design.

The 640-sample block is 0.2 s at 3,200 Hz. At 150 RPM, one shaft turn lasts 0.4 s, so different block widths capture different degrees of within-revolution dependence. Reporting 320, 640 and 1,280 screens that sensitivity. The same figure should state units and pairing direction. Numerical digits beyond what the interval supports may still serve as file-identity checks. In prose, two or three substantive figures usually communicate error effects better than serialized machine precision.

**Check yourself.** Confirm the blockwise interval does not change if you add a constant $c$ to every difference except by the same $c$. This follows because every bootstrap draw uses the same total sample size. Comparing synthetic constant differences provides a simple unit check before using actual time only.

### A.5. PMI, finite-record bias and a measurement falsifier

Let a periodic positive envelope be observed for exactly $T$ turns on an equally spaced angular grid with $S$ samples per turn. Assume its mean over all samples is exactly $\mu$. At one turn, centered samples match pointwise wherever both are present. A zero-padded, biased circular-free correlation sums only $T-1$ turns at that lag and all $T$ turns at zero lag. Thus $\rho=(T-1)/T$. The selected diagnostic definition is

$$
\mathrm{PMI}=\frac{R(S)}{R(0)-R(S)}
=\frac{\rho}{1-\rho}=T-1. \tag{A.5}
$$

This is a useful check because an unbiased correlation using only the number of overlapping terms in its normalization gives a very different result. It can make a perfectly repeating waveform yield a denominator of zero. Our compatibility and direct diagnostic modes document what each lag and normalization means. Different conditions with equal physical periodicity but different record length need not have identical PMI under this convention.

```python
import numpy as np
from src_gearbox.diagnostics.pmi import pmi_from_envelope

points_per_turn, turns = 2048, 8
angle = np.arange(points_per_turn * turns) / points_per_turn
envelope = 3.0 + np.cos(2*np.pi*angle)
score = pmi_from_envelope(envelope, points_per_turn)
same_physics_half_record = pmi_from_envelope(envelope[:len(envelope)//2], points_per_turn)
np.testing.assert_allclose((score.correlation,score.value), (7/8,7), atol=1e-12)
np.testing.assert_allclose(same_physics_half_record.value, 3, atol=1e-12)
literal_lag = pmi_from_envelope(envelope, points_per_turn-1)
assert literal_lag.status == "ok"
print(score, "literal historical lag:", literal_lag)
```

This exact example starts from an envelope. The real pipeline constructs envelope magnitude from the angle-resampled residual and checks invalid and very short inputs. Do not apply a full-record Hilbert transform before one-step prediction, since samples far to the right influence the analytic representation. It is legitimate when the purpose is an offline postrecord diagnostic. The exercise also does not simulate how a real crack modulates vibration. It isolates an equation and a lag convention using a checkable signal.

Next, use paired sensor interventions to ask whether a diagnostic effect survives a plausible measurement perturbation. If actual vibration is unchanged and only the predictor's RPM is perturbed, a substantial PMI increase is diagnostic nuisance sensitivity in the predicted residual. If only the angle map is perturbed, nominal time-domain prediction MSE should remain unchanged. This stage separation is a method-level falsifier. It demonstrates that an apparent vibration event may follow from altered RPM in either stage.

```python
from pathlib import Path
import numpy as np
import pandas as pd

s = pd.read_csv(Path("results/fault/rpm_stress.csv"))
m = s.loc[s.model.eq("primary_pi_src") & s.condition.eq("bias_plus_2pct")]
assert set(m.scope) == {"predictor", "diagnostics", "both"}
angle_only = m.loc[m.scope.eq("diagnostics")].iloc[0]
np.testing.assert_allclose(angle_only.mse,angle_only.clean_mse)
print(m[["scope","mse","lecturer_compatible","delta_pmi"]].to_string(index=False))
```

For this measurement, +2% RPM only in the predictor increases the selected model's PMI by roughly 0.098. The three scopes share the original vibration record. Their causal attributions pertain to a software intervention on recorded input, not a physical rig intervention. Without approved labels and representative independent conditions, a large group gap or AUC of one for index groups does not imply a deployed threshold with no false alarms.

### A.6. Inspect a fit, check the frozen bytes and evaluate trade-offs

Model identity joins raw data hashes, split definition, search source, selected variant, learned scales and serialized parameters. A convenient paper table shows rounded MSE and label. It cannot show whether the matching executable weights really correspond to those arrays. A small manifest check provides the missing link. Binary weights are read with pickle disabled. Temporary illustrative fits belong outside the frozen release registry.

```python
from pathlib import Path
import hashlib
import json
import numpy as np
from src_gearbox.src_model import load_model

root = Path(".")
manifest = json.loads((root / "checkpoints/src_registry.json").read_text())
entry = next(row for row in manifest["models"] if row["model"] == "primary_pi_src")
state = root / entry["checkpoint"]
assert hashlib.sha256(state.read_bytes()).hexdigest() == entry["sha256"]
assert manifest["target_mask_sha256"] == json.loads(
    (root/"results/metrics/evaluation_mask.json").read_text())["target_indices_sha256"]
model = load_model(state)
weights = model.readout_features
assert len(weights) == entry["feature_columns"] + 1
assert weights[0][:2] == ("intercept", "intercept")
with np.load(state, allow_pickle=False) as parameters:
    print("Stored arrays:", parameters.files)
print("Numeric terms available for grouped inspection:", len(weights))
```

A byte fingerprint depends on more than ideal floating-point results. Compression library, array layout, metadata and filing method can produce different bytes for numerically similar weights. Saved hashes track this exact archive. On a new host, compare masks and forecasts within the documented tolerance, then record a distinct build hash if the bytes differ. A manifest does not replace physical validation. It helps detect when a scientific comparison quietly changed artifacts.

The development table contains CPU batch latency from one machine, fitting time recorded under a specified scope, and coefficient counts with model-specific definitions. These measures describe different axes. For two threshold candidates in a deployment study, require comparable measured batch scopes before speaking of faster inference. The mean elapsed time divided by a nominal sample count gives amortized batch cost. A stream reaches one sample interval about every 312.5 microseconds under this acquisition rate. Whole-record processing at roughly 49,496 samples per second does not establish that each individual stream deadline is met. New streaming tests would need per-sample or bounded chunk measurements, latency percentiles and state warm-up policies.

**Check yourself.** Create a side-by-side table of primary, compact and archive state size, scored MSE and recorded latency. Inspect which columns are measured under matching checkpoints and protocol. If a forecast-only archive lacks replay cost, write “unavailable.” Never fill it with another model's runtime. A rational engineering choice can prefer one model under strict compute constraints and another under nominal MSE, depending on evidence outside the current data. Scientific accuracy comes from retaining the conditional definitions behind both decisions.
