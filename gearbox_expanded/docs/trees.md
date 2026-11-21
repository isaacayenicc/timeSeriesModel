# Random forest and gradient-boosted regression

Both ensembles receive a causal tabular row \(z_t\) and estimate the standardized vibration sample \(x_t\). A forest averages independently fitted trees. Boosting adds trees sequentially to the current predictor. The experiment tests how well threshold-based models can represent a rapidly changing signed waveform with the same prior information as the time-series models.

## Raw signal to feature row

The temporal compact version has 31 vibration delays. It uses consecutive lags 1–16 and spaced lags up to 240 samples. The physics compact version has 58 inputs. It adds the previous RPM reading and backward RPM differences over 16 and 64 samples, eight angularly spaced vibration samples, fourteen fixed quadratures and two past energy summaries. The angular lags include mesh-cycle multiples and fractions of one shaft revolution. Interpolation endpoints are past samples, and the angular search is curtailed at 768 samples. The quadratures use complex phase accumulation through the observed speeds at six gearbox orders and at 1100 Hz, with a 160-sample exponentially weighted history. The physical features summarize waveform phase and amplitude. They remain fixed during ensemble fitting.

Tree splits use the order of feature values. Scaling a feature by a positive constant and shifting it preserves that order. The experiments use the common feature builder and do not fit a feature scaler. The temporal and physics versions each use the same target indices, leaving the feature differences as the independent variable in a matched architecture configuration.

## Random forest regression

A regression tree recursively partitions input space. At a numeric split \((j, c)\), a row goes left if \(z_{tj}\le c\). The split statistic is the reduction in the within-node squared error relative to keeping the parent intact. A leaf predicts the mean vibration of fitting targets that reached it. A single finite-depth tree therefore gives a piecewise-constant form
\[
f_m(z_t)=\sum_{\ell\in\mathcal L_m}\bar x_{m\ell}\,\mathbf 1\{z_t\in R_{m\ell}\}.
\]
The forest takes \(\hat x_t=M^{-1}\sum_{m=1}^{M}f_m(z_t)\). Each tree uses a bootstrap sample of permitted fitting rows. Feature subsampling at each split reduces correlation among trees and can let important minority features enter some trees. Larger depth and smaller leaves improve fit to detail while increasing variance. This experiment limits depth, leaf size, bootstrap fraction, and forest size to keep fit time bounded. Bootstrapping happens inside the guarded fitting blocks. Fold selection evaluates deterministic contiguous time blocks, which accounts for the overlap of raw time windows that would undermine a random row holdout.

## Histogram gradient boosting regression

For squared error, the initial predictor is the mean of the available fitting targets. At boosting stage \(m\), the negative gradient is \(r_t=x_t-F_{m-1}(z_t)\). The tree partitions those residuals using split candidates formed from training-time feature bins. A regularized residual fit supplies a leaf value. An update has the form
\[
F_m(z_t)=F_{m-1}(z_t)+\eta h_m(z_t),
\]
where \(\eta\) is the learning rate and \(h_m\) is the stage's piecewise-constant tree. For a fixed weighted region with unit weights and squared loss, a leaf adjustment with L2 penalty has the characteristic denominator \(n_\ell+\lambda\) in place of \(n_\ell\). This reduces large updates in small leaves. Histogram binning reduces fitting cost. Inference retains the learned numeric thresholds and has no full-record bin recalculation. Validation-based early stopping is disabled, and the finite search varies the boosting budget alongside leaf count, regularization and learning rate using only the prescribed training folds.

Boosting can make a sequence of small corrective fits where the forest averages many deeper trees. Both operate on threshold-defined regions. The large number of phase, amplitude and operating-speed combinations seen in one short record can make those regions too coarse for extrapolation. The phase-informed feature set presents some of the needed geometry explicitly. Model-specific guarded results quantify how much this helps on supplied data.

## Experiment and serialization

The three held-out ranges are [4800,7200), [11200,13600) and [18400,20800) in `sigtrain`. A 1,792-sample guard on each side isolates fitting blocks, and a 1,792-sample warm-up is applied locally to each allowed fit block. Fold evaluation uses true earlier observations as available in deployment. An equally weighted mean of fold MSE selects each family. The development score uses 22,208 targets over [1792,24000) of `sigvalid`. Each family is refit on the full designated training record after selection. The candidate grid has a matched temporal/physics pair plus broader architecture settings. The ensemble's piecewise thresholds, topology, leaf values and feature definition are saved in one numerical NPZ checkpoint without pickled estimator state.

Checkpoint inference from stored node arrays matches the selected scikit-learn estimator on the complete scored development suffix to 2e-12 absolute tolerance. The export test also covers split routing for missing inputs. Normal data intake applies the shared finite-data protocol. Tree parameter counts in result tables report total stored nodes; each node includes an index topology and a threshold or leaf value. This complexity measure differs from the count of trainable floating-point weights in neural models. CPU latency includes reconstructing its causal feature matrix and traversing its learned trees from an in-memory 7.5-second record. Seven warm repeats measure elapsed time, while the reported cold call is the first prediction after reading the checkpoint. The run measures CPU only.
