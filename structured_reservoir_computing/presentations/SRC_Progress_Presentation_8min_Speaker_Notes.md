# SRC Progress speaker notes

Planned speaking duration is **8:00**. Backup slides are untimed.

## 1. Predicting gearbox vibration at changing speed

**0:00–0:45**

This is a retrospective reconstruction of the progress stage from the completed project. Imagine following a gear through its speed cycle. A shaft-synchronous component changes frequency in time, even when it repeats consistently in shaft angle. The practical objective is to predict the next healthy-reference vibration sample. We then inspect the remaining signal for evidence that repeats with shaft angle. This deck explains the initial study design and the representative evidence available in the released run.

## 2. What should a better model improve?

**0:45–1:30**

We compare mean squared one-step error against the course reference on fixed target indices. We measure actual CPU batch time and report GPU availability separately. The diagnostic question concerns whether a repetitive disturbance remains visible after subtracting the predicted waveform. A lower MSE can coexist with lower pulse preservation. This distinction gives us a way to compare model structure and diagnostic utility.

## 3. Use the official training and validation records

**1:30–2:30**

The supplied prediction records each have twenty-four thousand samples at 3,200 hertz. Keep the supplied standardization and use training-only transforms for new features. Treat record grouping as an acquisition question. Exact joint vibration and speed checks can reveal shared signals across the candidate and official files. Verify labels separately because an index group cannot establish a verified condition. The final blind test and fault files are unavailable in this release.

## 4. Follow mesh harmonics in a shaft-angle frame

**2:30–3:40**

The monitored shaft's 37-tooth gear provides a nominal mesh order of thirty-seven. RPM determines instantaneous hertz and makes mesh ridges drift in a time-frequency display. Angular resampling offers a complementary view in cycles per turn. Energy near the nominal harmonics motivates order-locked features. Broad fixed-hertz content can come from several mechanisms, so it needs separate controlled testing. Whole-record spectra are explanatory observations. Prediction features use only measurements available through the previous sample.

## 5. Build a structured reservoir of past observations

**3:40–4:45**

The reservoir is a deliberately structured expansion of measured history. It includes raw vibration lags, equal-angle histories, finite damped harmonic states and prior innovations. A learned readout maps these coordinates into the next sample. The optional clipped historical-error correction updates only after a target arrives. The profile can select or disable a feature family within the guarded training folds. The oscillator state summarizes previous measurements at their relative phase.

## 6. Constrain the feature library before fitting

**4:45–5:45**

Each group has an engineering interpretation. Sample lags model short material and structural responses. Angular lags compare points at similar tooth phase. Oscillator states retain coherent energy with a finite decay, and innovation lags can capture serial error structure. A restricted interaction library tests useful coupling with speed and amplitude. Group-scaled regularization limits the effect of weak, correlated inputs. The archive and direct regression models provide useful controls for this contribution.

## 7. Guard the held-out time blocks

**5:45–6:40**

Each held-out block is contiguous. A gap on both sides protects raw histories used for fit. Rank candidates only on training blocks using the predeclared combination of clean error, RPM stress error and synthetic pulse distortion. The supplied validation record is reserved for the selected models. Keep every benchmark on targets 1,792 through 23,999 and apply the same strict prior-input convention where eligible.

## 8. Use a verified comparison to frame the next test

**6:40–7:25**

PI-SRC MSE 0.329973 is 11.16% lower than the strict sparse LPV–ARMA result. The complete report distinguishes newly fitted methods, checkpoint replays and the external publication’s independent segment. At the progress stage, this is representative evidence for the method and the validation rules.

## 9. Prepare for an independent test

**7:25–8:00**

Candidate-record scores describe index groups while condition labels remain unverified. The reported computational measurements concern available batch CPU inference. The frozen manifest preserves model and evaluation identity for a future independent test. Continue by confirming condition labels, acquisition independence and the intended streaming hardware.

## 10. Order spectra align rotation-locked energy

**Backup**

Order is cycles per reference revolution. A spectral peak helps motivate a reservoir group. Guarded prediction and matched ablation test its incremental value.

## 11. Publication and project comparisons

**Backup**

The course paper and the supplied development record represent different evaluation segments. Report the published result as external context and compare methods on the frozen local scoring mask.
