# Neural comparators for one-step vibration prediction

A forecasting window for target \(t\) contains the \(L\) measured pairs \((x_{t-L}, n_{t-L}),\ldots,(x_{t-1}, n_{t-1})\). Here \(x\) is vibration in the course file's standardized units and \(n\) is RPM. The target is the next measured \(x_t\), and all models take their history from the measured record during one-step evaluation. Each fit centers and scales the two input channels using its allowed training slices. The label stays in the supplied standardized vibration units. Hence MSE is directly comparable on a common target mask.

The affine shortcut gives each comparator direct access to every prior sample. The nonlinear encoder is the varying component.

\[
\hat x_t = b + w^\top\operatorname{vec}(W_t) + g_\theta(W_t).
\]

All coefficients in this equation are fitted together. The shortcut has the same form across the five families and can learn an autoregression while the encoder learns departures from it. Because all five candidates include the shortcut, these results describe neural predictors with a linear lag term. A separate shortcut-off comparison would be needed to attribute a given difference to nonlinear capacity alone.

## Architectures and time support

| Family | Nonlinear encoder | History representation |
|---|---|---|
| MLP | Two fully connected GELU layers followed by a scalar readout | Flattened vibration and RPM samples |
| LSTM | One unidirectional recurrent layer with input, forget and output gates. The last hidden state feeds the scalar head. | Chronological eight-sample patches |
| GRU | One unidirectional recurrent layer with update and reset gates. The last hidden state feeds the scalar head. | Chronological eight-sample patches |
| CNN | Two ordinary kernel-3 convolutions with local padding followed by a fully connected scalar head | Chronological eight-sample patches |
| TCN | Kernel-2 causal convolutions with dilations 1, 2, 4, … and residual addition. The last patch feeds the scalar head. | Chronological eight-sample patches |

A patch concatenates all eight vibration–RPM pairs into 16 channels. No raw samples are dropped. The 64, 128 and 192-sample histories contain 8, 16 and 24 steps. Patch framing changes the step size of the recurrent and convolutional operators from 0.3125 ms to 2.5 ms at the course sampling rate. LSTM and GRU reset state at the beginning of each target window, so their memory is bounded by the supplied history. The symmetric CNN processes the full historical window. Its intermediate features can combine both neighbors because both precede the forecast target. The TCN left-pads every dilated layer, so its intermediate features also have strict patch-level left support. In either convolutional architecture the entire input window ends at \(t-1\).

For a kernel of width 2 with successively doubled dilation, \(D\) layers cover \(1+\sum_{j=0}^{D-1}2^j=2^D\) patches. The implementation chooses enough layers to cover the configured window. For 24 patches it has a 32-patch theoretical receptive field and supplies zeros at positions before the start of the window. It never carries padding or state across data records.

LSTM's explicit cell state can preserve features over multiple steps. Its gates select what enters that state, what survives and how much enters the output. A GRU combines the state and output representation and uses the reset and update gates to control candidate and retained state. A CNN shares local kernels across patch positions. A TCN obtains a larger contiguous receptive field using dilation. These are different parameterizations of the same measured-history forecasting problem.

## Selection and benchmarking

The three prespecified configurations per family use histories of 64, 128 and 192 samples, widths of 16, 24 and 32, and 8, 12 and 16 epochs respectively. Learning rates are 0.002, 0.002 and 0.0015. All use AdamW with weight decay 0.003, gradient-norm clipping at 1 and batches of 512. The same candidate schedules and seeds are used across families. They have different parameter counts and CPU costs, so this aligns the number of training opportunities and the raw input information.

Each trial scores the three training blocks [4800, 7200), [11200, 13600) and [18400, 20800). Its fitting slices end 1,792 samples before each held-out block and restart 1,792 samples after the block. Each slice provides only its local complete windows, and scaling uses only those retained slices. A held-out forecast can use measured past samples from the same block as they would arrive in one-step use. It cannot use the target or any later sample. The three equally weighted fold errors select a configuration before accessing the validation values for scoring. Then the model is refit on all of `sigtrain` and evaluated against `sigvalid` at zero-based targets [1792, 24000).

Validated checkpoints store the configuration, input scaling, sampling rate and typed tensor arrays in a compressed NPZ. Reloading uses `allow_pickle=False`. The numerical replay check compares every scored prediction. The causal intervention modifies both vibration and RPM starting at the midpoint of validation and checks earlier forecasts, including the midpoint target itself. The same one-thread CPU protocol times five repeated in-memory predictions on the complete validation record, including input scaling, window construction and batched inference. Externally timed runs can differ under shared CPU load, and a batch time does not measure per-sample online latency.

## References

- Hochreiter, S. and Schmidhuber, J. (1997). *Long Short-Term Memory*. Neural Computation 9(8), 1735–1780. [DOI](https://doi.org/10.1162/neco.1997.9.8.1735)
- Cho, K. et al. (2014). *Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation*. [arXiv:1406.1078](https://arxiv.org/abs/1406.1078)
- Bai, S., Kolter, J. Z. and Koltun, V. (2018). *An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling*. [arXiv:1803.01271](https://arxiv.org/abs/1803.01271)
