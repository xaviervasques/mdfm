# Design note: which backbone, and how much infrastructure (2026-09-24)

The owner asked whether MD_FM should use Samba, a mixture of experts, or another
architecture, and whether some architectures now need less infrastructure for equivalent
performance. This note answers from a read-only literature study: three independent research passes,
each re-checked against its sources by a skeptical second reading, then three judges with different
priorities (minimal infrastructure, gate performance, engineering risk). Evidence and
verdicts: reports/architecture_study/architecture_study_2026-09-24.json. Every compute
figure below is an estimate from published arithmetic and an assumed 35 percent model FLOP
utilization on the RTX 5080; plan task T0.9 measured 32 percent with length bucketing (section 4).

## 1. Answer

Keep the dense transformer of design section 4. Do not use Samba, a pure state-space model
or a mixture of experts in the backbone. The three judges agree on this, each from its own
priority. The architectures the owner has in mind do save infrastructure, but in a regime
MD_FM is not in: long sequences decoded one token at a time, and very large models trained
on very large data. MD_FM's cost sits elsewhere, and cutting it there moves the whole
program onto the local GPU.

## 2. Why not Samba or a state-space backbone

- The windows are short. A 10 s window gives 40 tokens per channel group; most hours are
  single-sensor records of about 80 to 120 tokens, and the largest multimodal window is
  about 1,500 tokens. At that length attention is at most about a quarter of the encoder's
  compute on the largest windows and a few percent on typical ones.
- The measured speed advantage of Mamba-style and linear-attention kernels over fused
  attention starts between 2,000 and 4,000 tokens (Mamba, arXiv 2312.00752; Mamba-2, arXiv
  2405.21060; flash-linear-attention benchmarks). Samba itself trained slower than a
  transformer at 4,000 tokens (arXiv 2406.07522, table 3), and the 8B Mamba-2 hybrid had
  the same training utilization as its transformer (arXiv 2406.07887). Their large gains,
  3 to 8 times, come from generating long sequences token by token; the MD_FM encoder reads
  a window in both directions and generates nothing.
- Quality at our scale favours attention. In EEG, the closest domain with a matched
  comparison, the transformer LaBraM beats the Mamba model FEMBA at a fraction of the
  parameters (arXiv 2502.06438 against 2405.18765). In a same-recipe comparison at 1B,
  pure Mamba-2 and the sliding-window attention of Samba lost recall against full
  attention (arXiv 2411.13676); cross-modal alignment in MD_FM is exactly that kind of
  lookup.
- On this machine, mamba-ssm lists Linux only, and the Triton kernels behind Mamba-2 and
  Gated DeltaNet reach Windows only through a community fork whose support for the RTX 5080
  (sm_120) is recent. The dense model needs nothing beyond PyTorch.

## 3. Why not a mixture of experts

- A mixture of experts does not reduce the compute per token at a fixed active size; it
  adds parameters that need more data to train. Its wins are measured with long training
  on large data (Time-MoE on about 117 billion points, arXiv 2409.16040; Moirai-MoE, arXiv
  2410.10469). Under a short schedule dense models overtake it (arXiv 2402.07871), and at
  small scale a dense model matched on total parameters beat it (arXiv 2605.13769).
- MD_FM is limited by subjects, not by tokens: about 1.2 to 1.7 billion patch tokens per
  epoch, so a compute-optimal 200M run is only a few epochs. No study found shows a mixture
  of experts helping at thousands of subjects and tens of thousands of hours against a
  dense baseline of the same study.
- A learned router can pick up site or subject identity, which the site-leakage gate G10
  exists to prevent. The per-modality stems of the design already route by modality, with
  no router to learn.
- The expert matrix products on sm_120 most likely run on a generic fallback in PyTorch,
  not a native kernel (inferred from PyTorch issue 160891 and PR 196075).

## 4. Where the infrastructure actually goes, and how to cut it

The design's cluster estimate (about 600 H100-hours for stages 1 and 2) corresponds to a
heavy configuration: a decoder queried at every 10 ms of every window, and gradients
through 5 unrolled refinement steps. The backbone choice moves wall time by only a few
percent at these window lengths. Four changes to how the dense model is trained cut the
cost by an order of magnitude:

1. Pack windows of different lengths into one batch with block-diagonal and typed masks,
   instead of padding every window to the largest (exact, arXiv 2107.02027). Against
   padding to the maximum the saving is up to 12 to 18 times; against bucketing by
   modality set it is smaller and unmeasured.
2. Query the body decoder at about 64 random time points per window during training
   instead of all 1,000, as Perceiver IO does (arXiv 2107.14795), plus the neighbouring
   points the accelerometer kernel needs for its second derivative. All points are kept at
   evaluation. This cuts decoder training compute about 15 times.
3. Do not backpropagate through the unrolled refinement steps; refine with stopped
   gradients, and keep the unrolled variant as an ablation.
4. Train in bf16 with PyTorch's fused scaled-dot-product attention, not the flash-attn
   package, which does not build for sm_120 (flash-attention issue 2361). FP8 is not worth
   it at width 768 on this GPU.

Measured on 2026-09-28 (scripts/bench_encoder.py, reports/bench/bench_encoder.md; torch
2.11.0+cu128 on the RTX 5080): the 12 x 768 encoder (85M parameters) trains at 24.9 useful
TFLOPS, 22 percent of the nominal peak, when windows are packed into rows of 4,096 tokens,
because the fused attention kernels still compute the masked pairs (3.8 percent of the pairs
are attended); with windows bucketed by length (rows of 128, 640 and 1,536 tokens) it trains
at 36.4 TFLOPS, 32 percent of the nominal peak, about 68,000 useful tokens per second. The
cuDNN attention backend is the fastest under the boolean typed mask, the flash backend
refuses the mask, and FlexAttention does not run on this Windows build. With the typed mask,
the gradient of body outputs with respect to nuisance and neural inputs is exactly zero.
Bucketing, not long packed rows, is therefore the batching of the lean configuration.

Estimated effect (inferred at 35 percent utilization, which the bucketed measurement
approaches): a 200M model in this lean
configuration takes about 6.6 to 9 hours per epoch on the RTX 5080, about 12 to 16 hours per
epoch with 5 refinement steps on the sampled queries, and the 60M pilot about a third of that. Its
model states take about 3.2 GB of the 16 GB. The pilot, the backbone ablation, the G12
baseline and a single 200M run all fit locally. The cluster (owner decision OD5) is needed
only if the heavy configuration proves necessary for G1 or G3, or to run three seeds of
every 200M arm in reasonable calendar time.

## 5. Per component

| component | choice |
|---|---|
| tokenizers | per-modality 1D convolutional stems as designed, with layer or group normalization and never batch normalization, so no statistic is shared across records; 0.5 s patches as an ablation |
| local 10 s encoder | dense bidirectional pre-norm transformer, 12 layers, width 768, continuous-time rotary positions, fused attention with one boolean typed mask per packed batch |
| session aggregator | the only place recurrence is justified: a GRU or a small gated linear recurrence in plain PyTorch over window summaries, trained with truncated backpropagation |
| body decoder and flow head | cross-attention time-query decoder, about 64 sampled queries per window in training |
| scalp-EEG encoder | small dense transformer over channel and time tokens; start at 5M to 10M parameters and grow to 25M only if linear probes improve; criss-cross channel and time attention (CBraMod, arXiv 2412.07236) as an ablation |
| long recordings (capture24, seizeit2) | no long-context backbone: 10 s windows, hours capped per subject and per source in every epoch, context carried by the session aggregator only |

The dense model also makes the one-way rule provable: with the typed mask, the gradient of
every body output with respect to every nuisance and neural input must be exactly zero, a
single autograd test that becomes part of G17.

## 6. The ablation that settles it on our data

Pre-registered inside G12 and G13, at 60M active parameters on the RTX 5080, equal data,
equal compute and wall time logged, three seeds, identical tokenizers, kernels, decoder,
sampler and refinement schedule:
- A, the dense transformer;
- B, a channel-time factorized transformer;
- C, a Gated DeltaNet or Mamba-2 hybrid, three linear layers per attention layer, with a
  two engineer-day build limit on Windows and sm_120 (failing to build is recorded as a
  result);
- D, mixture-of-experts feed-forward layers (8 experts, top 2) against the dense model
  matched on active parameters and a dense model matched on total parameters.

Primary endpoints: G1 joint error and G3 stride error, reported separately for short
single-sensor windows and full multimodal windows; secondary: the site-leakage number of
G10. The dense model stays unless an arm beats it beyond the three-seed interval on a
primary endpoint without losing on G10.

## 7. Risks that matter more than the backbone

- The 35 percent utilization is an assumption; every wall time above scales with it.
- The typed mask may push fused attention onto a slower kernel; benchmark the three
  attention backends in week 1 and keep a plain boolean-mask path as the reference.
- The data, not the architecture, may stop G13: comparable published encoders sit at 4M
  to 85M parameters and several plateau there. 60M is the likely final size.
- capture24 and seizeit2 dominate the hours; without per-source and per-subject sampling
  caps the backbone learns wrist accelerometry and two-channel EEG.
- Biosignal foundation models encode subject identity strongly (arXiv 2606.06647); the
  leakage probe of every gate matters more than the backbone choice.
- No matched attention, state-space and mixture-of-experts comparison exists for motion or
  IMU pose pretraining; the recommendation extrapolates from EEG, time series and language,
  which is why section 6 pre-registers the test.
