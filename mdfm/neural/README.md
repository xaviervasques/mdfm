# mdfm.neural: neural statistics and descriptors

Status: DRAFT. The library implements the statistical conventions and the descriptor formulas of the neural rule
file docs/rules/NEURAL_RULES_DRAFT.md (itself a DRAFT, not hashed). Every argument that fills an OPEN item of that
file is explicit, and its default is a `Draft(value, "OPEN-nn", note)`; every result carries a `draft` tuple naming
the draft defaults it used ("q (OPEN-58)"), so an output computed on a draft value says so. Passing a value removes
its entry. When the owner settles an OPEN item, the Draft default is replaced by the settled value (or kept with its
flag removed) in a dated change.

Tested on synthetic signals only. No function reads a record, a manifest or a file; the caller passes arrays. No
GPU, no torch: numpy and scipy of the mdfm environment.

## Contents

| function | rule | OPEN items (draft default) |
|---|---|---|
| stats.phipson_smyth_p | NR-1.7 p = (1 + b) / (k + 1), smallest p 1 / (k + 1); NR-1.8 the 20-null floor (`enters_pass_decision`) | sidedness has no default: each test states 'greater' or 'two-sided' (OPEN-38, OPEN-49 for the primaries) |
| stats.draw_null_indices | NR-1.7 nulls drawn without replacement when not enumerated; NR-5.7 | OPEN-33 draw count (999) |
| stats.holm | NR-7.3 Holm step-down, adjusted p in Wright's form max over l <= i of min(1, (m - l + 1) p(l)); NR-7.4 not-run primaries enter with p = 1 (the caller's job) | OPEN-01 alpha (0.05) |
| stats.benjamini_hochberg, benjamini_yekutieli, fdr | NR-8.2 step-up within a family; BY at q / c(m), c(m) = sum 1 / i | OPEN-58 q (0.05) and method (BH) |
| stats.n_cluster_resamples, enumerate_cluster_resamples | NR-1.5 C(2n - 1, n) resamples (35 for 4 patients), weight n! / prod(c_i!) / n^n | none |
| stats.cluster_bootstrap, cluster_mean, weighted_quantile | NR-1.5, NR-1.6: resample the unit of independence; all resamples enumerated with their weights when C(2n - 1, n) is not larger than the replicate count, drawn otherwise; weighted percentile interval (inverted CDF) | OPEN-04 replicates (2000) and method (percentile, the only one implemented); OPEN-01 level (0.95) |
| stats.autocorrelation, tau_int, tau_int_pair | NR-5.6 (a): tau_int of each series alone, never of a cross-correlation; 'draft' dt (1 + 2 sum rho) or 'sokal' dt (1/2 + sum rho); Sokal's automatic window M >= c tau(M) | OPEN-31 convention ('draft'), window constant (6, stated in Sokal's convention so both conventions share one window and differ by exactly 2), series ('max' of neural and body), autocorrelation estimator ('unbiased'; the register does not name it, proposed as an addition to OPEN-31) |
| stats.s_min, k_shift, admissible_shifts, k_swap, k_stride | NR-5.7: s_min = ceil(3 tau / dt) (at least 1), k_shift = N - 2 s_min + 1, k_swap = R_s - 1, k_stride = n! - 1 | OPEN-33 spacing of distinct shifts (one sample) |
| descriptors.BANDS, HOP_S, MIN_WINDOW_S | NR-3.1 bands, NR-3.4 hop and window minima | OPEN-15 band edges (closed) via `band_bins` |
| descriptors.welch_psd | spectral estimator of NR-3.3 | OPEN-17 segment (2 s), overlap (0.5), taper (Hann) |
| descriptors.spectral_fit, aperiodic_model, gaussian_sum | NR-3.3 aperiodic plus Gaussian peaks in semilog space; peak center, power above the aperiodic fit (log10 and dB = 10 log10), bandwidth = 2 SD; offset and exponent (and knee) from the same fit | OPEN-17 fit range (1 to 45 Hz; it does not cover the gamma band of NR-3.1, 60 to 90 Hz, so a gamma descriptor needs a fit range of its own, OPEN-17 with the gamma window of OPEN-18), mode (fixed), width limits (1 to 12 Hz), maximum peaks (6), minimum height (0.1 log10, 1 dB), threshold (2 SD), edge and overlap drops (1 and 0.75 SD), final joint fit (yes) |
| descriptors.band_periodic_power | NR-3.3 band summary; a band not wholly inside the fit range is refused (ValueError), under both summaries | OPEN-17 summary (mean of the flattened spectrum over the band) and the value written when no peak is fitted (0 dB) |
| descriptors.band_envelope, detect_bursts | NR-3.6 rate per minute, duration in ms, native-rate envelope | OPEN-19 envelope (Hilbert of a zero-phase Butterworth band-pass, order 4, no smoothing), threshold (75th percentile of the record, or a given value), minimum duration (0.1 s, a run of exactly that length kept), edge bursts (excluded), edge guard (a run starting or ending within 3 cycles of the band's lower edge of the record start or end counts as cut) |
| descriptors.erd_percent, erd | NR-3.7 ERD = 100 (P_event - P_baseline) / P_baseline, intertrial power | OPEN-20 baseline window (-2 to -1 s from onset), filter order (4; not named in the register); the event window has no default |
| descriptors.cross_spectra, coherence_null_limit, phase_slope_lag, lag_direction | NR-3.8 lag from the phase slope, weights 1 / var(phase), var = (1 / (2 n_seg)) (1 / R^2 - 1) (Halliday et al. 1995 eq. 6.7); positive lag: the neural series leads; the bins kept need not be contiguous, and their phases are put on branches by a global lag search over one alias period, not by unwrapping along frequency; a lag is identified modulo the segment length (`alias_period_ms`); NR-5.2 direction only when lag +/- z SE excludes 0 | OPEN-21 segment (1 s, disjoint), taper (none), fit (through the origin), slope frequencies (band bins whose coherence exceeds its null limit at 0.05, at least 3); OPEN-01 level of the direction rule (0.95) |
| descriptors.bandwidth_refusal, bandwidth_statement | NR-3.14 refused when f_hi >= fs / 2 or f_hi above the declared low-pass corner | OPEN-23 undeclared corner (refuse) |
| descriptors.stimulation_exclusion, frequency_mask | NR-3.16 the stimulation frequency is required (0 when off); a band around half of it removed; refused when nothing remains | OPEN-24 half-width (2 Hz), multiples (one half), mode (exclusion; the entrainment model is not implemented) |
| descriptors.DescriptorValue, band_descriptor, descriptor_rows, SCHEMA_FIELDS | NR-3.5 every value carries an interval, a status (measured, prior-only, refused), a bandwidth statement, the window in s and the frequency resolution in Hz; a value without them is not written; refused values carry no value and a reason; NR-3.16 a gamma value carries its stimulation frequency | none |

## Choices the tests forced

- Phase-slope taper. With a Hann taper adjacent frequency bins are correlated, while the weighted fit of NR-3.8
  treats them as independent: the lag errors then spread wider than the SE says (tests/test_neural_descriptors.py,
  test_hann_taper_makes_the_se_run_low). The draft taper is none (boxcar), under which the errors of a planted
  delay fall within 2 SE at the nominal rate (test_phase_slope_se_is_calibrated_under_the_draft_taper).
- Coherence floor of the slope fit. Eq. 6.7 is a large-sample approximation; on bins whose coherence is near zero it
  understates the phase variance (a uniform phase has variance pi^2 / 3), and without a floor a direction is claimed
  on most uncoupled pairs (test_phase_slope_without_coupling_rarely_claims_a_direction). The draft lets a bin in only
  where the coherence exceeds its null limit 1 - alpha^(1 / (n_seg - 1)); with fewer than 3 such bins the lag is
  not estimable (NaN, no direction). Near that floor the SE still runs low (test_se_runs_low_near_the_coherence_null,
  pinned as a known limit): a lag and its direction are read only where coupling is established by its null
  (NR-5.6).
- Minimum peak height. At 0 the fit takes noise maxima of a peak-free spectrum as peaks; the draft is 0.1 log10
  (1 dB) (test_peak_free_spectrum_gives_no_peak).
- Phase branches across a coherence gap. The coherence floor leaves out the bins of a band where the coupling dips,
  so the bins fitted need not be contiguous, and unwrapping along frequency across such a gap returned a wrong lag
  with a tiny SE. The phases are put on their branches by a global search of the lag over one alias period
  (test_phase_slope_across_a_coherence_gap).
- Origin fit and a constant phase offset. With a body series of opposite sign (a sensor mounted the other way) the
  origin fit misses the delay with a tiny SE, flagged only by a chi2_dof far above 1, while the intercept fit
  returns it (test_sign_inverted_body_series). A point for OPEN-21 (origin or intercept); a caller of the draft
  origin fit reads chi2_dof.
- Burst edge guard. The band-pass and Hilbert edge artefacts pull the envelope of a burst cut by the record end
  under the threshold before the last sample, so a rule on the edge sample alone counts that burst
  (test_short_and_edge_bursts_are_excluded); the draft guard of 3 cycles of the band's lower edge excludes the cut
  bursts planted in test_bursts_cut_by_either_record_edge_are_excluded.

## Sources

The formulas follow the text of the rule file and its citations (Phipson and Smyth 2010; Holm 1979 and Wright 1992;
Benjamini and Hochberg 1995, Benjamini and Yekutieli 2001; Sokal 1997; Donoghue et al. 2020; Tinkhauser et al.
2017; Pfurtscheller and Lopes da Silva 1999; Halliday et al. 1995 and Rosenberg et al. 1989). The rule file states
that those citations were written from memory (NR-0.7) and are checked before the hash; this library adds no
citation of its own. The spectral fit is written from the method's description with scipy (nothing installed); it
is not a port of a reference package and was not compared with one.

## Tests

    <conda>/envs/mdfm/python.exe -m pytest -q -p no:cacheprovider tests/test_neural_stats.py tests/test_neural_descriptors.py

Run with CUDA_VISIBLE_DEVICES=-1 and OMP_NUM_THREADS=2; every input is synthetic and seeded.
