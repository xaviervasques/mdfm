"""The G12 baselines (design docs/MFM_DESIGN_2026-09-24.md, G12 body premium; rule draft docs/rules/G12_RULE.md;
task T25): masked multimodal transformers over the body model's own tokens, with no body, trained at equal compute,
read through linear probes on their frozen tokens.

Two arms share everything but the target:
- mfr, masked-feature reconstruction (the design's baseline): the stem features (model/tokenize.py,
  ContinuousStem.features) of the masked measurement tokens predicted from the visible tokens;
- jepa, the JEPA arm of docs/proposals/JEPA_EXPLORATION_DRAFT.md section 4: the EMA target encoder's latent token of
  each masked patch predicted from the visible tokens (model/jepa.py: predictor, target, mask rule), trained on that
  loss alone.

Modules: config (the G12 configuration and the run configuration it derives from the body run's), data (the token
collate and the pose3d heading anchors), model (the model, the feature head, the two forward passes, the anchored
masks), compute (the equal-compute budget read from the body run's log), trainer (the training loop, a subclass of
train/loop.Trainer that keeps its sampler, schedule, checkpoints, provenance, guard and caps), load (a baseline
checkpoint read back for its readouts, the tokenizer and encoder only).

The reading side (task G12R): readouts (rule sections 6 and 7: frozen tokens, the probes fitted on the 54 training-fold
weargait_pd controls, the variant choice, the three G3 dev lines and their per-row predictions; it reads the G3
walkway targets) and comparison (rule section 8 on per-row predictions: the separation, the comparator, the person
bootstrap of the difference of seed means, NOT EVALUATED while the rule's preconditions are missing). No module of
the training side (config, data, model, compute, trainer) imports them.

The tokenizer and the encoder are model/tokenize.Tokenizer and model/encoder.Encoder, unchanged, built in the order
mfm.MFM builds them, so that under one seed the baseline starts from the body model's initial tokenizer and encoder.
Nothing on the training side imports mdfm.gates, reads a label, a lockbox, EVAL or CODY record, or changes a module the
stage-1 loop imports.
"""
