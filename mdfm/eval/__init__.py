"""Evaluation tools shared by the gate harnesses (mdfm.gates) and the dev hooks of training: split conformal
calibration and predictive scores (conformal.py); the inference adapters of a trained checkpoint (infer.py:
checkpoint loading with its provenance and state_dict audit, bodies inferred on windows on the 100 Hz clock, and
the predictors the G1 and G3 harnesses call). Nothing here passes a lockbox or EVAL flag, and infer.py reads a
checkpoint and the windows or records it is handed, never a record by itself. This package does not import
infer.py at import time: mdfm.gates.g1 imports conformal.py, and infer.py imports mdfm.gates."""
