
## Known quirk

In the Git Bash shells of the working sessions no user variable is set, and torch._dynamo
(imported by the optimizers) then fails on the Unix module pwd while naming its cache
folder. Scripts that import torch set TORCHINDUCTOR_CACHE_DIR to cache/torchinductor in this
tree (gitignored) before the import; an Anaconda PowerShell terminal does not need it.
 `conda activate mdfm` works in any Anaconda terminal.
- Python 3.11 from conda-forge only (no defaults channel).
- torch from the PyTorch CUDA 12.8 index, the build that supports the RTX 5080 (sm_120).
- Everything else from PyPI, listed in env/requirements.txt; the exact versions installed
  are frozen in env/requirements-lock.txt after creation.
- Large artefacts (checkpoints, caches) go to <store> (the disk with the most free space),
  never into this tree.

## Create or rebuild

    conda create -n mdfm -c conda-forge --override-channels python=3.11 pip -y
    conda activate mdfm
    python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
    python -m pip install -r env/requirements.txt
    python -m pip freeze > env/requirements-lock.txt
    python scripts/check_env.py

## Use from a terminal

    conda activate mdfm
    cd <repo>
    python scripts/check_env.py

Sessions call the interpreter directly: <conda>/envs/mdfm/python.exe.

## The open body model (plan phase C, installed 2026-11-12)

The synthetic patient moves to Anny (NAVER LABS Europe, PyPI package anny; code Apache-2.0, model data CC0 under
anny/data/mpfb2), installed with the owner's approval on 2026-11-12:

    python -m pip install anny==0.6.2

It brought warp-lang 1.19.0, trimesh 5.2.0, roma 1.8.0 and safetensors 0.8.0 (env/requirements-lock.txt). Never
install the extra anny[smpl] (it pulls smplx and chumpy, the non-commercial SMPL family), never the [examples] extra
(it pulls py-soma-x and gradio, not needed), and never let Anny fetch its non-commercial archive: the SMPL and SMPL-X
topologies need it, and mdfm/synth/openbody.py refuses them (allow list, blocked modules, disarmed download).
Anny's cache is pinned to <store>/anny_cache (openbody.CACHE_DIR; the variable MDFM_ANNY_CACHE moves it), never
~/.cache/anny; the first load builds it in about 90 s (three safetensors files, 828 MB; reports/openbody/
licence_audit_fresh.json). The audit of the network and of the cache: scripts/openbody_audit.py [--fresh].
