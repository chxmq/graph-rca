# RCAEval datasets (local cache)

Zips downloaded from https://zenodo.org/records/14590730 (RE2/RE3: the
log-bearing subsets used by experiment 10). Extracted directories are
deleted after runs to save ~36 GB; re-extract before re-running:

    cd research/data/rcaeval && for z in *.zip; do unzip -q "$z" -d "${z%.zip}"; done

Experiment results and per-case checkpoints live in
research/experiments/10_rcaeval/data/ and do not require the raw data.
