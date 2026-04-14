# CREMA-D

This folder contains preprocessing, training, and summarization scripts for the CREMA-D experiments in our speech emotion recognition project.

## Dataset

The experiments in this folder are based on the **CREMA-D (Crowd-sourced Emotional Multimodal Actors Dataset)** corpus.

Please obtain the raw dataset from its **official source** and make sure your use complies with the dataset's license or terms of use.  
This repository does **not** redistribute the raw audio files.

## Files

- `prepare_cremad_features.py`  
  Extracts handcrafted acoustic features for CREMA-D.

- `prepare_cremad_temporal.py`  
  Prepares temporal / deep-feature-related inputs for CREMA-D.

- `run_ablation_chain_AF.py`  
  Runs ablation experiments for the AF configuration chain.

- `run_ablation_chain_Csat_noCB.py`  
  Runs the SAT_noCB-related ablation chain.

- `run_ablation_chain_seed.py`  
  Runs repeated experiments across random seeds.

- `run_cremad_baseline_BC.py`  
  Runs baseline experiments on CREMA-D.

- `run_deep_only_with_spkhead_v2.py`  
  Runs the deep-only setting with speaker-head analysis.

- `run_main_ablation_v2_ext_clear.py`  
  Main CREMA-D ablation script used in the project.

- `summarize_baseline_BC_v2.py`  
  Summarizes baseline results.

- `summarize_main_v2_ext_clear.py`  
  Summarizes the main ablation results.

## Notes

- Raw data paths should be configured locally before running the scripts.
- Generated features, checkpoints, and result files are not included in this repository by default.
- Please avoid uploading raw dataset files, model checkpoints, or large intermediate `.npy` files to GitHub.

## Related project goal

These scripts support experiments on robust speech emotion recognition, with a focus on mitigating speaker shortcuts through hidden-state augmentation and speaker-adversarial training.
