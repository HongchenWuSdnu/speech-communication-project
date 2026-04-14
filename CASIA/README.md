# CASIA

This folder contains preprocessing, split-generation, training, and summarization scripts for the CASIA experiments in our speech emotion recognition project.

## Dataset

The experiments in this folder are based on the **CASIA Chinese emotional speech corpus** released by the **Institute of Automation, Chinese Academy of Sciences**.

Please obtain the raw dataset from its **official source** and make sure your use complies with the corresponding license or terms of use.  
This repository does **not** redistribute the raw audio files.

## Experimental setting

The CASIA experiments are conducted under a **leave-one-speaker-out (LOSO)** protocol.

## Files

- `bulid_casia_v2_npys.py`  
  Builds processed NumPy files for CASIA experiments.

- `make_casia_loso_splits.py`  
  Generates LOSO splits for CASIA.

- `make_casia_metadata.py`  
  Prepares metadata used in preprocessing and experiments.

- `run_casia_loso_all.sh`  
  Shell script for running CASIA LOSO experiments.

- `run_main_ablation_v2_ext_clear.py`  
  Main ablation / experiment script for CASIA.

- `summarize_casia_loso.py`  
  Summarizes CASIA LOSO results.

## Notes

- Raw data paths should be configured locally before running the scripts.
- Generated features, checkpoints, logs, and intermediate files are not included in this repository by default.
- Please avoid uploading raw dataset files, checkpoints, or large intermediate files to GitHub.

## Related project goal

These scripts support generalization experiments for robust speech emotion recognition, with a focus on mitigating speaker shortcuts through hidden-state augmentation and speaker-adversarial training.
