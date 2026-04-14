# IEMOCAP

This folder contains preprocessing and summarization scripts for the IEMOCAP experiments in our speech emotion recognition project.

## Dataset

The experiments in this folder are based on the **IEMOCAP (Interactive Emotional Dyadic Motion Capture)** database.

Please obtain the raw dataset from its **official source** and make sure your use complies with the corresponding license or terms of use.  
This repository does **not** redistribute the raw audio files.

## Experimental setting

The IEMOCAP experiments are conducted under a **session-level leave-one-session-out (LOSO)** protocol.

## Files

- `prepare_iemocap_v2_and_splits.py`  
  Prepares IEMOCAP data and generates the session-level splits used in the experiments.

- `summarize_iemocap_loso.py`  
  Summarizes the IEMOCAP LOSO experiment results.

## Notes

- Raw data paths should be configured locally before running the scripts.
- Generated features, checkpoints, logs, and intermediate files are not included in this repository by default.
- Please avoid uploading raw dataset files, checkpoints, or large intermediate files to GitHub.

## Related project goal

These scripts support generalization experiments for robust speech emotion recognition, with a focus on mitigating speaker shortcuts through hidden-state augmentation and speaker-adversarial training.
