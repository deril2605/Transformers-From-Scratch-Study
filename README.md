# Transformers From Scratch

This repo is a study-and-build project for training a small GPT-style language model from scratch, then fine-tuning it on a supervised Q&A dataset.

There are two parallel ways to follow the project:

- the original notebook workflow for learning and experimentation
- the Azure Container Apps job projects for long-running GPU training

## Main notebook

The core study notebook is:

- [notebooks/gpt-from-scratch.ipynb](C:/Users/deril/OneDrive/Desktop/Deril/Development/Transformers-From-Scratch/notebooks/gpt-from-scratch.ipynb)

For a markdown-first walkthrough of the notebook explanations, use:

- [notebooks/README.md](C:/Users/deril/OneDrive/Desktop/Deril/Development/Transformers-From-Scratch/notebooks/README.md)

## ACA projects

Pretraining lives in:

- [aca_pretraining](C:/Users/deril/OneDrive/Desktop/Deril/Development/Transformers-From-Scratch/aca_pretraining)

Important pretraining files:

- [pretraining.py](C:/Users/deril/OneDrive/Desktop/Deril/Development/Transformers-From-Scratch/aca_pretraining/pretraining.py)
- [create-aca-gpu-job.ipynb](C:/Users/deril/OneDrive/Desktop/Deril/Development/Transformers-From-Scratch/aca_pretraining/create-aca-gpu-job.ipynb)
- [download_latest_blob_run.ipynb](C:/Users/deril/OneDrive/Desktop/Deril/Development/Transformers-From-Scratch/aca_pretraining/download_latest_blob_run.ipynb)
- [README.md](C:/Users/deril/OneDrive/Desktop/Deril/Development/Transformers-From-Scratch/aca_pretraining/README.md)

Supervised fine-tuning lives in:

- [aca_sft](C:/Users/deril/OneDrive/Desktop/Deril/Development/Transformers-From-Scratch/aca_sft)

Important SFT files:

- [sft.py](C:/Users/deril/OneDrive/Desktop/Deril/Development/Transformers-From-Scratch/aca_sft/sft.py)
- [create-aca-sft-job.ipynb](C:/Users/deril/OneDrive/Desktop/Deril/Development/Transformers-From-Scratch/aca_sft/create-aca-sft-job.ipynb)
- [download_latest_sft_blob_run.ipynb](C:/Users/deril/OneDrive/Desktop/Deril/Development/Transformers-From-Scratch/aca_sft/download_latest_sft_blob_run.ipynb)
- [README.md](C:/Users/deril/OneDrive/Desktop/Deril/Development/Transformers-From-Scratch/aca_sft/README.md)

## Current workflow

The repo is currently set up to support this flow:

1. Train the base GPT model with the notebook or the ACA pretraining job.
2. Save artifacts and logs to Azure Blob Storage.
3. Download the latest pretraining run locally if needed.
4. Run supervised fine-tuning as a separate ACA GPU job.
5. Download the latest SFT run for notebook inference, plotting, and evaluation.

## Where to start

If you want the learning path, start here:

- [notebooks/gpt-from-scratch.ipynb](C:/Users/deril/OneDrive/Desktop/Deril/Development/Transformers-From-Scratch/notebooks/gpt-from-scratch.ipynb)

If you want the production-style GPU batch path, start here:

- [aca_pretraining/README.md](C:/Users/deril/OneDrive/Desktop/Deril/Development/Transformers-From-Scratch/aca_pretraining/README.md)
- [aca_sft/README.md](C:/Users/deril/OneDrive/Desktop/Deril/Development/Transformers-From-Scratch/aca_sft/README.md)
