# Detecting-chromatin-state-alterations-in-PBMCs-associated-with-T2DM
This repository contains the codes and notebooks necessary to reproduce results reported in the Publication titled "Detecting chromatin state alterations in PBMCs associated with Type 2 Diabetes Mellitus."



<img width="960" height="540" alt="github_t2dm_coverpage" src="https://github.com/user-attachments/assets/daaaaf18-eec3-4e5d-90e3-bf38cec9268f" />

## Project Overview
Chrometrics delivers an end‑to‑end workflow for decoding chromatin organization in PBMC nuclei captured under compression and control conditions. The repo combines a segmentation notebook that generates label volumes and QC material, an installable NMCO feature engine (`nmco/` plus `measure_nmco_features.py`), and analysis notebooks that compare hand-crafted descriptors with latent spaces learned by VAEs across both full and independent patient cohorts.

## Requirements
Install everything with `pip install -r requirements.txt`:
- `numpy>=1.18.5`, `pandas>=1.1.2`, `matplotlib>=3.3.2`
- `opencv-python>=4.4.0.42`, `tifffile>=2020.10.1`, `scikit-image>=0.17.2`
- `scipy>=1.5.2`, `scikit-learn>=0.23.2`, `tqdm>=4.50.0`

## Notebook Tour

### `nuclear_segmentation.ipynb`
- Builds 3D binary masks for every `dapi*.tiff` stack via adaptive multi-Otsu thresholding, slice-wise hole filling, and 3×3×3 median filtering before volume-based cleanup (`utilities.filter_regions_by_volume`).
- Prunes elongated/tiny detections (>100 px effective length or outside 800–20 000 voxels) to keep well-formed nuclei ready for NMCO measurements.
- Generates max-intensity QC panels pairing raw and labeled volumes, rescales the anisotropic z-axis, and saves Napari screenshots for both images and labels to `../screenshots`.
- Produces segmentation masks (`bw_dapi*.tiff`) that drive all downstream notebooks and CLI tools.

### `Train_VAE_get_Embedding.ipynb`
- Loads the curated NMCO feature table (`nmco_features_filtered_with_qc_3rd_july_2024.csv`), converts nuclei into uniform 128×128 maximum-projection patches grouped by pathology, and builds PyTorch datasets with balanced pathology/patient splits (`Train_VAE_get_Embedding.ipynb:2`).
- Defines and trains a convolutional VAE (MSE reconstruction loss, configurable epochs/checkpoints) while logging training curves and saving state dict snapshots for later reuse (`Train_VAE_get_Embedding.ipynb:6`).
- Runs inference to capture latent vectors, reconstructions, and metadata for both train and held-out partitions; exports them as `latent_list*.pkl`, `recon_list*.pkl`, `pathalogy*.pkl`, and `patient_id*.pkl` so analysis notebooks can consume a consistent embedding space (`Train_VAE_get_Embedding.ipynb:1101`).

### `Embedding_analysis_on_hand_crafted_features_full.ipynb`
- Loads the NMCO feature matrix (`nmco_features_filtered_jumbled_26th_aug.csv`), removes unstable descriptors, scales numeric columns, and visualizes correlations/Wasserstein distances to understand redundancy (`Embedding_analysis_on_hand_crafted_features_full.ipynb:1`).
- Balances nuclei per patient/disease, runs Leiden clustering on the feature manifold, and maps clusters to interpretable chromatin states while keeping consensus with VAE labels.
- Builds patient-by-cluster enrichment tables, plots disease-specific cluster usage, and uses igraph to visualize patient similarity graphs.
- Trains Random Forest classifiers with permutation tests to benchmark pathology prediction, reports feature importances, and uses ANOVA/statannotations to tie cluster frequencies back to clinical covariates (`cohort_details.csv`).

### `Embedding_analysis_on_hand_crafted_features_Independent_Dataset.ipynb`
- Splits non-hypertension patients into disjoint train/test cohorts, scales features identically to the full dataset, and assigns independent nuclei to the prelearned Leiden states via k-NN (`Embedding_analysis_on_hand_crafted_features_Independent_Dataset.ipynb:1`).
- Aggregates cluster frequencies per patient in the hold-out cohort and runs leave-one-out plus one-vs-rest Random Forest classifiers to quantify generalization.
- Outputs per-patient prediction tables, ROC/AUC metrics for binary comparisons, and mirrors the enrichment plots used on the main cohort.

### `Embedding_analysis_on_VAE_features_full.ipynb`
- Starts from latent embeddings saved by the VAE notebook (`latent_list_df_lab`), balances nuclei per patient, and applies Leiden clustering plus custom palettes to label nine stable chromatin states (C1–C9) (`Embedding_analysis_on_VAE_features_full.ipynb:1`).
- Saves cluster assignments (`vae_leiden_csv*`), plots UMAP/2D projections with patient and disease overlays, and constructs cluster–patient bipartite graphs to highlight morphometric programs shared across subjects.
- Builds enrichment matrices, runs multi-class and binary classifiers with permutation-based nulls, and summarizes confusion matrices plus AUROC to show that VAE embeddings retain diagnostic signal while linking metadata from `cohort_details.csv`.

## Data & Supporting Assets
- `chrometric_feature_description.csv` documents every NMCO descriptor.
- Multiple `nmco_features_df*.csv` and `latent_list*.pkl` snapshots capture preprocessing runs (balanced, patient-filtered, independent cohort) so notebook experiments can be reproduced quickly.
- `example_data/` plus the filtered CSVs provide minimal inputs for verifying the segmentation + feature pipeline before scaling to full cohorts.

## Repro Notes
1. Install dependencies (`pip install -r requirements.txt`) and ensure the repo root is on `PYTHONPATH` so `nmco` imports succeed.
2. Run `nuclear_segmentation.ipynb` to refresh label volumes, execute `Train_VAE_get_Embedding.ipynb` when you need updated embeddings, then launch the analysis notebooks corresponding to handcrafted vs. VAE experiments.
3. Typical order: segmentation → feature extraction / VAE training → hand-crafted analysis (full + independent) → VAE embedding analysis; rerun whichever stage aligns with your experiment.

