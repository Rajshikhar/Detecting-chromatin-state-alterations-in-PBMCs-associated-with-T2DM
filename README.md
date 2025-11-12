# Detecting-chromatin-state-alterations-in-PBMCs-associated-with-T2DM
This repository contains the codes and notebooks necessary to reproduce results reported in the Publication titled "Detecting chromatin state alterations in PBMCs associated with Type 2 Diabetes Mellitus."



<img width="960" height="540" alt="github_t2dm_coverpage" src="https://github.com/user-attachments/assets/daaaaf18-eec3-4e5d-90e3-bf38cec9268f" />

## Project Overview
Chrometrics packages the tooling we use to characterize chromatin organization (NMCO) from fluorescent PBMC images collected under compressed vs. control conditions. The repository couples a reusable nuclear segmentation workflow with feature extraction utilities (`nmco/`) and analysis notebooks that compare handcrafted NMCO descriptors with latent representations learned by VAEs.

## Repository Highlights
- `nuclear_segmentation.ipynb` – parent notebook that segments DAPI stacks, filters nuclei by 3D morphology, and produces QC-ready visualizations.
- `nmco/` & `measure_nmco_features.py` – installable module and CLI that turn segmented nuclei + raw intensities into interpretable morphometric, intensity-distribution, and texture features.
- `Embedding_analysis_on_*` notebooks – downstream studies that benchmark handcrafted NMCO tables against VAE embeddings on both full and independent cohorts.
- `example_data/`, `chrometric_feature_description.csv`, and multiple `*_features*.csv/pkl` files – curated references for reproducing analyses or training classifiers.

## Nuclear Segmentation Workflow
1. **Data ingestion** – loads all `dapi*.tiff` z-stacks from `../stacks`, keeping bookkeeping arrays for filenames and downstream QC.
2. **Adaptive multi-Otsu thresholding** – computes `skimage.filters.threshold_multiotsu` per stack; chooses the darker/lighter bound automatically to handle bright vs. dim acquisitions.
3. **Binary cleanup** – fills voids slice-wise (`ndi.binary_fill_holes`), denoises by a 3×3×3 median filter, labels 3D components, and removes objects outside `[800, 20000]` voxels or with elongated bounding boxes (>100 px effective length) via `utilities.filter_regions_by_volume`.
4. **Segmentation export** – writes masked volumes back to `../stacks` as `bw_dapi*.tiff`, preserving original naming to keep raw/label pairs aligned.
5. **Quality control** – builds a randomized categorical colormap (jet-derived) so each label renders distinctly, rescales anisotropic axes (×4 in z) for max-intensity projections, and plots raw vs. labeled overlays.
6. **Napari review** – spins up Napari in 3D, applies consistent camera angles/zoom, and saves paired screenshots (`*_img_screenshot.png`, `*_label_screenshot.png`) in `../screenshots` for audit trails or figure panels.

The notebook is intentionally linear so you can tweak thresholds, kernel sizes, or volume limits without touching other cells.

## Running the Notebook
### Requirements
Python 3.9+, `numpy`, `scipy`, `scikit-image`, `tifffile`, `pandas`, `matplotlib`, `natsort`, `imageio`, and `napari`. Install via `pip install -r requirements.txt` and make sure `nmco/` is on `PYTHONPATH`.

### Quick Start
1. Place raw DAPI stacks in `../stacks` (each named `dapi*.tiff`) and create `../screenshots`.
2. Open `nuclear_segmentation.ipynb`, adjust `min_size`, `max_size`, and other hyperparameters near the top if needed.
3. Run cells sequentially; segmentation masks (`bw_dapi*.tiff`) and screenshot PNGs will be written automatically.

### Outputs
- Cleaned segmentation volumes ready for NMCO feature extraction.
- Max-projection QC figures showing raw vs. labeled nuclei.
- Napari-generated PNGs that capture 3D context for each patient/condition.

## Downstream Analysis
- Use `measure_nmco_features.py` or `nmco.utils.run_nuclear_feature_extraction` to build feature tables per nucleus.
- Handcrafted feature notebooks (`Embedding_analysis_on_hand_crafted_features_full.ipynb`) analyze class balance, patient-level stability, and classifier performance across treatments.
- VAE notebooks (`Embedding_analysis_on_VAE_features_*.ipynb`) compare latent spaces, assess cluster reproducibility, and fuse embeddings with clinical metadata from `cohort_details.csv`.
- Independent Dataset notebooks ( `Embedding_analysis_on_hand_crafted_features_Independent_Dataset.ipynb`) analyse the generalizability of our framework within our dataset.

## Data & Metadata
- `chrometric_feature_description.csv` documents every NMCO metric.
- `nmco_features_df*.csv` and `latent_list_*.pkl` store precomputed features/embeddings.
- `logs/`, `vae_leiden_csv*/`, and `results_para_search.csv` capture experiment outputs and parameter sweeps.

## Citation 
