# Machine Learning for Efficient Filtering of In-Detector Bad Images at PETRA III

Summer student project (DESY / PETRA IV–III beamline environment, 2024).
Real serial crystallography detectors can randomly produce unusable frames
(e.g. after fast-shutter mis-triggers). This project filters those bad images
with a small neural network so they can be detected → tagged → removed before
hitting the disk data pipeline, reducing data volume in advance.

## Data

Raw frames are EuXFEL/PETRA-style HDF5 files, e.g.
`G24 : Automatic, siO2_PPG_fastshutter, 3 files containing ~4000 images each`,
stored under `.../entry/data/data` (514×1030 detector images). The notebook also
uses KMeans-clustered labels produced with the average-per-pixel values.

Loading uses the tick framework's HDF5 codecs (`hdf5plugin`), so data can
be read by both `h5py` (file-level average) and `torch.utils.data.Dataset`
(training).

## Repository Structure

```
├── src/
│   └── local_module.py     # dataset/utils: file finder, H5Dataset, mask transform,
│                           # average-transform, vector_distance, min-max norm
├── scripts/
│   ├── avg_n.py            # loop over all HDF5 files, mask, compute mean per image,
│                           # write per-file h5 output
│   └── modeling.py         # baseline: 2-component GaussianMixture on averaged values
├── notebooks/
│   └── Pytorch_MLP_weight.ipynb  # full experiment: loaders, model, weight strategy,
│                                  # training loop, evaluation
└── requirements.txt
```

## Approach

Three stages were explored (in order of the repo history):

1. **Average-value per image** — `scripts/avg_n.py`
   Prepares per-image mean values, computed after multiplying each frame by the
   detector mask (`Det_029_mask.h5`, removing bad pixels).
2. **Baseline filtering** — `scripts/modeling.py`
   Fit a 2-component `GaussianMixture` on the normalized average values as a
   simple statistical baseline for "good vs bad".
3. **MLP with class weights** — `notebooks/Pytorch_MLP_weight.ipynb`
   The real approach: an MLP that takes a whole 514×1030 frame and returns
   good/bad, with class weights because ~98% of data are good frames:

   `weight = total_images / (n_classes × images_in_class_i)`

   Turning off class weighting is reported to drop recall for the rare class
   badly (see notebook cell on manual weights).

Because almost all data are "good", training loss is misleading — hence the
confusion matrix / classification report cells at the end of the notebook.

## Running

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1           # source .venv/bin/activate on Linux/macOS
pip install -r requirements.txt

python scripts/avg_n.py                # batch average-value extraction
python scripts/modeling.py             # statistical filter baseline
jupyter lab notebooks/Pytorch_MLP_weight.ipynb
```

## Data layout expected

The scripts/notebook were written against a specific beam-time directory
tree (Linux host):

- `.../siO2_PPG_fastshutter*data*.h5 raw frames + .joblib label files`
- `.../masks/Det_029_mask.h5` detector bad-pixel mask (group `blemish`)
- `.../KMeans_model/Labels/` labels used for training

Update the constants at the top of each script/notebook to your local paths.

## Status

- Early exploration after the 2024 summer project — not pip-installable as a package.
- Experimental: hard-coded paths, no tests, no CLI.

## Acknowledgement

Built as a Summer Student project at DESY (PETRA III/EuXFEL detectors context).
