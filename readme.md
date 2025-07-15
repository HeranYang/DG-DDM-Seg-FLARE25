```
project_root/

├── EarlyStopforpseudolabel/
├── our_method/
├── SLAug/

├── eval/

├── inputs/

├── preprocess_training/
├── preprocess_validation/

├── fortrain_pseudolabel.py
├── eval_del.m
├── predict.sh
├── requirements.txt
├── run.sh
└── README.md
```

## Environment & Dependencies

> **CUDA ≥ 11.8 | Python ≥ 3.9**
>  To avoid version mismatches, create a fresh Conda environment with **Python 3.9 or later**.

```bash
# Create and activate a new environment (name is arbitrary)
conda create -n FLARE python=3.9
conda activate FLARE

# Install GPU‑enabled PyTorch (CUDA 11.8)
pip install torch==2.2.2+cu118 torchvision==0.17.2+cu118 \
            torchaudio==2.2.2 --extra-index-url https://download.pytorch.org/whl/cu118

# Install project‑specific dependencies
pip install -r requirements.txt
```

------

## Dataset Overview

FLARE25 provides **50 annotated CT scans** for training:

```
train_CT_gt_label/
├── imagesTr/
│   ├── FLARE22_Tr_0001_0000.nii.gz
│   └── ...
└── labelsTr/
    ├── FLARE22_Tr_0001_0000.nii.gz
    └── ...
```

Validation set (MRI) example:

```
validation/
├── MRI_imagesVal/
│   ├── amos_7296_0000.nii.gz
│   └── ...
└── MRI_labelsVal/
    ├── amos_7296.nii.gz
    └── ...
```

------

## Data Pre‑processing

> **Edit the paths in each script first, then run.** Absolute paths are recommended.

### 1. Training set

```bash
python preprocess_training.py
# Inside preprocess_training.py
root_image_path = "/abs/path/to/imagesTr"
root_label_path = "/abs/path/to/labelsTr"
save_path  = "/abs/path/"
```

### 2. Validation set

```bash
python preprocess_validation.py
# Inside preprocess_validation.py
MRI_IMG_DIR = "/abs/path/to/MRI_imagesVal"
MRI_LBL_DIR = "/abs/path/to/MRI_labelsVal"
MRI_OUT_DIR = "/abs/path/"
```

------

## Path Configuration

Before any training or inference, replace **all** dataset placeholders with **absolute paths**:

| Scenario                                   | Files to Edit                                           | Key Field(s)                                                 |
| ------------------------------------------ | ------------------------------------------------------- | ------------------------------------------------------------ |
| **EarlyStopforpseudolabel** (train & test) | `configs/xxx/xxx_train.json``configs/xxx/xxx_test.json` | `datasets.dataroot`                                          |
| **our_method** (train)                     | `config/xxx/xxx_train.json`                             | `datasets.train.dataroot``datasets.train.cond_pesudolabel_sequence` |
| **SLAug** (validate)                       | `dataloaders/AbdominalDataset.py`                       | `BASEDIR`                                                    |
| **our_method** (validate)                  | `config/xxx/xxx_test_mp.json`                           | `datasets.train.dataroot``datasets.train.cond_pesudolabel_sequence` |

------

## Full Training Work

### Stage 1 — Pseudo‑label Generation

```bash
cd EarlyStopforpseudolabel

# 1 · Train 10 base models
chmod +x train10.sh
./train10.sh

# 2 · Save pseudo‑labels
python sr_test.py -p val -c config/SR3_EffNet_test.json

# 3 · Select the best model via MATLAB script
eval_del.m

# 4 · Generate multiple pseudo‑label sets
python fortrain_pesudolabel.py
```

### Stage 2 — Model Training

```bash
cd our_method
python our_train.py -p train -c config/our_train.json
```

### Stage 3 — Validation & Prediction

```bash
cd SLAug
# Train model to generate pseudo‑label for validation
python main.py --base configs/efficientUnet_CHAOS_to_SABSCT_mine.yaml --seed 23
```

> **Quick test**
>  Place raw validation data in:
>
> ```
> ./inputs/MRI_imagesVal
> ./inputs/MRI_labelsVal
> ```
>
> Specify model weights in the corresponding config, then simply run:
>
> ```bash
> ./predict.sh   # includes pre‑processing, inference, post‑processing
> ```

------

## Docker

To run the inference using Docker, use the following command:

> Note: This is the official inference script. When running predictions, please replace `input_dir` and `output_dir` with your own input and output directories in `run.sh`. The input MRI images must be in `.nii.gz` format.

Place the data to be tested in the `./FLARE_Test/` directory and then run the program.

```
./run.sh
```

Docker Container download link [HuggingFace](https://huggingface.co/hryang/DG-DDM-Seg-FLARE25/).