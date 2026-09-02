# Representation Learning for Real-Life PPG (RL-PPG) 🫀⌚️
### This is the official implementation for the paper: **"Take it Personally: The Limits of General SSL Representations for Real-Life PPG Emotion Detection"**

#### Authors: Dominika Kunc, Przemysław Kazienko, and Stanisław Saganowski

#### 🔗 [Preprint](https://arxiv.org/pdf/2608.14675)
## 🛠 Setup & Installation

We recommend using Conda to manage your environment.
 1. Create the Environment
 Create the environment from the provided file
```
conda env create -f environment.yml
```

 2. Activate the environment
```
conda activate rl-ppg
```

## 🚀 Running the Experiments

#### Self-Supervised pretraining


The core logic is contained within main.py. The script supports various training modes and dataset selections.

Usage Example

To run a self-supervised experiment with a specific seed and dataset:

```
python main.py --experiment_description "rl_ppg" \
               --run_description "run_1" \
               --seed 123 \
               --training_mode self_supervised_ppg \
               --selected_dataset [DATASET_NAME]
```
Note: Ensure that the `--selected_dataset` name matches the folder name in your data/ directory and has a corresponding configuration file in `config_files/`. 

#### Evaluation & Downstream Tasks

To evaluate the learned representations or run standard classification, use `classification.py`.
Please modify specific script parameters (e.g., modalities, dataset, etc.) before running the script.
Important: This must be run from the main project directory (RL-PPG).

Run classification evaluation
```
python classification.py 
```

## 📊 Results & Logs

All experiment outputs, including model checkpoints and performance metrics, are saved in the experiments_logs directory by default.

## 📦 Pre-trained Models & Representations

To support Open Science and reproducibility, we provide the pre-trained weights resulting from our experiments.

#### Pre-trained Weights

The pre-trained weights for the encoder can be found in the following directory:

```src/experiments_logs/```

These weights are stored as PyTorch checkpoints (`.ckp`).

#### Feature Extraction

If you wish to use these pre-trained encoders as fixed feature extractors for your own physiological datasets, we provide an exemplary script:

`get_representations.py`: Use this script to load a pre-trained encoder and extract embeddings from new wearable PPG data.

## 📄 License & Usage

Research & Academic Use: We highly encourage the use of this code for academic research and reproducibility. Please cite our paper if you find this work helpful.

Commercial Use: This code and the associated models are provided for non-commercial research purposes only.

## ✍️ Citation

If you use this code or our findings in your research, please consider citing (preprint):

```
@article{kunc2026take,
      title={Take it Personally: The Limits of General SSL Representations for Real-Life PPG Emotion Detection}, 
      author={Dominika Kunc and Przemysław Kazienko and Stanisław Saganowski},
      year={2026},
      eprint={2608.14675},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      conference = {ACII},

      url={https://arxiv.org/abs/2608.14675}, }
```

## 🤝 Credits & Acknowledgements

This implementation is built upon and inspired by the [TS-TCC](https://github.com/emadeldeen24/ts-tcc)  repository. We have extended the original Time-Series Temporal and Contextual Contrastive Learning framework to support:

- Extension of the encoders' architectures by adding more layers and changing activation functions.

- Implementation of new downstream tasks for wearable-based time series.
- Evaltuation in Across-Time setting.
