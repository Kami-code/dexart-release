# DexArt: Benchmarking Generalizable Dexterous Manipulation with Articulated Objects

[[Project Page]](https://www.chenbao.tech/dexart/) [[arXiv]](https://arxiv.org/abs/2305.05706) [[Paper]](https://www.chenbao.tech/dexart/static/paper/dexart.pdf)
-----

[DexArt: Benchmarking Generalizable Dexterous Manipulation with Articulated Objects](https://www.chenbao.tech/dexart/), 


[Chen Bao](https://chenbao.tech)\*, [Helin Xu](https://helinxu.github.io/)\*, [Yuzhe Qin](https://yzqin.github.io/), [Xiaolong Wang](https://xiaolonw.github.io/), CVPR 2023.


DexArt is a novel benchmark and pipeline for learning multiple dexterous manipulation tasks.
This repo contains the **simulated environment** and **training code** for DexArt.

![DexArt Teaser](docs/teaser.png)

## News
**[2023.11.21]** All the RL checkpoints are available now!🎈 They are included in the assets. See [Main Results](https://github.com/Kami-code/dexart-release#main-results) to reproduce the results in the paper! <br>
**[2023.4.18]**  Code and vision pre-trained models are available now! <br>
**[2023.3.24]**  DexArt is accepted by CVPR 2023! 🎉 <br>

## Installation

1. Clone the repo and Create a conda env with all the Python dependencies.

```bash
git clone git@github.com:Kami-code/dexart-release.git
cd dexart-release
conda create --name dexart python=3.8
conda activate dexart
pip install -e .    # for simulation environment
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 -c pytorch    # for visualizing trained policy and training 
```

2. Download the assets from
the [Google Drive](https://drive.google.com/file/d/1qc-v50eTEjpkRoWsxfqExvC1P_EKSFAa/view?usp=drive_link) and place 
the `asset` directory at the project root directory.

## File Structure
The file structure is listed as follows:

`dexart/env/`: environments

`assets/`: tasks annotations, object, robot URDFs and RL checkpoints

`examples/`: example code to try DexArt

`stable_baselines3/`: RL training code modified from [stable_baselines3](https://github.com/DLR-RM/stable-baselines3)



## Quick Start

### Example of Random Action


```bash
python examples/random_action.py --task_name=laptop
```

`task_name`: name of the environment [`faucet`, `laptop`, `bucket`, `toilet`]

### Example for Visualizing Point Cloud Observation 

```bash
python examples/visualize_observation.py --task_name=laptop
```
`task_name`: name of the environment [`faucet`, `laptop`, `bucket`, `toilet`]


### Example for Visualizing Policy

```bash
python examples/visualize_policy.py --task_name=laptop --checkpoint_path assets/rl_checkpoints/laptop/laptop_nopretrain_0.zip
```

`task_name`: name of the environment [`faucet`, `laptop`, `bucket`, `toilet`]

`use_test_set`: flag to determine evaluating with seen or unseen instances

### Example for Training RL Agent

```bash
python3 examples/train.py --n 100 --workers 10 --iter 5000 --lr 0.0001 &&
--seed 100 --bs 500 --task_name laptop --extractor_name smallpn &&
--pretrain_path ./assets/vision_pretrain/laptop_smallpn_fulldata.pth 
```
`n`: the number of rollouts to be collected in a single episode

`workers`: the number of simulation progress

`iter`: the total episode number to be trained

`lr`: learning rate of RL

`seed`: seed of RL

`bs`: batch size of RL update

`task_name`: name of training environment [`faucet`, `laptop`, `bucket`, `toilet`]

`extractor_name`: different PointNet architectures [`smallpn`, `meduimpn`, `largepn`]

`pretrain_path`: path to downloaded pre-trained model. [Default: `None`]

`save_freq`: save the model every `save_freq` episodes. [Default: `1`]

`save_path`: path to save the model. [Default: `./examples`]

## Main Results
```bash
python examples/evaluate_policy.py --task_name=laptop --checkpoint_path assets/rl_checkpoints/laptop/laptop_nopretrain_0.zip --eval_per_instance 100
python examples/evaluate_policy.py --task_name=laptop --use_test_set --checkpoint_path assets/rl_checkpoints/laptop/laptop_nopretrain_0.zip --eval_per_instance 100
```

`task_name`: name of the environment [`faucet`, `laptop`, `bucket`, `toilet`]

`use_test_set`: flag to determine evaluating with seen or unseen instances

### Faucet

| Method                | Split       | Seed 0    | Seed 1    | Seed 2    | Avg               | Std               |
|-----------------------|-------------|-----------|-----------|-----------|-------------------|-------------------|
| No Pre-train          | train/test  | 0.52/0.34 | 0.00/0.00 | 0.44/0.43 | 0.32/0.26         | 0.23/0.18         |
| Segmentation on PMM   | train/test  | 0.42/0.38 | 0.25/0.15 | 0.14/0.11 | 0.27/0.21         | 0.11/0.12         |
| Classification on PMM | train/test  | 0.40/0.33 | 0.19/0.14 | 0.07/0.09 | 0.22/0.18         | 0.14/0.10         |
| Reconstruction on DAM | train/test  | 0.27/0.17 | 0.37/0.30 | 0.36/0.21 | 0.33/0.22         | 0.05/**0.05**     |
| SimSiam on DAM        | train/test  | 0.80/0.60 | 0.40/0.24 | 0.72/0.53 | 0.64/0.46         | 0.17/0.16         |
| Segmentation on DAM   | train/test  | 0.80/0.56 | 0.76/0.53 | 0.82/0.66 | **0.79**/**0.59** | **0.02**/**0.05** |

### Laptop

| Method                | Split       | Seed 0    | Seed 1    | Seed 2    | Avg               | Std               |
|-----------------------|-------------|-----------|-----------|-----------|-------------------|-------------------|
| No Pre-train          | train/test  | 0.78/0.41 | 0.78/0.31 | 0.81/0.50 | 0.79/0.41         | **0.02**/0.08     |
| Segmentation on PMM   | train/test  | 0.91/0.62 | 0.90/0.53 | 0.77/0.48 | 0.86/0.54         | 0.06/0.08         |
| Classification on PMM | train/test  | 0.96/0.51 | 0.58/0.35 | 0.96/0.62 | 0.83/0.49         | 0.18/0.11         |
| Reconstruction on DAM | train/test  | 0.85/0.56 | 0.91/0.63 | 0.80/0.43 | 0.85/0.54         | 0.05/0.08         |
| SimSiam on DAM        | train/test  | 0.84/0.59 | 0.83/0.34 | 0.89/0.51 | 0.86/0.48         | 0.03/0.10         |
| Segmentation on DAM   | train/test  | 0.89/0.57 | 0.94/0.67 | 0.89/0.58 | **0.91**/**0.60** | **0.02**/**0.04** |

### Bucket

| Method                | Split       | Seed 0    | Seed 1    | Seed 2    | Avg               | Std           |
|-----------------------|-------------|-----------|-----------|-----------|-------------------|---------------|
| No Pre-train          | train/test  | 0.36/0.55 | 0.58/0.69 | 0.52/0.49 | 0.49/0.57         | 0.09/0.08     |
| Segmentation on PMM   | train/test  | 0.62/0.62 | 0.00/0.00 | 0.40/0.41 | 0.34/0.34         | 0.26/0.26     |
| Classification on PMM | train/test  | 0.55/0.47 | 0.50/0.51 | 0.67/0.73 | 0.57/0.57         | 0.07/0.11     |
| Reconstruction on DAM | train/test  | 0.49/0.49 | 0.58/0.46 | 0.40/0.59 | 0.49/0.51         | 0.07/**0.05** |
| SimSiam on DAM        | train/test  | 0.00/0.00 | 0.53/0.38 | 0.73/0.78 | 0.42/0.39         | 0.30/0.32     |
| Segmentation on DAM   | train/test  | 0.70/0.68 | 0.70/0.74 | 0.79/0.85 | **0.73**/**0.75** | **0.04**/0.07 |

### Toilet

| Method                | Split       | Seed 0    | Seed 1    | Seed 2    | Avg               | Std               |
|-----------------------|-------------|-----------|-----------|-----------|-------------------|-------------------|
| No Pre-train          | train/test  | 0.80/0.47 | 0.75/0.51 | 0.63/0.43 | 0.72/0.47         | 0.07/0.03         |
| Segmentation on PMM   | train/test  | 0.78/0.42 | 0.62/0.46 | 0.64/0.47 | 0.68/0.45         | 0.07/0.02         |
| Classification on PMM | train/test  | 0.78/0.33 | 0.65/0.43 | 0.66/0.44 | 0.69/0.40         | 0.06/0.05         |
| Reconstruction on DAM | train/test  | 0.78/0.58 | 0.73/0.48 | 0.75/0.49 | 0.75/0.52         | 0.02/0.05         |
| SimSiam on DAM        | train/test  | 0.84/0.54 | 0.81/0.49 | 0.84/0.45 | 0.83/0.50         | **0.01**/0.04     |
| Segmentation on DAM   | train/test  | 0.86/0.54 | 0.84/0.53 | 0.86/0.56 | **0.85**/**0.54** | **0.01**/**0.01** |

## Visual Pretraining

We have uploaded the code to generate a dataset and pretrain our models in [examples/pretrain](https://github.com/Kami-code/dexart-release/tree/main/examples/pretrain). You can refer to [examples/pretrain/run.sh](https://github.com/Kami-code/dexart-release/blob/main/examples/pretrain/run.sh) for a detailed usage.

## Bibtex

```
@inproceedings{bao2023dexart,
  title={DexArt: Benchmarking Generalizable Dexterous Manipulation with Articulated Objects},
  author={Bao, Chen and Xu, Helin and Qin, Yuzhe and Wang, Xiaolong},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={21190--21200},
  year={2023}
}
```

## Acknowledgements

This repository employs the same code structure for simulation environment and training code to that used in [DexPoint](https://github.com/yzqin/dexpoint-release).
