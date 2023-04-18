# DexArt: Benchmarking Generalizable Dexterous Manipulation with Articulated Objects

[[Project Page]](https://www.chenbao.tech/dexart/) [[Paper]](https://www.chenbao.tech/dexart/static/paper/dexart.pdf)
-----

[DexArt: Benchmarking Generalizable Dexterous Manipulation with Articulated Objects](https://www.chenbao.tech/dexart/), 
Chen Bao*, Helin Xu*, Yuzhe Qin, Xiaolong Wang, CVPR 2023.


DexArt is a novel benchmark and pipeline for learning multiple dexterous manipulation tasks.
This repo contains the **simulated environment** code for DexArt.
The learning part for DexArt will release soon.

![DexArt Teaser](docs/teaser.png)

## Bibtex

```
@inproceedings{
    anonymous2023dexart,
    title={DexArt: Benchmarking Generalizable Dexterous Manipulation with Articulated Objects},
    author={Anonymous},
    booktitle={Conference on Computer Vision and Pattern Recognition 2023},
    year={2023},
    url={https://openreview.net/forum?id=v-KQONFyeKp}
}
```

## Installation

1. Clone the repo and Create a conda env with all the Python dependencies.

```bash
git clone git@github.com:Kami-code/dexart-sim.git
cd dexart-sim
conda create --name dexart python=3.8
conda activate dexart
pip install -e .
```

2. Download the demonstrations from
the [Google Drive](https://drive.google.com/file/d/1JdReXZjMaqMO0HkZQ4YMiU2wTdGCgum1/view?usp=sharing) and place 
the `asset` directory at the project root directory.

## File Structure
The file structure is listed as follows:

`dexart/env/`: environments

`examples/`: example code to try DexArt

`assets/`: tasks annotations, object and robot URDFs


## Quick Start

### Random Action Example


```bash
cd examples
python random_action.py --task_name=laptop
```

You can also try different task_name: faucet, laptop, bucket, toilet

### Visualize Point Cloud Observation Example

```bash
cd examples
python visualize_observation.py --task_name=laptop
```
You can also try different task_name: faucet, laptop, bucket, toilet
