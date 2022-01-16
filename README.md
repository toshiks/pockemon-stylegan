<div align="center">

# Your Project Name

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>

</div>

## Description

Stylegan2-ada approach based on pytorch lightning. [Original code](https://github.com/rosinality/stylegan2-pytorch). 

With this approach you can train network to generate pokemons. For that you should download [pokemon dataset](https://www.kaggle.com/kvpratama/pokemon-images-dataset).  

<center>
<img alt="Example" src="static/example.png" width="521" height="521">
</center>
<center>
<em>Generation example. The training took approximately 18 hours.</em>
</center>

## Dependencies

* CUDA 10.2
* All requirements

Install dependencies

```bash
# clone project
git clone https://github.com/toshiks/pockemon-stylegan.git
cd pockemon-stylegan

# [OPTIONAL] create conda environment
conda create -n myenv python=3.8
conda activate myenv

# install requirements
pip install -r requirements.txt
```

## How to run

Prepare dataset

```bash
python prepare_data.py --out data/lmdb --n_worker N_WORKER --size 128 DATASET_PATH
```

Train model with default configuration

```bash
# train on CPU
python run.py experiment=example_simple.yaml trainer.gpus=0

# train on GPU (two gpus with at least 8GB memory)
python run.py experiment=example_simple.yaml
```