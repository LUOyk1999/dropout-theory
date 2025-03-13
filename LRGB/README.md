# Graph Classification

## Background

Based on the GPS and LRGB-tuned codebase: https://github.com/rampasek/GraphGPS and https://github.com/toenshoff/LRGB

## Python environment setup with Conda

```bash
conda create -n graphgps python=3.10
conda activate graphgps

conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
pip install torch_geometric==2.3.0
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu117.html

# RDKit is required for OGB-LSC PCQM4Mv2 and datasets derived from it.  
conda install openbabel fsspec rdkit -c conda-forge

pip install pytorch-lightning yacs torchmetrics
pip install performer-pytorch
pip install tensorboardX
pip install ogb
pip install wandb

conda clean --all
```


## Running Training
```bash
conda activate graphgps

python main.py --cfg configs/LRGB-tuned/peptides-func.yaml --repeat 3 wandb.use False

python main.py --cfg configs/LRGB-tuned/peptides-struct.yaml --repeat 3 wandb.use False

python main.py --cfg configs/LRGB-tuned/cifar10.yaml --repeat 2 seed 0 gnn.layers_mp 10 gnn.dropout 0.2 wandb.use False

python main.py --cfg configs/LRGB-tuned/mnist.yaml --repeat 2 seed 0 gnn.layers_mp 9 gnn.dropout 0.2 wandb.use False
```

