export CUDA_VISIBLE_DEVICES=0
python main.py --cfg configs/LRGB-tuned/peptides-func.yaml --repeat 3 wandb.use False

export CUDA_VISIBLE_DEVICES=0
python main.py --cfg configs/LRGB-tuned/peptides-struct.yaml --repeat 3 wandb.use False

export CUDA_VISIBLE_DEVICES=0
python main.py --cfg configs/LRGB-tuned/cifar10.yaml --repeat 2 seed 0 gnn.layers_mp 10 gnn.dropout 0.2 wandb.use False

export CUDA_VISIBLE_DEVICES=0
python main.py --cfg configs/LRGB-tuned/mnist.yaml --repeat 2 seed 0 gnn.layers_mp 9 gnn.dropout 0.2 wandb.use False
