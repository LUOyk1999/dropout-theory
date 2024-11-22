export CUDA_VISIBLE_DEVICES=0
python main.py --cfg configs/LRGB-tuned/peptides-func-dropout02.yaml --repeat 3 wandb.use False

export CUDA_VISIBLE_DEVICES=0
python main.py --cfg configs/LRGB-tuned/peptides-struct-dropout02.yaml --repeat 3 wandb.use False

export CUDA_VISIBLE_DEVICES=0
python main.py --cfg configs/LRGB-tuned/peptides-func-dropout00.yaml --repeat 3 wandb.use False

export CUDA_VISIBLE_DEVICES=0
python main.py --cfg configs/LRGB-tuned/peptides-struct-dropout00.yaml --repeat 3 wandb.use False