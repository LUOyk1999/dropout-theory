# cora

python main.py --gnn gcn --dataset cora --lr 0.001 --local_layers 3  --hidden_channels 512 --weight_decay 5e-4 --dropout 0.7 --rand_split_class --valid_num 500 --test_num 1000 --seed 123 --device $1 --runs 5
python main.py --gnn sage --dataset cora --lr 0.001 --local_layers 3  --hidden_channels 256 --weight_decay 5e-4 --dropout 0.7 --rand_split_class --valid_num 500 --test_num 1000 --seed 123 --device $1 --runs 5
python main.py --gnn gat --dataset cora --lr 0.001 --local_layers 3  --hidden_channels 512 --weight_decay 5e-4 --dropout 0.2 --rand_split_class --valid_num 500 --test_num 1000 --seed 123 --device $1 --runs 5 --res

# citeseer

python main.py --gnn gcn --dataset citeseer --lr 0.001 --local_layers 2 --hidden_channels 512 --weight_decay 0.01 --dropout 0.5 --rand_split_class --valid_num 500 --test_num 1000 --seed 123 --device $1 --runs 5
python main.py --gnn sage --dataset citeseer --lr 0.001 --local_layers 3 --hidden_channels 512 --weight_decay 0.01 --dropout 0.4 --rand_split_class --valid_num 500 --test_num 1000 --seed 123 --device $1 --runs 5
python main.py --gnn gat --dataset citeseer --lr 0.001 --local_layers 3 --hidden_channels 256 --weight_decay 0.01 --dropout 0.5 --rand_split_class --valid_num 500 --test_num 1000 --seed 123 --device $1 --runs 5

# pubmed

python main.py --gnn gcn --dataset pubmed --lr 0.005 --local_layers 2 --hidden_channels 512 --weight_decay 5e-4 --dropout 0.6 --rand_split_class --valid_num 500 --test_num 1000 --seed 123 --device $1 --runs 5
python main.py --gnn sage --dataset pubmed --lr 0.005 --local_layers 4 --hidden_channels 512 --weight_decay 5e-4 --dropout 0.7 --rand_split_class --valid_num 500 --test_num 1000 --seed 123 --device $1 --runs 5
python main.py --gnn gat --dataset pubmed --lr 0.01 --local_layers 2 --hidden_channels 512 --weight_decay 5e-4 --dropout 0.5 --rand_split_class --valid_num 500 --test_num 1000 --seed 123 --device $1 --runs 5

# the other homophilic graphs

for hidden_channels in 64 128 256 512
do
for layer in 1 2 3 4 5 6 7 8 9 10 15
do
for dropout in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7
do 

python main.py --pre_linear --dataset amazon-computer --hidden_channels $hidden_channels --epochs 1000 --lr 0.001 --runs 3 --local_layers $layer --weight_decay 5e-5 --dropout $dropout --device $1 --gnn gcn --bn
python main.py --pre_linear --dataset amazon-computer --hidden_channels $hidden_channels --epochs 1000 --lr 0.001 --runs 3 --local_layers $layer --weight_decay 5e-5 --dropout $dropout --device $1 --gnn sage --bn
python main.py --pre_linear --dataset amazon-computer --hidden_channels $hidden_channels --epochs 1000 --lr 0.001 --runs 3 --local_layers $layer --weight_decay 5e-5 --dropout $dropout --device $1 --gnn gat --bn

python main.py --pre_linear --dataset amazon-photo --hidden_channels $hidden_channels --epochs 1000 --lr 0.001 --runs 3 --local_layers $layer --weight_decay 5e-5 --dropout $dropout --device $1 --gnn gcn --bn
python main.py --pre_linear --dataset amazon-photo --hidden_channels $hidden_channels --epochs 1000 --lr 0.001 --runs 3 --local_layers $layer --weight_decay 5e-5 --dropout $dropout --device $1 --gnn sage --bn
python main.py --pre_linear --dataset amazon-photo --hidden_channels $hidden_channels --epochs 1000 --lr 0.001 --runs 3 --local_layers $layer --weight_decay 5e-5 --dropout $dropout --device $1 --gnn gat --bn

python main.py --pre_linear --dataset coauthor-cs --hidden_channels $hidden_channels --epochs 1500 --lr 0.001 --runs 3 --local_layers $layer --weight_decay 5e-4 --dropout $dropout --device $1 --gnn gcn --bn
python main.py --pre_linear --dataset coauthor-cs --hidden_channels $hidden_channels --epochs 1500 --lr 0.001 --runs 3 --local_layers $layer --weight_decay 5e-4 --dropout $dropout --device $1 --gnn sage --bn
python main.py --pre_linear --dataset coauthor-cs --hidden_channels $hidden_channels --epochs 1500 --lr 0.001 --runs 3 --local_layers $layer --weight_decay 5e-4 --dropout $dropout --device $1 --gnn gat --bn

python main.py --pre_linear --dataset coauthor-physics --hidden_channels $hidden_channels --epochs 1500 --lr 0.001 --runs 3 --local_layers $layer --weight_decay 5e-4 --dropout $dropout --device $1 --gnn gcn --bn
python main.py --pre_linear --dataset coauthor-physics --hidden_channels $hidden_channels --epochs 1500 --lr 0.001 --runs 3 --local_layers $layer --weight_decay 5e-4 --dropout $dropout --device $1 --gnn sage --bn
python main.py --pre_linear --dataset coauthor-physics --hidden_channels $hidden_channels --epochs 1500 --lr 0.001 --runs 3 --local_layers $layer --weight_decay 5e-4 --dropout $dropout --device $1 --gnn gat --bn

python main.py --pre_linear --dataset wikics --hidden_channels $hidden_channels --epochs 1000 --lr 0.001 --runs 3 --local_layers $layer --weight_decay 0.0 --dropout $dropout --device $1 --gnn gcn --bn
python main.py --pre_linear --dataset wikics --hidden_channels $hidden_channels --epochs 1000 --lr 0.001 --runs 3 --local_layers $layer --weight_decay 0.0 --dropout $dropout --device $1 --gnn sage --bn
python main.py --pre_linear --dataset wikics --hidden_channels $hidden_channels --epochs 1000 --lr 0.001 --runs 3 --local_layers $layer --weight_decay 0.0 --dropout $dropout --device $1 --gnn gat --bn

done
done
done