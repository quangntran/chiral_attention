# An Attention Graph Neural Network for Stereo-active Molecules

## Authors
- Sang T. Truong, DePauw University
- Quang N. Tran, Minerva Schools at KGI

## Usage
To run the optimal model, run the following command: 
```
python train.py --data_path 'data/d4_docking/d4_docking.csv' --split_path 'data/d4_docking/full/split0.npy'  --log_dir 'log' --checkpoint_dir 'checkpoint' --n_epochs 100 --batch_size 64 --warmup_epochs 0 --gnn_type gcn --hidden_size 32 --depth 2 --dropout 0 --message tetra_permute_concat --n_layers 2 --attn_type gat --gat_act leakyrelu --gat_depth 2 --heads 8 --concat
```
## Dependency
```
!pip install kora
!pip install -q torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.0+cu101.html
!pip install -q torch-sparse -f https://pytorch-geometric.com/whl/torch-1.7.0+cu101.html
!pip install -q torch-geometric
```
