# An Attention Graph Neural Network for Stereo-active Molecules

## Authors
- Sang T. Truong, DePauw University
- Quang N. Tran, Minerva Schools at KGI

## Usage
To run the optimal model, run the following command: 
```
python main.py --task regression --data_path DATAPATH --split_path PATH_TO_SPLIT_DATA  --log_dir LOG_PATH --checkpoint_dir CHECKPOINT_PATH --n-epochs 100 --batch_size 64 --warmup_epochs --gnn_type gcn --hidden_size 32 --depth 2 --dropout 0 --message tetra_permute_concat --n_layers 2 --attn_type gat --gat_act leakyrelu --gat_depth 2 --heads 8 --concat
```
## Dependency
```
!pip install kora
!pip install -q torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.0+cu101.html
!pip install -q torch-sparse -f https://pytorch-geometric.com/whl/torch-1.7.0+cu101.html
!pip install -q torch-geometric
```
