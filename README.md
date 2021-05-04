# An Attention Graph Neural Network for Stereo-active Molecules

## Authors
- Sang T. Truong, DePauw University
- Quang N. Tran, Minerva Schools at KGI

## Usage
To run the optimal model, run the following command: 
```
python train.py --data_path 'data/d4_docking/d4_docking.csv' --split_path 'data/d4_docking/full/split0.npy'  --log_dir 'log' --checkpoint_dir 'checkpoint' --n_epochs 100 --batch_size 64 --warmup_epochs 0 --gnn_type gcn --hidden_size 32 --depth 2 --dropout 0 --message tetra_permute_concat --n_layers 2 --attn_type gat --gat_act leakyrelu --gat_depth 2 --heads 8 --concat
```

To test:
```
python stereonet/test.py --no_shuffle --data_path 'stereonet/data/d4_docking/d4_docking.csv' --split_path 'stereonet/data/d4_docking/full/split0.npy'  --log_dir 'gdrive/My Drive/Colab Notebooks/gcnn/log-s9' --checkpoint_dir 'gdrive/My Drive/Colab Notebooks/gcnn/log-s9' --model_path 'gdrive/My Drive/Colab Notebooks/gcnn/log-s9/best_model' --eval_output_dir 'gdrive/My Drive/Colab Notebooks/gcnn/log-s9/viz_best_model' --n_epochs 100 --batch_size 64 --warmup_epochs 2 --gnn_type gin --hidden_size 32 --depth 2 --dropout 0 --message tetra_permute_concat --n_layers 2 --attn_type gat --gat_act leakyrelu --gat_depth 2 --heads 3
```
where
* `--no_shuffle`: does not shuffle the dataset
* `--model_path` is the path to the model to load (in the example above, it's the `best_model`-- but it could be, for example, `'gdrive/My Drive/Colab Notebooks/gcnn/log-s9/1_model'`)
* `--eval_output_dir` is where the outputs are stored. These include: the predictions (and the targets), the visualization results
* The other parameters should be kept the same as training parameters (`gnn_type`, `hidden_size`, etc.)

To do residual analysis:
```
python stereonet/analyze.py --prediction_data_path 'gdrive/My Drive/Colab Notebooks/gcnn/log-s9/viz_best_model' --output_path 'gdrive/My Drive/Colab Notebooks/gcnn/log-s9/viz_best_model'
```

## Dependency
Reinstall PyTorch version:
```
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```
Install PyTorch Geometric:
```
pip install kora
# Add this in a Google Colab cell to install the correct version of Pytorch Geometric.
import torch

def format_pytorch_version(version):
  return version.split('+')[0]

TORCH_version = torch.__version__
TORCH = format_pytorch_version(TORCH_version)

def format_cuda_version(version):
  return 'cu' + version.replace('.', '')

CUDA_version = torch.version.cuda
CUDA = format_cuda_version(CUDA_version)
print(TORCH, CUDA)
pip install torch-scatter     -f https://pytorch-geometric.com/whl/torch-{TORCH}+{CUDA}.html
pip install torch-sparse      -f https://pytorch-geometric.com/whl/torch-{TORCH}+{CUDA}.html
pip install torch-cluster     -f https://pytorch-geometric.com/whl/torch-{TORCH}+{CUDA}.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-{TORCH}+{CUDA}.html
pip install torch-geometric 
```

