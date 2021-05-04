import os, math, torch, kora.install.rdkit, pandas as pd
from model.parsing import parse_train_args
from data_model.data import construct_loader
from util import Standardizer, create_logger, get_loss_func
from model.main import GNN
import csv
import numpy as np
from model.training import *


args = parse_train_args()
torch.manual_seed(args.seed)


train_loader, val_loader = construct_loader(args)
mean = train_loader.dataset.mean
std = train_loader.dataset.std
stdzer = Standardizer(mean, std, args.task)
loss = get_loss_func(args)


# load best model
model = GNN(args, train_loader.dataset.num_node_features, train_loader.dataset.num_edge_features).to(args.device)
print('Model architecture: ', model)
state_dict = torch.load(os.path.join(args.log_dir, 'best_model'), map_location=args.device)
model.load_state_dict(state_dict)


# predict on train data
train_ys, train_preds, train_loss, train_acc, train_auc = test(model, train_loader, loss, stdzer, args.device, args.task, viz_dir=None)

train_residual = train_ys - train_preds
plt.plot(train_residual)
plt.savefig(args.log_dir)



