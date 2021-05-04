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

def train_and_save_predictions(loader, preds_path):
    # predict on train data
    ys, preds, loss_val, acc, auc = test(model, loader, loss, stdzer, args.device, args.task, viz_dir=None)

    # save predictions
    smiles = loader.dataset.smiles
#    preds_path = os.path.join(args.log_dir, 'preds_on_train.csv')
    pd.DataFrame(list(zip(smiles, ys, preds)), columns=['smiles', 'label', 'prediction']).to_csv(preds_path, index=False)

#def visualize_and_save():
    
# predict on train data
print('Evaluation on training data')
train_ys, train_preds, train_loss, train_acc, train_auc = train_and_save_predictions(train_loader, preds_path=os.path.join(args.log_dir, 'preds_on_train.csv'))

# predict on val data
print('Evaluation on validating data')
val_ys, val_preds, val_loss, val_acc, val_auc = train_and_save_predictions(val_loader, preds_path=os.path.join(args.log_dir, 'preds_on_val.csv'))
