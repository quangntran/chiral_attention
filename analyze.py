import matplotlib.pyplot as plt
import pandas as pd
from argparse import ArgumentParser, Namespace
import os
import numpy as np

def add_args(parser: ArgumentParser):
    # General arguments
    parser.add_argument('--prediction_data_path', type=str, help='Path the the folder that contains the csv files that contains targets and predictions. Inside this folder, the two csv files are required: preds_on_val.csv and preds_on_train.csv')
    
    parser.add_argument('--output_path', type=str, help='Directory to save the residual plot to')


def parse_train_args() -> Namespace:
    """
    Parses arguments for training (includes modifying/validating arguments).
    :return: A Namespace containing the parsed, modified, and validated args.
    """
    parser = ArgumentParser()
    add_args(parser)
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_train_args()
    # val predictions
    csv_path = os.path.join(args.prediction_data_path, 'preds_on_val.csv')
    df = pd.read_csv(csv_path)
    residual = np.array(df['label']) - np.array(df['prediction'])
    plt.hist(residual)
    
    
    
