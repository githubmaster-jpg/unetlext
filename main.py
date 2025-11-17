from args import get_args
import os
import pandas as pd
import torch
from dataset import Knee_dataset
from torch.utils.data import DataLoader
from model import UNetLext
from trainer import train_model
from evaluate import plotter, eval


def main():
    args = get_args()
    # step 1: read files
    train_set = pd.read_csv(os.path.join('/Users/tiia-/Downloads/cnn/Tibia/data/csv/train.csv'))
    val_set = pd.read_csv(os.path.join('/Users/tiia-/Downloads/cnn/Tibia/data/csv/val.csv')) #args.csv_dir , 'val.csv'
    test_set = pd.read_csv(os.path.join('/Users/tiia-/Downloads/cnn/Tibia/data/csv/test.csv'))

    # step 2: preparing out dataset
    train_dataset = Knee_dataset(train_set)
    val_dataset = Knee_dataset(val_set)
    test_dataset = Knee_dataset(test_set)

    # step 3: initializing the dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # initializing the model
    model = UNetLext(
        input_channels=1,
        output_channels=1,
        pretrained=False,
        path_pretrained='',
        restore_weights=False,
        path_weights=''
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    train_data = train_model(model, train_loader, val_loader)
    
    plotter(train_data)
    eval(model, test_loader, device)

if __name__ == '__main__':
    main()