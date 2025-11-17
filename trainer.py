from args import get_args
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import dice_loss_from_logits, dice_score_from_logits

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model(model, train_loader, val_loader):
    args = get_args()

    bce = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    running_loss = 0
    train_losses = []
    val_losses = []
    val_scores = []

    for epoch in range(args.epochs):
        model.train()

        for data_batch in train_loader:
            xrays = data_batch['xray'].to(device)
            masks = data_batch['mask'].to(device)

            optimizer.zero_grad()
            outputs = model(xrays)
            loss_bce = bce(outputs, masks)

            loss_dice = dice_loss_from_logits(outputs, masks)
            loss = loss_bce + loss_dice

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        val_loss, val_score = validate_model(model,val_loader, bce)
        print(f'epoch {epoch+1}/{args.epochs} |'
              f'train loss {train_loss:.4f} |'
              f'val loss {val_loss:.4f} |'
              f'val score {val_score:.4f} |')
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_scores.append(val_score)

    return [train_losses,val_losses,val_scores]

def validate_model(model, val_loader, bce):
    model.eval()
    val_loss = 0.0
    val_score = 0.0

    with torch.no_grad():
            for data_batch in val_loader:
                xrays = data_batch['xray'].to(device)
                masks = data_batch['mask'].to(device)

                outputs = model(xrays)
                loss_bce = bce(outputs, masks)

                loss_dice = dice_loss_from_logits(outputs, masks)
                loss = loss_bce + loss_dice

                loss_item = loss.item()
                val_loss += loss_item
                dice_score_item = dice_score_from_logits(outputs, masks)
                val_score += dice_score_item

            val_epoch_loss = val_loss / len(val_loader)
            val_epoch_dice = val_score / len(val_loader)

    return val_epoch_loss, val_epoch_dice
