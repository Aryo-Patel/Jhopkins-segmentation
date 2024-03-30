import datetime
import io
import pickle

import boto3
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import constants
from dataset import trainset, valset
from model import UNET
from utils import compute_dice_and_accuracy, save_results_to_s3

if constants.ON_ARM:
    DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
else:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_fn(loader, model, optimizer, loss_fn, scaler, epoch, save_object):
    loop = tqdm(loader)
    predictions = None
    targets = None
    data = None
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)

        data = data.unsqueeze(1)

        targets = targets.float().unsqueeze(1).to(device=DEVICE)
        targets_mask = (targets > 0).float().to(device=DEVICE)

        # calculate the loss
        if constants.ON_ARM:
            predictions = model(data)
            loss = loss_fn(predictions, targets_mask)
        else:
            with torch.autocast(device_type=DEVICE):
                predictions = model(data)
                loss = loss_fn(predictions, targets_mask)

        # backwards step
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if constants.SAVE_STATS:
            # saving the training loss
            save_results_to_s3(
                save_object["running_training_loss"],
                f"{save_object['save_path']}/loss.pickle",
            )

        loop.set_postfix(loss=loss.item())
    return {"data": data, "predictions": predictions, "targets": targets}


def main():
    time = datetime.datetime.now().strftime("%m-%d_%H:%M:%S")
    save_path = constants.RUNS_BASE_PATH + time
    save_object = {
        "save_path": save_path,
        "running_training_loss": [],
        "running_training_accuracy": [],
        "running_training_dice": [],
        "running_val_accuracy": [],
        "running_val_dice": [],
    }

    train_loader = DataLoader(
        trainset.batched(constants.BATCH_SIZE),
        num_workers=constants.NUM_WORKERS,
        batch_size=None,
    )
    val_loader = DataLoader(
        valset.batched(constants.BATCH_SIZE),
        num_workers=constants.NUM_WORKERS,
        batch_size=None,
    )

    model = UNET(in_channels=1, out_channels=1, features=constants.FEATURES).to(DEVICE)
    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=constants.LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(constants.NUM_EPOCHS):
        last_batch_dict = train_fn(
            train_loader, model, optimizer, loss_fn, scaler, epoch, save_object
        )

        # Save visual progress of files, save accuracy and training loss
        if constants.SAVE_STATS:
            stat_string = ""
            # save iamge generation
            for name, value in last_batch_dict.items():
                save_results_to_s3(value, f"{save_path}/{epoch}/{name}.pt")

            # save model
            save_results_to_s3(model.state_dict(), f"{save_path}/{epoch}/model.pt")

            train_stats = compute_dice_and_accuracy(
                model, train_loader, constants.DEVICE
            )
            val_stats = compute_dice_and_accuracy(model, val_loader, constants.DEVICE)

            for metric, value in train_stats.items():
                save_object[f"running_training_{metric}"].append(value)
                save_results_to_s3(
                    save_object[f"running_training_{metric}"],
                    f"{save_path}/train_{metric}.pkl",
                )
                stat_string += f"Train {metric}: {value}\n"

            for metric, value in val_stats.items():
                save_object[f"running_val_{metric}"].append(value)
                save_results_to_s3(
                    save_object[f"running_val_{metric}"],
                    f"{save_path}/val_{metric}.pkl",
                )
                stat_string += f"Test {metric}: {value}\n"

            print(f"Epoch {epoch} statistics: \n{stat_string}")


if __name__ == "__main__":
    main()
