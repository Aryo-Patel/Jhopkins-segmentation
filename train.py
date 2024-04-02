import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import constants
from dataset import trainset, valset
from model import UNET
from utils import compute_dice_and_accuracy, ftversky, save_results_to_s3, weighted_bce

if constants.ON_ARM:
    DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
else:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_fn(data, targets, model, optimizer, loss_fn, scaler, epoch, save_object):
    # loop = tqdm(loader)
    predictions = None
    # targets = None
    # data = None
    thresh = torch.nn.Threshold(0.5, 1)
    # for batch_idx, (data, targets) in enumerate(loop):
    data = data.to(device=DEVICE)

    data = data.unsqueeze(1)

    targets = targets.float().unsqueeze(1).to(device=DEVICE)
    targets_mask = (targets > 0).float().to(device=DEVICE)

    num_positives = targets_mask.sum()
    num_total = targets_mask.numel()

    # calculate the loss
    if constants.ON_ARM:
        predictions = model(data)

        predictions_mask = thresh(predictions)
        b = 1 - num_positives / num_total
        a = 1 - b
        loss = ftversky(predictions_mask, targets_mask, a, b)
        # loss = weighted_bce(predictions, targets_mask, weights)
        # loss = loss_fn(predictions, targets_mask)
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
        save_object["running_training_loss"].append(loss.item())

    print("loss: ", loss.item())
    # loop.set_postfix(loss=loss.item())
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
        pin_memory=True,
    )
    val_loader = DataLoader(
        valset.batched(constants.BATCH_SIZE),
        num_workers=constants.NUM_WORKERS,
        batch_size=None,
        pin_memory=True,
    )

    model = UNET(in_channels=1, out_channels=1, features=constants.FEATURES).to(DEVICE)
    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=constants.LEARNING_RATE)

    scaler = torch.cuda.amp.GradScaler()

    data, targets = next(iter(train_loader))

    for epoch in range(constants.NUM_EPOCHS):
        # print("running training batch")
        last_batch_dict = train_fn(
            data,
            targets,
            model,
            optimizer,
            loss_fn,
            scaler,
            epoch,
            save_object,
        )
        # print("finished training")

        # Save visual progress of files, save accuracy and training loss
        if constants.SAVE_STATS:
            # print("saving loss")
            # saving the training loss
            save_results_to_s3(
                save_object["running_training_loss"],
                f"{save_object['save_path']}/loss.pickle",
            )
            # print("finished saving loss")
            stat_string = ""
            # save iamge generation
            # print("saving batch info ")
            for name, value in last_batch_dict.items():
                save_results_to_s3(value, f"{save_path}/{epoch}/{name}.pt")

            # print("finished batch saving")

            # save model
            # print("saving model")

            save_results_to_s3(model.state_dict(), f"{save_path}/{epoch}/model.pt")

            # print("finished model saving")

            # print("computing accuracy")
            train_stats = compute_dice_and_accuracy(
                model, train_loader, constants.DEVICE
            )
            val_stats = compute_dice_and_accuracy(model, val_loader, constants.DEVICE)
            # print("saving accuracy")
            for metric, value in train_stats.items():
                save_object[f"running_training_{metric}"].append(value)
                stat_string += f"Train {metric}: {value}\n"
                save_results_to_s3(
                    save_object[f"running_training_{metric}"],
                    f"{save_path}/train_{metric}.pkl",
                )
            # print("finished computing trainign accuracy")
            for metric, value in val_stats.items():
                save_object[f"running_val_{metric}"].append(value)
                stat_string += f"Test {metric}: {value}\n"

                save_results_to_s3(
                    save_object[f"running_val_{metric}"],
                    f"{save_path}/val_{metric}.pkl",
                )
            # print("finished saving accuracy")

            print(f"Epoch {epoch} statistics: \n{stat_string}")
    torch.save(last_batch_dict, "lbdc.pt")


if __name__ == "__main__":
    main()
