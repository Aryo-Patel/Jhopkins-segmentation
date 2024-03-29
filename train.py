import datetime
import boto3
import io
import pickle
import torch
import torch.nn as nn
from model import UNET
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import DataPairsDataset
from dataset import trainset
import constants
import albumentations as A
from albumentations.pytorch import ToTensorV2

if constants.ON_ARM:
  DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
else:
  DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train_fn(loader, model, optimizer, loss_fn, scaler, epoch, save_object):
  loop = tqdm(loader)
  for batch_idx, (data, targets) in enumerate(loop):
    data = data.to(device = DEVICE)

    data = data.unsqueeze(1)
    print("data", data.shape)

    targets = targets.float().unsqueeze(1).to(device = DEVICE)
    print("targets",targets.shape)
    targets_mask = (targets > 0).float()

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
      s3 = boto3.client("s3")

      # saving the training loss
      buffer = io.BytesIO()
      save_object["running_training_loss"].append(loss.item())
      pickle.dump(save_object["running_training_loss"], buffer)

      s3.put_object(Bucket = constants.BUCKET_NAME, Key = f"{save_object['save_path']}/loss.pickle", Body = buffer.getvalue())

      if batch_idx == len(loop) - 2:
        # check the training accuracy
        sig_preds = torch.sigmoid(predictions)


        # save the batch, results, and predictions for last iteration on each epoch
        targets_cpu = targets.cpu()
        data_cpu = data.cpu()
        predictions_cpu = predictions.cpu()

        buffer = io.BytesIO()
        pickle.dump(targets_cpu, buffer)
        s3.put_object(Bucket = constants.BUCKET_NAME, Key = f"{save_object['save_path']}/{epoch}/targets.pt", Body = buffer.getvalue())

        buffer = io.BytesIO()
        pickle.dump(data_cpu, buffer)
        s3.put_object(Bucket = constants.BUCKET_NAME, Key = f"{save_object['save_path']}/{epoch}/data.pt", Body = buffer.getvalue())

        buffer = io.BytesIO()
        pickle.dump(predictions_cpu, buffer)
        s3.put_object(Bucket = constants.BUCKET_NAME, Key = f"{save_object['save_path']}/{epoch}/predictions.pt", Body = buffer.getvalue())

        # save the model snapshot
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        s3.put_object(Bucket = constants.BUCKET_NAME, Key = f"{save_object['save_path']}/{epoch}/snapshot.pt", Body = buffer.getvalue())

    loop.set_postfix(loss = loss.item())

def main():
  time = datetime.datetime.now().strftime("%m-%d_%H:%M:%S")
  save_path = constants.RUNS_BASE_PATH + time
  save_object = {
    "save_path": save_path,
    "running_training_loss": [],
    "running_validation_loss": [],
    "running_validation_error": [],
  }
  train_transform = A.Compose(
    [
        A.RandomCrop(height = constants.IMAGE_HEIGHT, width = constants.IMAGE_WIDTH, p = 1.0),
        A.Rotate(limit = 35, p = 1.0),
        A.HorizontalFlip(p = 0.5),
        A.VerticalFlip(p = 0.1),
        A.Normalize(mean = 0, std = 1, max_pixel_value = constants.MAX_PIXEL_VALUE),
        ToTensorV2()
    ]
  )

  train_loader = DataLoader(trainset.batched(constants.BATCH_SIZE), num_workers = 4, batch_size = None)

  # train_dataset = DataPairsDataset("train", transform=train_transform)
  # train_loader = DataLoader(
  #   train_dataset,
  #   batch_size = constants.BATCH_SIZE,
  #   num_workers = constants.NUM_WORKERS,
  #   pin_memory = True,
  #   shuffle = True
  #   )


  model = UNET(in_channels = 1, out_channels = 1, features=constants.FEATURES).to(DEVICE)
  loss_fn = nn.BCELoss()
  optimizer = optim.Adam(model.parameters(), lr = constants.LEARNING_RATE)
  scaler = torch.cuda.amp.GradScaler()

  for epoch in range(constants.NUM_EPOCHS):
    train_fn(train_loader, model, optimizer, loss_fn, scaler, epoch, save_object)


if __name__ == "__main__":
  main()