import torch
import torch.nn as nn
from model import UNET
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import DataPairsDataset
import constants
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train_fn(loader, model, optimizer, loss_fn, scaler):
  loop = tqdm(loader)
  for batch_idx, (data, targets) in enumerate(loop):
    data = data.to(device = DEVICE)
    targets = targets.float().unsqueeze(1).to(device = DEVICE)

    # calculate the loss
    with torch.autocast(device_type=DEVICE):
      predictions = model(data)
      loss = loss_fn(predictions, targets)

    # backwards step
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()


    loop.set_postfix(loss = loss.item())

def main():
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
  train_dataset = DataPairsDataset("train", transform=train_transform)
  train_loader = DataLoader(
    train_dataset,
    batch_size = constants.BATCH_SIZE,
    num_workers = constants.NUM_WORKERS,
    pin_memory = True,
    shuffle = True
    )


  model = UNET(in_channels = 1, out_channels = 1).to(DEVICE)
  loss_fn = nn.BCEWithLogitsLoss()
  optimizer = optim.Adam(model.parameters(), lr = constants.LEARNING_RATE)
  scaler = torch.cuda.amp.GradScaler()

  for epoch in range(constants.NUM_EPOCHS):
    train_fn(train_loader, model, optimizer, loss_fn, scaler)


if __name__ == "__main__":
    main()