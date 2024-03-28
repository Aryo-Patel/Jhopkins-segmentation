# global imports
from torch.utils.data import Dataset
import boto3
import os
import pickle
from PIL import Image
import numpy as np
import webdataset as wds

# local imports
import constants
import utils

class DataPairsDataset(Dataset):
  def __init__(self, train_test, transform=None):
    self.train_test = train_test
    self.transform = transform

    self.images = []
    s3 = boto3.resource("s3")
    bucket = s3.Bucket(constants.BUCKET_NAME)
    for obj in bucket.objects.filter(Prefix = f"Data/{self.train_test}/"):
        self.images.append(obj.key.split("/")[-1])

    self.images_container_path = f"{constants.RAW_BASE_PATH}"


  def __len__(self):
    return len(self.images)

  def __getitem__(self, idx):
    s3 = boto3.client("s3")
    response = s3.get_object(Bucket = constants.BUCKET_NAME, Key = f"Data/{self.train_test}/{self.images[idx]}")
    pickled_data = pickle.loads(response["Body"].read())

    brightfield_path = pickled_data[1]
    brightfield_image = np.array(utils.get_raw_image(brightfield_path))


    mask = pickled_data[2]
    mask[mask == 26113] = 1.0

    if self.transform is not None:
      augmentations = self.transform(image = brightfield_image, mask = mask)
      brightfield_image = augmentations["image"]
      mask = augmentations["mask"]
    return brightfield_image, mask

# TODO: Faster S3 --> Pytorch integration
# bucket_path = "s3://cindy-profiling/Data/train/"
trainset = wds.WebDataset("./_cache/train.tar").decode()
for obj in trainset:
  print(obj)
  break