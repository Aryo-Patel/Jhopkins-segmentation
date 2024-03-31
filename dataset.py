# global imports
import pickle

import albumentations as A
import boto3
import numpy as np
import webdataset as wds
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset

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
        for obj in bucket.objects.filter(Prefix=f"Data/{self.train_test}/"):
            self.images.append(obj.key.split("/")[-1])

        self.images_container_path = f"{constants.RAW_BASE_PATH}"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        s3 = boto3.client("s3")
        response = s3.get_object(
            Bucket=constants.BUCKET_NAME,
            Key=f"Data/{self.train_test}/{self.images[idx]}",
        )
        pickled_data = pickle.loads(response["Body"].read())

        brightfield_path = pickled_data[1]
        brightfield_image = np.array(utils.get_raw_image(brightfield_path))

        mask = pickled_data[2]
        mask[mask == 26113] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image=brightfield_image, mask=mask)
            brightfield_image = augmentations["image"]
            mask = augmentations["mask"]
        return brightfield_image, mask


# TODO: Faster S3 --> Pytorch integration
# bucket_path = "s3://cindy-profiling/Data/train/"
def identity(x):
    return x


train_transform = A.Compose(
    [
        A.RandomCrop(height=constants.IMAGE_HEIGHT, width=constants.IMAGE_WIDTH, p=1.0),
        A.Rotate(limit=35, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Normalize(mean=0, std=1, max_pixel_value=constants.MAX_PIXEL_VALUE),
        ToTensorV2(),
    ]
)


def apply_train_transformation(x):
    brightfield, mask = x[0], x[1]
    aug = train_transform(image=np.array(brightfield), mask=mask)
    return aug["image"][0], aug["mask"]


validation_transformation = A.Compose(
    [
        A.RandomCrop(height=constants.IMAGE_HEIGHT, width=constants.IMAGE_WIDTH, p=1.0),
        A.Normalize(mean=0, std=1, max_pixel_value=constants.MAX_PIXEL_VALUE),
        ToTensorV2(),
    ]
)


def apply_val_transformation(x):
    brightfield, mask = x[0], x[1]
    aug = validation_transformation(image=np.array(brightfield), mask=mask)
    return aug["image"][0], aug["mask"]


shard_names = []
for i in range(6):
    shard_names.append(f"./training_shards/shard-00000{i}.tar")
trainset = (
    wds.WebDataset(shard_names, shardshuffle=True)
    .shuffle(100)
    .decode("pil")
    .to_tuple("brightfield.pyd", "mask.pyd")
    .map(apply_train_transformation)
)

shard_names = []
for i in range(1):
    shard_names.append(f"./validation_shards/shard-00000{i}.tar")
valset = (
    wds.WebDataset(shard_names, shardshuffle=True)
    .shuffle(40)
    .decode("pil")
    .to_tuple("brightfield.pyd", "mask.pyd")
    .map(apply_val_transformation)
)
