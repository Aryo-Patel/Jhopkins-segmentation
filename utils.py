import io

import boto3
import torch
from PIL import Image

import constants


def get_raw_image(image_name):
    s3 = boto3.client("s3")
    response = s3.get_object(
        Bucket=constants.BUCKET_NAME, Key=constants.RAW_BASE_PATH + image_name
    )
    content = response["Body"].read()
    image_file = io.BytesIO(content)
    image = Image.open(image_file)
    return image


def save_results_to_s3(content, save_path):
    s3 = boto3.client("s3")
    buffer = io.BytesIO()
    torch.save(content, buffer)
    s3.put_object(
        Bucket=constants.BUCKET_NAME,
        Key=save_path,
        Body=buffer.getvalue(),
    )


def compute_dice_and_accuracy(model, loader, device):
    true_positives = 0
    num_samples_seen = 0
    num_correct = 0
    model.eval()

    with torch.no_grad():
        for data, targets in loader:
            data = data.to(device)
            targets = targets.to(device)

            data = data.unsqueeze(1)
            targets = targets.float().unsqueeze(1)
            targets_mask = (targets > 0).float().to(device)

            predictions = model(data)
            predictions = (predictions > 0.5).float().to(device)

            true_positives += torch.logical_and(targets_mask, predictions).sum()
            num_samples_seen += targets_mask.numel() + predictions.numel()
            num_correct += (targets_mask == predictions).sum()

    dice_score = (2 * true_positives) / num_samples_seen
    accuracy = num_correct / (num_samples_seen // 2)
    return {"dice": dice_score.item(), "accuracy": accuracy.item()}
