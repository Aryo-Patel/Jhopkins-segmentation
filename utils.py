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


def weighted_bce(predictions, target_mask, weights):
    """
    Weights should be ordered for the 0 valued samples and then for the 1-valued samples
    """
    # weight of 0's then weight of 1's
    loss = weights[0] * (1 - target_mask) * torch.log(1 - predictions) + weights[
        1
    ] * target_mask * torch.log(
        predictions
    )  # this is the most disgusting auto-formatting in my life
    return torch.neg(torch.mean(loss))


def ftversky(predictions, target_mask):
    a = 0.3
    b = 0.7
    gamma = 1 / 0.75
    epsilon = 1e-8

    num_same = (predictions * target_mask).sum() + epsilon
    false_positives = ((1 - target_mask) * target_mask).sum()
    false_negatives = (target_mask * (1 - predictions)).sum()
    tversky = num_same / (num_same + a * false_positives + b * false_negatives)

    return (1 - tversky) ** (gamma)


if __name__ == "__main__":
    save_results_to_s3("test", "test/test.pkl")
