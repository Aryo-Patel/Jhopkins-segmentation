import boto3
import constants
import io
from PIL import Image

def get_raw_image(image_name):
    s3 = boto3.client("s3")
    response = s3.get_object(Bucket = constants.BUCKET_NAME, Key = constants.RAW_BASE_PATH + image_name)
    content = response["Body"].read()
    image_file = io.BytesIO(content)
    image = Image.open(image_file)
    return image