import os, pdb, sys
import numpy as np
import pandas as pd
import argparse as ap
import matplotlib.pyplot as plt

from PIL import Image
from matplotlib import cm
from diffusers import StableDiffusionImageVariationPipeline
from torchvision import transforms

import tensorflow as tf
import tensorflow_datasets as tfds

tf2_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(tf2_dir)
import data_utils

# data_utils.preprocess_image(image, height, width, is_training=False, color_jitter_strength=0., test_crop=True)
DEVICE = "cuda:0"
NUM_IMAGES=2

def generate_image_variation(img, sd_pipe, tform):
    inp = tform(img).to(DEVICE).unsqueeze(0)
    out = sd_pipe(inp, guidance_scale=3.0, num_inference_steps=50, num_images_per_prompt=NUM_IMAGES)
    return out["images"]

def main(dataset, batch_num, **kwargs):
    output_dir = kwargs.get("output_dir", "image_variations")

    tform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(
            (224, 224),
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=False,
            ),
        transforms.Normalize(
        [0.48145466, 0.4578275, 0.40821073],
        [0.26862954, 0.26130258, 0.27577711]),
    ])

    sd_pipe = StableDiffusionImageVariationPipeline.from_pretrained(
    "lambdalabs/sd-image-variations-diffusers",
    revision="v2.0"
    )
    sd_pipe = sd_pipe.to(DEVICE)
    sd_pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))

    ds, ds_info = tfds.load(name=dataset, split='train', with_info=True)
    ds = ds.map(data_utils.preprocess_image)

    # This is line is temporary.
    # It needs to be changed to get the images for the batch equivalent to the batch_num provided.
    df = tfds.as_dataframe(ds.take(5), ds_info)
    for i in df.index.values:
        img = Image.fromarray(np.uint8(df.loc[i, 'image']*255))
        output_image_list = generate_image_variation(img, sd_pipe, tform)
        # I think we will save the image variations to tensorflow records
        # with the original image and all relevant metadata.


if __name__ == '__main__':
    parser = ap.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("batch_num")
    parser.add_argument("--output_dir", required=False, default="image_variations")

    args, _ = parser.parse_known_args()
    kwargs = dict(args._get_kwargs())
    main(**kwargs)