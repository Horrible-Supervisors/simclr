import os, pdb, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from matplotlib import cm
from diffusers import StableDiffusionImageVariationPipeline
from torchvision import transforms

import tensorflow as tf
import tensorflow_datasets as tfds

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry


CROP_PROPORTION = 0.875  # Standard for ImageNet.
# CROP_PROPORTION = 0.975  # Standard for ImageNet.
HEIGHT = 224
WIDTH = 224

def _compute_crop_shape(
    image_height, image_width, aspect_ratio, crop_proportion
):
    """Compute aspect ratio-preserving shape for central crop.

    The resulting shape retains `crop_proportion` along one side and a proportion
    less than or equal to `crop_proportion` along the other side.

    Args:
    image_height: Height of image to be cropped.
    image_width: Width of image to be cropped.
    aspect_ratio: Desired aspect ratio (width / height) of output.
    crop_proportion: Proportion of image to retain along the less-cropped side.

    Returns:
    crop_height: Height of image after cropping.
    crop_width: Width of image after cropping.
    """
    image_width_float = tf.cast(image_width, tf.float32)
    image_height_float = tf.cast(image_height, tf.float32)

    def _requested_aspect_ratio_wider_than_image():
        crop_height = tf.cast(tf.math.rint(
            crop_proportion / aspect_ratio * image_width_float), tf.int32)
        crop_width = tf.cast(tf.math.rint(
            crop_proportion * image_width_float), tf.int32)
        return crop_height, crop_width

    def _image_wider_than_requested_aspect_ratio():
        crop_height = tf.cast(
            tf.math.rint(crop_proportion * image_height_float), tf.int32)
        crop_width = tf.cast(tf.math.rint(
            crop_proportion * aspect_ratio *
            image_height_float), tf.int32)
        return crop_height, crop_width

    return tf.cond(
        aspect_ratio > image_width_float / image_height_float,
        _requested_aspect_ratio_wider_than_image,
        _image_wider_than_requested_aspect_ratio)

def center_crop(image, height, width, crop_proportion):
    """Crops to center of image and rescales to desired size.

    Args:
    image: Image Tensor to crop.
    height: Height of image to be cropped.
    width: Width of image to be cropped.
    crop_proportion: Proportion of image to retain along the less-cropped side.

    Returns:
    A `height` x `width` x channels Tensor holding a central crop of `image`.
    """
    shape = tf.shape(image)
    image_height = shape[0]
    image_width = shape[1]
    crop_height, crop_width = _compute_crop_shape(
        image_height, image_width, height / width, crop_proportion)
    offset_height = ((image_height - crop_height) + 1) // 2
    offset_width = ((image_width - crop_width) + 1) // 2
    image = tf.image.crop_to_bounding_box(
        image, offset_height, offset_width, crop_height, crop_width)
    image = tf.image.resize(image, [height, width],
                            method=tf.image.ResizeMethod.BICUBIC)
    return image

def preprocess_for_eval(image, height, width):
    """Preprocesses the given image for evaluation.

    Args:
    image: `Tensor` representing an image of arbitrary size.
    height: Height of output image.
    width: Width of output image.

    Returns:
    A preprocessed image `Tensor`.
    """
    image = center_crop(image, height, width, crop_proportion=CROP_PROPORTION)
    image = tf.reshape(image, [height, width, 3])
    image = tf.clip_by_value(image, 0., 1.)
    return image

def preprocess_image(features):
    """Preprocesses the given image.

    Args:
        image: `Tensor` representing an image of arbitrary size.

    Returns:
        A preprocessed image `Tensor` of range [0, 1].
    """
    image = features["image"]
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = preprocess_for_eval(image, HEIGHT, WIDTH)
    features["image"] = image
    return features

if __name__ == '__main__':
    NUM_IMAGES = 10
    device = "cuda:0"

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
        try:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=40960)])
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)

    label = 150
    imagenet_class_dir = f"/home/jrick6/repos/lora_finetuning/instance_dir/imagenet_class_{label}"
    # ImageNet Tensorflow Dataset
    ds, ds_info = tfds.load(name='imagenet2012', split='train', with_info=True)
    ds = ds.map(preprocess_image).filter(lambda x: tf.equal(x['label'],label))
    df = tfds.as_dataframe(ds.take(5), ds_info)
    os.makedirs(imagenet_class_dir, exist_ok=True)
    for idx, row in df.iterrows():
        Image.fromarray(np.uint8(row['image'] * 255)).save(f"{imagenet_class_dir}/{row['file_name'].decode()}")
        # row['file_name'].decode()
        # row['image']
        # row['label']
    
    # fig = tfds.show_examples(ds.take(1), ds_info, rows=1, cols=1, plot_scale=7)
    # fig.savefig("test.jpg")
    pdb.set_trace()

    out_dir = f"output"

    ####### IMAGENET #######
    # i = 5
    for i in df.index.values:
        output_dir = os.path.join(out_dir, f"output_images_{i}")
        os.makedirs(output_dir, exist_ok=True)
        input_image = df.loc[i, 'image']

        im = Image.fromarray(np.uint8(input_image*255))
        # im2 = Image.open("/home/jrick6/repos/dreambooth_tutorial/dog/alvan-nee-9M0tSjb-cpA-unsplash.jpeg")
        # im = Image.fromarray(np.uint8(cm.gist_earth(input_image)*255))
        # im2 = Image.open("test.jpg")

        # Stable Diffusion Image Variation Testing
        ###################################
        sd_pipe = StableDiffusionImageVariationPipeline.from_pretrained(
        "lambdalabs/sd-image-variations-diffusers",
        revision="v2.0"
        )
        sd_pipe = sd_pipe.to(device)
        sd_pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))
        ###################################

        # Segment Anything Testing
        ###################################
        # sam = sam_model_registry["vit_b"](checkpoint="/home/jrick6/repos/simclr/segment_anything/sam_vit_b_01ec64.pth")
        # mask_generator = SamAutomaticMaskGenerator(sam)
        # masks = mask_generator.generate(np.array(im))
        ###################################

        im.save(os.path.join(output_dir, f"input_image{i}.jpg"))

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

        inp = tform(im).to(device).unsqueeze(0)
        # guidance_scale=3.0
        # num_inference_steps=50.0

        # num_inf_step_list = [50, 75, 100, 150, 200]
        num_inf_step_list = [50]
        # gs_list = [3.0, 7.5, 15.0, 25.0, 50.0, 100.0]
        # gs_list = [0.25, 0.5, 1.0, 1.5, 2.0, 2.5]
        gs_list = [3.0]
        for num_inference_steps in num_inf_step_list:
            for guidance_scale in gs_list:
                os.makedirs(f"{output_dir}/gs_{guidance_scale}_steps_{num_inference_steps}", exist_ok=True)
                out = sd_pipe(inp, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps, num_images_per_prompt=NUM_IMAGES)
                output_image_list = out["images"]
                for idx, output_image in enumerate(output_image_list):
                    output_image.save(f"{output_dir}/gs_{guidance_scale}_steps_{num_inference_steps}/output_image_{idx}_{guidance_scale}_{num_inference_steps}.jpg")

    ####### CIFAR-10-128 #######
    # i = 5
    # for i in range(10):
    #     output_dir = os.path.join(out_dir, f"output_images_cifar_128_{i}")
    #     os.makedirs(output_dir, exist_ok=True)
    #     class_dir = f"/home/jrick6/repos/simclr/cifar_upscale/cifar10-128/train/class{i}"
    #     input_image_path = f"/home/jrick6/repos/simclr/cifar_upscale/cifar10-128/train/class{i}/{os.listdir(class_dir)[0]}"
    #     input_image = Image.open(input_image_path)
    #     im = input_image # Image.fromarray(np.uint8(input_image*255))
    #     # im2 = Image.open("/home/jrick6/repos/dreambooth_tutorial/dog/alvan-nee-9M0tSjb-cpA-unsplash.jpeg")
    #     # im = Image.fromarray(np.uint8(cm.gist_earth(input_image)*255))
    #     # im2 = Image.open("test.jpg")

    #     # Stable Diffusion Image Variation Testing
    #     ###################################
    #     sd_pipe = StableDiffusionImageVariationPipeline.from_pretrained(
    #     "lambdalabs/sd-image-variations-diffusers",
    #     revision="v2.0"
    #     )
    #     sd_pipe = sd_pipe.to(device)
    #     sd_pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))
    #     ###################################

    #     # Segment Anything Testing
    #     ###################################
    #     # sam = sam_model_registry["vit_b"](checkpoint="/home/jrick6/repos/simclr/segment_anything/sam_vit_b_01ec64.pth")
    #     # mask_generator = SamAutomaticMaskGenerator(sam)
    #     # masks = mask_generator.generate(np.array(im))
    #     ###################################

    #     im.save(os.path.join(output_dir, f"input_image_{i}.jpg"))

    #     tform = transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Resize(
    #             (224, 224),
    #             interpolation=transforms.InterpolationMode.BICUBIC,
    #             antialias=False,
    #             ),
    #         transforms.Normalize(
    #         [0.48145466, 0.4578275, 0.40821073],
    #         [0.26862954, 0.26130258, 0.27577711]),
    #     ])
    #     resize_tform = transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Resize(
    #             (224, 224),
    #             interpolation=transforms.InterpolationMode.BICUBIC,
    #             antialias=False,
    #             ),
    #     ])

    #     test = resize_tform(im)
    #     test_arr = np.transpose(test.numpy(), (1,2,0))
    #     test_im = Image.fromarray(np.uint8(test_arr*255))
    #     test_im.save(os.path.join(output_dir, f"input_image_test_{i}.jpg"))

    #     inp = tform(im).to(device).unsqueeze(0)
    #     # guidance_scale=3.0
    #     # num_inference_steps=50.0

    #     # num_inf_step_list = [50, 75, 100, 150, 200]
    #     num_inf_step_list = [50]
    #     # gs_list = [3.0, 7.5, 15.0, 25.0, 50.0, 100.0]
    #     # gs_list = [0.25, 0.5, 1.0, 1.5, 2.0, 2.5]
    #     gs_list = [3.0]
    #     for num_inference_steps in num_inf_step_list:
    #         for guidance_scale in gs_list:
    #             os.makedirs(f"{output_dir}/gs_{guidance_scale}_steps_{num_inference_steps}", exist_ok=True)
    #             out = sd_pipe(inp, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps, num_images_per_prompt=NUM_IMAGES)
    #             output_image_list = out["images"]
    #             for idx, output_image in enumerate(output_image_list):
    #                 output_image.save(f"{output_dir}/gs_{guidance_scale}_steps_{num_inference_steps}/output_image_{idx}_{guidance_scale}_{num_inference_steps}.jpg")

    # cifar_ds, cifar_ds_info = tfds.load(name='cifar10', split='train', with_info=True)
    # cifar_df = tfds.as_dataframe(cifar_ds.take(10), cifar_ds_info)
    # ####### CIFAR-10 #######
    # # i = 5
    # for i in cifar_df.index.values:
    #     output_dir = os.path.join(out_dir, f"output_images_cifar_{i}")
    #     os.makedirs(output_dir, exist_ok=True)
    #     input_image = cifar_df.loc[i, 'image']

    #     im = Image.fromarray(np.uint8(input_image*255))
    #     # im2 = Image.open("/home/jrick6/repos/dreambooth_tutorial/dog/alvan-nee-9M0tSjb-cpA-unsplash.jpeg")
    #     # im = Image.fromarray(np.uint8(cm.gist_earth(input_image)*255))
    #     # im2 = Image.open("test.jpg")

    #     # Stable Diffusion Image Variation Testing
    #     ###################################
    #     sd_pipe = StableDiffusionImageVariationPipeline.from_pretrained(
    #     "lambdalabs/sd-image-variations-diffusers",
    #     revision="v2.0"
    #     )
    #     sd_pipe = sd_pipe.to(device)
    #     sd_pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))
    #     ###################################

    #     # Segment Anything Testing
    #     ###################################
    #     # sam = sam_model_registry["vit_b"](checkpoint="/home/jrick6/repos/simclr/segment_anything/sam_vit_b_01ec64.pth")
    #     # mask_generator = SamAutomaticMaskGenerator(sam)
    #     # masks = mask_generator.generate(np.array(im))
    #     ###################################

    #     im.save(os.path.join(output_dir, f"input_image_{i}.jpg"))

    #     tform = transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Resize(
    #             (224, 224),
    #             interpolation=transforms.InterpolationMode.BICUBIC,
    #             antialias=False,
    #             ),
    #         transforms.Normalize(
    #         [0.48145466, 0.4578275, 0.40821073],
    #         [0.26862954, 0.26130258, 0.27577711]),
    #     ])
    #     resize_tform = transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Resize(
    #             (224, 224),
    #             interpolation=transforms.InterpolationMode.BICUBIC,
    #             antialias=False,
    #             ),
    #     ])

    #     test = resize_tform(im)
    #     test_arr = np.transpose(test.numpy(), (1,2,0))
    #     test_im = Image.fromarray(np.uint8(test_arr*255))
    #     test_im.save(os.path.join(output_dir, f"input_image_test_{i}.jpg"))

    #     pdb.set_trace()
    #     inp = tform(im).to(device).unsqueeze(0)
    #     # guidance_scale=3.0
    #     # num_inference_steps=50.0

    #     # num_inf_step_list = [50, 75, 100, 150, 200]
    #     num_inf_step_list = [50]
    #     # gs_list = [3.0, 7.5, 15.0, 25.0, 50.0, 100.0]
    #     # gs_list = [0.25, 0.5, 1.0, 1.5, 2.0, 2.5]
    #     gs_list = [3.0]
    #     for num_inference_steps in num_inf_step_list:
    #         for guidance_scale in gs_list:
    #             os.makedirs(f"{output_dir}/gs_{guidance_scale}_steps_{num_inference_steps}", exist_ok=True)
    #             out = sd_pipe(inp, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps, num_images_per_prompt=NUM_IMAGES)
    #             output_image_list = out["images"]
    #             for idx, output_image in enumerate(output_image_list):
    #                 output_image.save(f"{output_dir}/gs_{guidance_scale}_steps_{num_inference_steps}/output_image_{idx}_{guidance_scale}_{num_inference_steps}.jpg")

    ####### SIMCLR DOG #######
    # output_dir = os.path.join(out_dir, f"output_images_simclr_dog")
    # os.makedirs(output_dir, exist_ok=True)
    # input_image = Image.open("/home/jrick6/repos/simclr/tf2/simclr_dog.png")

    # im = Image.new("RGB", input_image.size, (255, 255, 255))
    # im.paste(input_image, mask=input_image.split()[3]) # 3 is the alpha channel
    # # Image.fromarray(np.uint8(input_image*255))
    # # im2 = Image.open("/home/jrick6/repos/dreambooth_tutorial/dog/alvan-nee-9M0tSjb-cpA-unsplash.jpeg")
    # # im = Image.fromarray(np.uint8(cm.gist_earth(input_image)*255))
    # # im2 = Image.open("test.jpg")

    # # Stable Diffusion Image Variation Testing
    # ###################################
    # sd_pipe = StableDiffusionImageVariationPipeline.from_pretrained(
    # "lambdalabs/sd-image-variations-diffusers",
    # revision="v2.0"
    # )
    # sd_pipe = sd_pipe.to(device)
    # sd_pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))
    # ###################################

    # # Segment Anything Testing
    # ###################################
    # # sam = sam_model_registry["vit_b"](checkpoint="/home/jrick6/repos/simclr/segment_anything/sam_vit_b_01ec64.pth")
    # # mask_generator = SamAutomaticMaskGenerator(sam)
    # # masks = mask_generator.generate(np.array(im))
    # ###################################

    # # im.save(os.path.join(output_dir, f"input_image_simclr_dog.jpg"))

    # tform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Resize(
    #         (224, 224),
    #         interpolation=transforms.InterpolationMode.BICUBIC,
    #         antialias=False,
    #         ),
    #     transforms.Normalize(
    #     [0.48145466, 0.4578275, 0.40821073],
    #     [0.26862954, 0.26130258, 0.27577711]),
    # ])

    # inp = tform(im).to(device).unsqueeze(0)
    # # guidance_scale=3.0
    # # num_inference_steps=50.0

    # # num_inf_step_list = [50, 75, 100, 150, 200]
    # num_inf_step_list = [50]
    # # gs_list = [3.0, 7.5, 15.0, 25.0, 50.0, 100.0]
    # # gs_list = [0.25, 0.5, 1.0, 1.5, 2.0, 2.5]
    # gs_list = [3.0]
    # for num_inference_steps in num_inf_step_list:
    #     for guidance_scale in gs_list:
    #         os.makedirs(f"{output_dir}/gs_{guidance_scale}_steps_{num_inference_steps}", exist_ok=True)
    #         out = sd_pipe(inp, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps, num_images_per_prompt=NUM_IMAGES)
    #         output_image_list = out["images"]
    #         for idx, output_image in enumerate(output_image_list):
    #             output_image.save(f"{output_dir}/gs_{guidance_scale}_steps_{num_inference_steps}/output_image_{idx}_{guidance_scale}_{num_inference_steps}.jpg")