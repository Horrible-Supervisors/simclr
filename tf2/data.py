# coding=utf-8
# Copyright 2020 The SimCLR Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific simclr governing permissions and
# limitations under the License.
# ==============================================================================
"""Data pipeline."""

import functools
from absl import flags
from absl import logging

import pdb
import numpy as np

import data_util
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

import data_random_util

FLAGS = flags.FLAGS

def build_input_fn(builder, global_batch_size, topology, is_training):
  """Build input function.

  Args:
    builder: TFDS builder for specified dataset.
    global_batch_size: Global batch size.
    topology: An instance of `tf.tpu.experimental.Topology` or None.
    is_training: Whether to build in training mode.

  Returns:
    A function that accepts a dict of params and returns a tuple of images and
    features, to be used as the input_fn in TPUEstimator.
  """

  def _input_fn(input_context):
    """Inner input function."""
    batch_size = input_context.get_per_replica_batch_size(global_batch_size)
    logging.info('Global batch size: %d', global_batch_size)
    logging.info('Per-replica batch size: %d', batch_size)
    preprocess_fn_pretrain = get_preprocess_fn(is_training, is_pretrain=True)
    preprocess_fn_finetune = get_preprocess_fn(is_training, is_pretrain=False)
    num_classes = builder.info.features['label'].num_classes

    # @tf.py_function(Tout=[tf.float32, tf.float32])
    # def map_fn(image, label, id):
    #   """Produces multiple transformations of the same batch."""
    #   if is_training and FLAGS.train_mode == 'pretrain':
    #     xs = []
    #     for _ in range(2):  # Two transformations
    #       xs.append(preprocess_fn_pretrain(image))
    #     image = tf.concat(xs, -1)
    #   else:
    #     image = preprocess_fn_finetune(image)
    #   label = tf.one_hot(label, num_classes)
    #   return image, label

    def map_fn(inp_dict):
      """Produces multiple transformations of the same batch."""
      image = inp_dict['image']
      label = inp_dict['label']
      id = inp_dict['id']
      if is_training and FLAGS.train_mode == 'pretrain':
        xs = []
        for _ in range(2):  # Two transformations
          xs.append(preprocess_fn_pretrain(image))
        image = tf.concat(xs, -1)
      else:
        image = preprocess_fn_finetune(image)
      label = tf.one_hot(label, num_classes)
      return image, label

    # @tf.py_function(Tout=[tf.float32, tf.float32])
    # def img_var_map_fn(image, label, id):
    #   """Produces multiple transformations of the same batch."""
    #   if is_training and FLAGS.train_mode == 'pretrain':
    #     xs = []
    #     num_variations = 5
    #     out_dir = "/home/jrick6/tensorflow_datasets/imagenette_id_variations/full-size-v2/1.0.0"
    #     format_train = "imagenette-train"
    #     num_shards = 16
    #     xs = data_util.get_image_variations(id.numpy(), num_variations, out_dir, format_train, num_shards)
    #     image = tf.concat(xs, -1)
    #   else:
    #     image = preprocess_fn_finetune(image)
    #   label = tf.one_hot(label, num_classes)
    #   return image, label

    def img_var_map_fn(inp_dict, **kwargs):
      """Produces multiple transformations of the same batch."""
      image = inp_dict['image']
      label = inp_dict['label']
      id = inp_dict['id']
      if is_training and FLAGS.train_mode == 'pretrain':
        xs = []
        num_variations = FLAGS.num_variations
        variation_list = []
        for idx in range(num_variations):
          variation_list.append(tf.expand_dims(inp_dict[f'variation_{idx}'], -1))
        var_tensor = tf.concat(variation_list, -1)

        print(var_tensor, flush=True)
        if FLAGS.augmentation_mode == "variations_only":
          image = data_random_util.get_random_index(var_tensor)
        elif FLAGS.augmentation_mode == "variations_then_default":
          cur_image = data_random_util.get_random_index(var_tensor)
          print(cur_image, flush=True)
          x0 = cur_image[:, :, :3]
          x1 = cur_image[:, :, 3:]
          xs.append(preprocess_fn_pretrain(x0))
          xs.append(preprocess_fn_pretrain(x1))
          image = tf.concat(xs, -1)
        elif FLAGS.augmentation_mode == "variations_or_default":
            if np.random.random() < FLAGS.variations_or_default_chance:
              for _ in range(2):
                xs.append(preprocess_fn_pretrain(image))
              image = tf.concat(xs, -1)
            else:
              image = data_random_util.get_random_index(var_tensor)
        elif FLAGS.augmentation_mode == "variations_and_default":
           xs.append(preprocess_fn_pretrain(image))
           xs.append(data_random_util.get_random_index(var_tensor)[:, :, :3])
           image = tf.concat(xs, -1)
      else:
        image = preprocess_fn_finetune(image)
      label = tf.one_hot(label, num_classes)
      return image, label

    logging.info('num_input_pipelines: %d', input_context.num_input_pipelines)
    dataset = builder.as_dataset(
        split=FLAGS.train_split if is_training else FLAGS.eval_split,
        shuffle_files=is_training,
        as_supervised=False,
        # Passing the input_context to TFDS makes TFDS read different parts
        # of the dataset on different workers. We also adjust the interleave
        # parameters to achieve better performance.
        read_config=tfds.ReadConfig(
            interleave_cycle_length=32,
            interleave_block_length=1,
            input_context=input_context))
    if FLAGS.cache_dataset:
      dataset = dataset.cache()
    if is_training:
      options = tf.data.Options()
      options.experimental_deterministic = False
      options.experimental_slack = True
      dataset = dataset.with_options(options)
      buffer_multiplier = 50 if FLAGS.image_size <= 32 else 10
      dataset = dataset.shuffle(batch_size * buffer_multiplier)
      dataset = dataset.repeat(-1)
    if is_training and FLAGS.variations:
      dataset = dataset.map(
          img_var_map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    else:
      dataset = dataset.map(
          map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=is_training)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

  return _input_fn


def build_distributed_dataset(builder, batch_size, is_training, strategy,
                              topology):
  input_fn = build_input_fn(builder, batch_size, topology, is_training)
  return strategy.distribute_datasets_from_function(input_fn)


def get_preprocess_fn(is_training, is_pretrain):
  """Get function that accepts an image and returns a preprocessed image."""
  # Disable test cropping for small images (e.g. CIFAR)
  if FLAGS.image_size <= 32:
    test_crop = False
  else:
    test_crop = True
  color_jitter_strength = FLAGS.color_jitter_strength if is_pretrain else 0.
  return functools.partial(
      data_util.preprocess_image,
      height=FLAGS.image_size,
      width=FLAGS.image_size,
      is_training=is_training,
      color_jitter_strength=color_jitter_strength,
      test_crop=test_crop)
