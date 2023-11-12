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

# import random_util

FLAGS = flags.FLAGS

def index_func0(tensor):
    if FLAGS.DEBUG_LOG:
       print("index_func0", flush=True)
    return tensor[:,:,:,0]

def index_func1(tensor):
    if FLAGS.DEBUG_LOG:
       print("index_func1", flush=True)
    return tensor[:,:,:,1]

def index_func2(tensor):
    if FLAGS.DEBUG_LOG:
       print("index_func2", flush=True)
    return tensor[:,:,:,2]

def index_func3(tensor):
    if FLAGS.DEBUG_LOG:
       print("index_func3", flush=True)
    return tensor[:,:,:,3]

def index_func4(tensor):
    if FLAGS.DEBUG_LOG:
       print("index_func4", flush=True)
    return tensor[:,:,:,4]

def index_func5(tensor):
    if FLAGS.DEBUG_LOG:
       print("index_func5", flush=True)
    return tensor[:,:,:,5]

def index_func6(tensor):
    if FLAGS.DEBUG_LOG:
       print("index_func6", flush=True)
    return tensor[:,:,:,6]

def index_func7(tensor):
    if FLAGS.DEBUG_LOG:
       print("index_func7", flush=True)
    return tensor[:,:,:,7]

def index_func8(tensor):
    if FLAGS.DEBUG_LOG:
       print("index_func8", flush=True)
    return tensor[:,:,:,8]

def index_func9(tensor):
    if FLAGS.DEBUG_LOG:
       print("index_func9", flush=True)
    return tensor[:,:,:,9]

def index_func10(tensor):
    if FLAGS.DEBUG_LOG:
       print("index_func10", flush=True)
    return tensor[:,:,:,10]

def index_func11(tensor):
    if FLAGS.DEBUG_LOG:
       print("index_func11", flush=True)
    return tensor[:,:,:,11]

def index_func12(tensor):
    if FLAGS.DEBUG_LOG:
       print("index_func12", flush=True)
    return tensor[:,:,:,12]

def index_func13(tensor):
    if FLAGS.DEBUG_LOG:
       print("index_func13", flush=True)
    return tensor[:,:,:,13]

def index_func14(tensor):
    if FLAGS.DEBUG_LOG:
       print("index_func14", flush=True)
    return tensor[:,:,:,14]

def index_func15(tensor):
    if FLAGS.DEBUG_LOG:
       print("index_func15", flush=True)
    return tensor[:,:,:,15]

def index_func16(tensor):
    if FLAGS.DEBUG_LOG:
       print("index_func16", flush=True)
    return tensor[:,:,:,16]

def index_func17(tensor):
    if FLAGS.DEBUG_LOG:
       print("index_func17", flush=True)
    return tensor[:,:,:,17]

def index_func18(tensor):
    if FLAGS.DEBUG_LOG:
       print("index_func18", flush=True)
    return tensor[:,:,:,18]

def index_func19(tensor):
    if FLAGS.DEBUG_LOG:
       print("index_func19", flush=True)
    return tensor[:,:,:,19]

def index_func20(tensor):
    if FLAGS.DEBUG_LOG:
       print("index_func20", flush=True)
    return tensor[:,:,:,20]

def index_func21(tensor):
    if FLAGS.DEBUG_LOG:
       print("index_func21", flush=True)
    return tensor[:,:,:,21]

def index_func22(tensor):
    if FLAGS.DEBUG_LOG:
       print("index_func22", flush=True)
    return tensor[:,:,:,22]

def index_func23(tensor):
    if FLAGS.DEBUG_LOG:
       print("index_func23", flush=True)
    return tensor[:,:,:,23]

def index_func24(tensor):
    if FLAGS.DEBUG_LOG:
       print("index_func24", flush=True)
    return tensor[:,:,:,24]

class Lambda:
    def __init__(self, func, arg):
        self._func = func
        self._arg = arg
        
    def __call__(self):
        return self._func(self._arg)

def get_random_index(tensor):
    list_of_funcs = [
      index_func0, index_func1, index_func2, index_func3, index_func4, 
      index_func5, index_func6, index_func7, index_func8, index_func9, 
      index_func10, index_func11, index_func12, index_func13, index_func14, 
      index_func15, index_func16, index_func17, index_func18, index_func19, 
      index_func20, index_func21, index_func22, index_func23, index_func24
    ]
    branch_index = tf.random.uniform(shape=[], minval=0, maxval=len(list_of_funcs), dtype=tf.int32)
    output = tf.switch_case(
        branch_index=branch_index, 
        branch_fns=[Lambda(func, tensor) for func in list_of_funcs], 
    )
    return output


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

        # random_indices = tf.convert_to_tensor(random_util.random_indices)
        # print(random_indices, flush=True)
        # print(random_indices[id], flush=True)

        print(var_tensor, flush=True)
        if FLAGS.augmentation_mode == "variations_only":
          for _ in range(2):
            xs.append(tf.image.convert_image_dtype(get_random_index(var_tensor), dtype=tf.float32))
          image = tf.concat(xs, -1)
        elif FLAGS.augmentation_mode == "variations_then_default":
          for _ in range(2):
            xs.append(preprocess_fn_pretrain(get_random_index(var_tensor)))
          image = tf.concat(xs, -1)
        elif FLAGS.augmentation_mode == "variations_or_default":
            if np.random.random() < FLAGS.variations_or_default_chance:
              for _ in range(2):
                xs.append(preprocess_fn_pretrain(image))
            else:
              for _ in range(2):
                xs.append(tf.image.convert_image_dtype(get_random_index(var_tensor), dtype=tf.float32))
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
