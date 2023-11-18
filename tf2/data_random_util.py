import functools
from absl import flags
from absl import logging

import pdb
import numpy as np

import tensorflow.compat.v2 as tf

FLAGS = flags.FLAGS

def index_func(idx0, idx1, tensor):
    if FLAGS.DEBUG_LOG:
       print(f"index_func {idx0}, {idx1}", flush=True)
    xs = []
    xs.append(tf.image.convert_image_dtype(tensor[:,:,:,idx0], dtype=tf.float32))
    xs.append(tf.image.convert_image_dtype(tensor[:,:,:,idx1], dtype=tf.float32))
    return tf.concat(xs, -1)

class Lambda:
    def __init__(self, func, arg):
        self._func = func
        self._arg = arg
        
    def __call__(self):
        return self._func(self._arg)

def get_random_index(tensor):
    list_of_funcs = []
    for i in range(FLAGS.num_variations):
        for j in range(FLAGS.num_variations):
            if i != j:
                list_of_funcs.append(functools.partial(index_func, i, j))

    branch_index = tf.random.uniform(shape=[], minval=0, maxval=len(list_of_funcs), dtype=tf.int32)
    output = tf.switch_case(
        branch_index=branch_index, 
        branch_fns=[Lambda(func, tensor) for func in list_of_funcs], 
    )
    return output