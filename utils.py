import tensorflow as tf
import numpy as np


def get_padding_mask(inputs, padding_value=0):
  """Creates a binary tensor to mask out padded tokens.

  Args:
    inputs: int tensor of shape [batch_size, src_seq_len], token ids
      of source sequences.
    padding_value: int scalar, the vocabulary index of the PAD token. 

  Returns:
    mask: binary tensor of shape [batch_size, src_seq_len], storing ones for 
      padded tokens and zeros for regular tokens.
  """
  mask = tf.cast(tf.equal(inputs, padding_value), 'float32')
  return mask
