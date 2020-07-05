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


def compute_loss(labels, logits, smoothing, vocab_size, padding_value=0):
  """Computes average (per-token) cross entropy loss.

  1. Applies label smoothing -- all entries in the groundtruth label tensor  
     get non-zero probability mass.
  2. Computes per token loss of shape [batch_size, tgt_seq_len], where padded
     positions are masked, and then the sum of per token loss is normalized by
     the total number of non-padding entries.

  Args:
    labels: int tensor of shape [batch_size, tgt_seq_len], the groundtruth
      token ids.
    logits: float tensor of shape [batch_size, tgt_seq_len, vocab_size], the
      predicted logits of tokens over the vocabulary.
    smoothing: float scalar, the amount of label smoothing applied to the
      one-hot class labels. 
    vocab_size: int scalar, num of tokens (including SOS and EOS) in the 
      vocabulary.
    padding_value: int scalar, the vocabulary index of the PAD token. 

  Returns:
    loss: float scalar tensor, the per-token cross entropy
  """
  # effective_vocab = vocab - {SOS_ID}
  effective_vocab_size = vocab_size - 1

  # prob mass allocated to the token that should've been predicted 
  on_value = 1.0 - smoothing 
  # prob mass allocated to all other tokens
  off_value = smoothing / (effective_vocab_size - 1)

  # [batch_size, tgt_seq_len, vocab_size] 
  labels_one_hot = tf.one_hot(
      labels,
      depth=vocab_size,
      on_value=on_value,
      off_value=off_value)

  # compute cross entropy over all tokens in vocabulary but SOS_ID (i.e. 0)
  # because SOS_ID should never appear in the decoded sequence
  # [batch_size, tgt_seq_len]
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
      labels=labels_one_hot[:, :, 1:], logits=logits[:, :, 1:])

  # this is the entropy when the softmax'ed logits == groundtruth labels
  # so it should be deducted from `cross_entropy` to make sure the minimum 
  # possible cross entropy == 0
  normalizing_constant = -(on_value * tf.math.log(on_value) +
      (effective_vocab_size - 1) * off_value * tf.math.log(off_value + 1e-20))
  cross_entropy -= normalizing_constant

  # mask out predictions where the labels == `padding_value`  
  weights = tf.cast(tf.not_equal(labels, padding_value), 'float32')
  cross_entropy *= weights
  loss = tf.reduce_sum(cross_entropy) / tf.reduce_sum(weights)
  return loss


class LearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  """Learning rate schedule."""
  def __init__(self, learning_rate, hidden_size, warmup_steps):
    """Constructor.

    Args:
      learning_rate: float scalar, the base learning rate.
      hidden_size: int scalar, the hidden size of continuous representation.
      warmup_steps: int scalar, the num of warm-up steps
    """
    super(LearningRateSchedule, self).__init__()
    self._learning_rate = learning_rate
    self._hidden_size = hidden_size
    self._warmup_steps = tf.cast(warmup_steps, 'float32')

  def __call__(self, global_step):
    """Computes learning rate with linear warmup and rsqrt decay.

    Args:
      global_step: int scalar tensor, the current global step. 

    Returns:
      learning_rate: float scalar tensor, the learning rate as a function of
        the input `global_step`. 
    """
    global_step = tf.cast(global_step, 'float32')
    learning_rate = self._learning_rate
    learning_rate *= (self._hidden_size**-0.5)
    # linear warmup
    learning_rate *= tf.minimum(1.0, global_step / self._warmup_steps)
    # rsqrt decay
    learning_rate /= tf.sqrt(tf.maximum(global_step, self._warmup_steps))
    return learning_rate


def save_attention_weights(filename, data):
  """Saves attention weights data to *.npy file.

  Args:
    filename: string scalar, filename.
    data: a list or tuple or dict of numpy arrays, the attention weights and 
      token ids of input and translated sequence.
  """
  np.save(filename, data)
