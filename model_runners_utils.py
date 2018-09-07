import tensorflow as tf

import dataset
import prediction_models


def build_prediction_model(config, mode):
  """Builds a `Seq2SeqPredictionModel` instance.

  Args:
    config: a `Seq2SeqModel` protobuf message.
    mode: string scalar, mode of dataset (train, eval or infer).

  Returns:
    prediction_model: a `Seq2SeqPredictionModel` instance.
  """
  random_seed = config.random_seed
  src_vocab_size = config.src_vocab_size
  tgt_vocab_size = config.tgt_vocab_size

  config = config.prediction_model
  keep_prob = (config.keep_prob 
      if mode == tf.contrib.learn.ModeKeys.TRAIN else 1.0)
  is_inferring = (mode == tf.contrib.learn.ModeKeys.INFER)
  initializer = _build_initializer(config.initializer, random_seed)

  if config.HasField('attention_type'):
    prediction_model = prediction_models.AttentionSeq2SeqPredictionModel(
        unit_type=config.unit_type,
        num_units=config.num_units,
        forget_bias=config.forget_bias,
        keep_prob=keep_prob,
        encoder_type=config.encoder_type,
        time_major=config.time_major,
        share_vocab=config.share_vocab,
        is_inferring=is_inferring,
        initializer=initializer,
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        num_encoder_layers=config.num_encoder_layers,
        num_encoder_res_layers=config.num_encoder_res_layers,
        num_decoder_layers=config.num_decoder_layers,
        num_decoder_res_layers=config.num_decoder_res_layers,
        attention_type=config.attention_type,
        output_attention=config.output_attention)
  else:
    prediction_model = prediction_models.VanillaSeq2SeqPredictionModel(
        unit_type=config.unit_type,
        num_units=config.num_units,
        forget_bias=config.forget_bias,
        keep_prob=keep_prob,
        encoder_type=config.encoder_type,
        time_major=config.time_major,
        share_vocab=config.share_vocab,
        is_inferring=is_inferring,
        initializer=initializer,
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        num_encoder_layers=config.num_encoder_layers,
        num_encoder_res_layers=config.num_encoder_res_layers,
        num_decoder_layers=config.num_decoder_layers,
        num_decoder_res_layers=config.num_decoder_res_layers)

  return prediction_model


def build_dataset(config, mode):
  """Builds a `Seq2SeqDataset` instance.

  Args:
    config: a `Seq2SeqModel` protobuf message. 
    mode: string scalar, mode of dataset (train, eval or infer).

  Returns:
    dataset: a 'Seq2SeqDataset' instance. 
  """
  random_seed = config.random_seed
  src_vocab_size = config.src_vocab_size
  tgt_vocab_size = config.tgt_vocab_size
  config = config.dataset

  if mode == tf.contrib.learn.ModeKeys.TRAIN:
    return dataset.TrainerSeq2SeqDataset(
        batch_size=config.batch_size,
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        shuffle_buffer_size=config.shuffle_buffer_size,
        num_buckets=config.num_buckets,
        src_max_len=config.src_max_len,
        tgt_max_len=config.tgt_max_len,
        sos=config.sos,
        eos=config.eos,
        random_seed=random_seed)
  elif mode == tf.contrib.learn.ModeKeys.EVAL:
    return dataset.EvaluatorSeq2SeqDataset(
        batch_size=config.batch_size,
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        num_buckets= config.num_buckets,
        src_max_len=None,
        tgt_max_len=None,
        sos=config.sos,
        eos=config.eos)
  elif mode == tf.contrib.learn.ModeKeys.INFER:
    return dataset.InferencerSeq2SeqDataset(
        batch_size=1,
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        src_max_len=None, 
        sos=config.sos,
        eos=config.eos)
  else:
    raise ValueError('Unknown dataset mode: {}'.format(mode))
  
  return dataset


def _build_initializer(config, random_seed):
  """Builds weight initializer.

  Args:
    config: an `Initializer` protobuf message.
    random_seed: int scalar, random seed.

  Returns:
    initializer: an initializer instance.
  """
  if config.WhichOneof('initializer_oneof') == 'uniform':
    config = config.uniform
    initializer = tf.random_uniform_initializer(
        minval=config.min_val, maxval=config.max_val, seed=random_seed)
  elif config.WhichOneof('initializer_oneof') == 'truncated_normal': 
    config = config.truncated_normal
    initializer = tf.truncated_normal_initializer(
        mean=config.mean, stddev=config.stddev, seed=random_seed)
  else:
    raise ValueError('Unknown initializer')
  return initializer


def create_loss(tensor_dict, logits, time_major):
  """Builds the graph that creates the loss tensor from the logits and 
  groundtruth tensors.

  Args:
    tensor_dict: a dict mapping from tensor names to tensors. Must have the 
      groundtruth entries `tgt_output_ids` and `tgt_seq_lens`.
    logits: float tensor with shape [max_tgt_time, batch, tgt_vocab_size]/
      [batch, max_tgt_time, tgt_vocab_size], the prediction logits.
    time_major: bool scalar, whether `logits` is in time major format.

  Returns:
    loss: float scalar tensor, the loss averaged over a batch of sequence pairs.
  """
  tgt_output_ids = tensor_dict['tgt_output_ids']
  tgt_seq_lens = tensor_dict['tgt_seq_lens']
  max_time = tf.reduce_max(tgt_seq_lens)

  if time_major:
    tgt_output_ids = tf.transpose(tgt_output_ids)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=tgt_output_ids, logits=logits)
  cross_entropy_weights = tf.sequence_mask(
      tgt_seq_lens, max_time, dtype=tf.float32)
  if time_major:
    cross_entropy_weights = tf.transpose(cross_entropy_weights)
  loss = tf.divide(tf.reduce_sum(cross_entropy * cross_entropy_weights), 
      tf.to_float(tf.size(tgt_seq_lens)))
  return loss


def build_optimizer(config):
  """Builds optimizer.

  Args:
    config: an `Optimization` protobuf message.

  Returns:
    optimizer: an optimizer instance.
    learning_rate: float scalar tensor, learning rate.
  """
  
  learning_rate = _build_learning_rate(config)

  if config.optimizer == 'sgd':
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  elif config.optimizer == 'adam':
    optimizer = tf.train.AdamOptimizer(learning_rate)
  else:
    raise ValueError('Unknown optimizer: {}'.format(config.optimizer))
  return optimizer, learning_rate


def _build_learning_rate(config):
  """Builds learning rate tensor.

  Args:
    config: an `Optimization` protobuf message.

  Returns:
    learning_rate: float scalar tensor, learning rate.
  """
  global_step = tf.train.get_or_create_global_step()
  learning_rate = tf.constant(config.base_learning_rate)
  learning_rate = _apply_learning_rate_warmup(
      config, learning_rate, global_step)
  learning_rate = _apply_learning_rate_decay(
      config, learning_rate, global_step)
  return learning_rate


def _apply_learning_rate_warmup(config, learning_rate, global_step):
  """Applies learning rate warm up. Warms up training with smaller
  learning rate at the beginning.

  Args:
    config: an `Optimization` protobuf message.
    learning_rate: float scalar tensor, learning rate.
    global_step: int scalar tensor, global_step.

  Returns:
    learning_rate: float scalar tensor, learning rate with warm-up.
  """
  if config.warmup_scheme == 't2t':
    warmup_factor = tf.exp(tf.log(0.01) / config.warmup_steps)
    inv_decay = tf.pow(warmup_factor, 
                       tf.to_float(config.warmup_steps - global_step))
  else:
    raise ValueError('Unknown warmup scheme {}'.format(config.warmup_scheme))
    
  return tf.cond(
      global_step < config.warmup_steps,
      lambda: inv_decay * learning_rate,
      lambda: learning_rate,
      name='learning_rate_warmup')


def _apply_learning_rate_decay(config, learning_rate, global_step):
  """Applies learning rate decay. Decays learning rate over time.
  
  Args:
    config: an `Optimization` protobuf message.
    learning_rate: float scalar tensor, learning rate.
    global_step: int scalar tensor, global_step.

  Returns:
    learning_rate: float scalar tensor, decayed learning rate.
  """
  if config.decay_scheme in ('luong5', 'luong10', 'luong234'):
    decay_factor = 0.5
    if config.decay_scheme == 'luong5':
      start_decay_step = int(config.num_train_steps / 2)
      decay_times = 5
    elif config.decay_scheme == 'luong10':
      start_decay_step = int(config.num_train_steps / 2)
      decay_times = 10
    elif config.decay_scheme == 'luong234':
      start_decay_step = int(config.num_train_steps * 2 / 3)
      decay_times = 4
    remain_steps = config.num_train_steps - start_decay_step
    decay_steps = int(remain_steps / decay_times)
  elif not config.decay_scheme:
    start_decay_step = config.num_train_steps
    decay_steps = 0
    decay_factor = 1.0
  elif config.decay_scheme:
    raise ValueError('Unknown decay scheme {}'.format(config.decay_scheme))

  return tf.cond(
      global_step < start_decay_step,
      lambda: learning_rate,
      lambda: tf.train.exponential_decay(
          learning_rate,
          (global_step - start_decay_step),
          decay_steps, decay_factor, staircase=True),
      name='learning_rate_decay')


def get_symbols_count(src_seq_lens, tgt_seq_lens):
  """Returns the total num of symbols in source sequence and target sequence.

  Args:
    'src_seq_lens': int tensor with shape [batch], the lengths of unpadded 
        source sequences in a batch.
    'tgt_seq_lens': int tensor with shape [batch], the lengths of unpadded 
        target sequences in a batch.

  Returns:
    int scalar tensor, symbols count.
  """
  return tf.reduce_sum(src_seq_lens) + tf.reduce_sum(tgt_seq_lens)


def get_predict_count(tgt_seq_lens):
  """Returns the total num of symbols in the target sequence.

  Args:
    'tgt_seq_lens': int tensor with shape [batch], the lengths of unpadded 
        target sequences in a batch.

  Returns:
    int scalar tensor, predicted symbols count.
  """
  return tf.reduce_sum(tgt_seq_lens)


def apply_gradient_clip(grads, max_grad_norm):
  """Applies gradient clipping.

  Args:
    grads: a list/tuple of gradients w.r.t weight variables.
    max_grad_norm: float scalar, the maximum gradient norm that all gradients
      are clipped to, in order to prevent exploding gradients.

  Returns:
    clipped_grads: a list/tuple of gradient tensors after clipping.
    grad_norm_summary: a string scalar tensor, the protobuf message containing
      `grad_norm`.
    grad_norm: float scalar tensor, the global norm of weight variables before
      clipping.
  """
  clipped_grads, grad_norm = tf.clip_by_global_norm(
      grads, max_grad_norm)
  grad_norm_summary = [tf.summary.scalar('gradient_norm', grad_norm),
      tf.summary.scalar('clipped_gradient', tf.global_norm(clipped_grads))]
  return clipped_grads, grad_norm_summary, grad_norm


def create_persist_saver(max_to_keep=5):
  """Creates persist saver for persisting variables to a checkpoint file.

  Args:
    max_to_keep: int scalar or None, max num of checkpoints to keep. If None,
      keeps all checkpoints.
        
  Returns:
    persist_saver: a tf.train.Saver instance.
  """
  persist_saver = tf.train.Saver(max_to_keep=max_to_keep)
  return persist_saver


def create_restore_saver():
  """Creates restore saver for persisting variables to a checkpoint file.

  Returns:
    restore_saver: a tf.train.Saver instance.
  """
  restore_saver = tf.train.Saver()
  return restore_saver

