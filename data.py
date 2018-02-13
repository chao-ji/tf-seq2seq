from __future__ import division

import tensorflow as tf

UNK = "<unk>"
SOS = "<s>"
EOS = "</s>"
UNK_ID = 0

class Seq2SeqDataset(object):
  def __init__(self,
      hparams,
      mode, 
      src_placeholder=None, 
      batch_size_placeholder=None):

    self._mode = mode
    self._src_vocab_table = tf.contrib.lookup.index_table_from_file(
        hparams.src_vocab_file, default_value=UNK_ID)
    self._tgt_vocab_table = tf.contrib.lookup.index_table_from_file(
        hparams.tgt_vocab_file, default_value=UNK_ID)

    if mode != tf.contrib.learn.ModeKeys.INFER:
      self._reverse_tgt_vocab_table = None
    else:
      self._reverse_tgt_vocab_table = \
          tf.contrib.lookup.index_to_string_table_from_file(
              hparams.tgt_vocab_file, default_value=UNK)
 
    self._src_eos_id = tf.cast(self.src_vocab_table.lookup(
        tf.constant(hparams.eos)), tf.int32)
    self._tgt_sos_id = tf.cast(self.tgt_vocab_table.lookup(
        tf.constant(hparams.sos)), tf.int32)
    self._tgt_eos_id = tf.cast(self.tgt_vocab_table.lookup(
        tf.constant(hparams.eos)), tf.int32)

    if mode != tf.contrib.learn.ModeKeys.INFER:
      (self._initializer, self._src_input_ids, self._tgt_input_ids,
          self._tgt_output_ids, self._src_seq_lens, self._tgt_seq_lens
          ) = self._get_iterator(hparams)
    else:
      (self._initializer, self._src_input_ids, self._tgt_input_ids,
          self._tgt_output_ids, self._src_seq_lens, self._tgt_seq_lens
          ) = self._get_infer_iterator(hparams, src_placeholder,
          batch_size_placeholder)

  @property
  def mode(self):
    return self._mode

  @property
  def src_vocab_table(self):
    return self._src_vocab_table

  @property
  def tgt_vocab_table(self):
    return self._tgt_vocab_table

  @property
  def reverse_tgt_vocab_table(self):
    return self._reverse_tgt_vocab_table

  @property
  def src_eos_id(self):
    return self._src_eos_id

  @property
  def tgt_sos_id(self):
    return self._tgt_sos_id

  @property
  def tgt_eos_id(self):
    return self._tgt_eos_id

  @property
  def initializer(self):
    return self._initializer

  @property
  def src_input_ids(self):
    return self._src_input_ids

  @property
  def tgt_input_ids(self):
    return self._tgt_input_ids

  @property
  def tgt_output_ids(self):
    return self._tgt_output_ids

  @property
  def src_seq_lens(self):
    return self._src_seq_lens

  @property
  def tgt_seq_lens(self):
    return self._tgt_seq_lens

  def init_iterator(self, sess, feed_dict=None):
    if self.mode == tf.contrib.learn.ModeKeys.INFER and not feed_dict:
        raise ValueError("`feed_dict` can't be None in infer mode")
    sess.run(self.initializer, feed_dict)

  def _get_infer_iterator(self,
      hparams, 
      src_placeholder, 
      batch_size_placeholder):
    if src_placeholder is None or batch_size_placeholder is None:
      raise ValueError("`src_placeholder` and `batch_size_placeholder`", 
          "must be provided in infer mode")
    src_dataset = tf.data.Dataset.from_tensor_slices(src_placeholder)
    batch_size = batch_size_placeholder
    src_vocab_table = self.src_vocab_table
    src_max_len = hparams.src_max_len_infer
  
    src_eos_id = self.src_eos_id 
    src_dataset = src_dataset.map(lambda src: tf.string_split([src]).values)

    if src_max_len:
      src_dataset = src_dataset.map(lambda src: src[:src_max_len])
    
    src_dataset = src_dataset.map(
        lambda src: tf.cast(src_vocab_table.lookup(src), tf.int32))

    src_dataset = src_dataset.map(lambda src: (src, tf.size(src)))

    def batching_func(x):
      return x.padded_batch(
          batch_size,
          padded_shapes=(
              tf.TensorShape([None]),
              tf.TensorShape([])),
          padding_values=(
              src_eos_id,
              0))
    
    batched_dataset = batching_func(src_dataset)
    batched_iter = batched_dataset.make_initializable_iterator()
    (src_input_ids, src_seq_lens) = batched_iter.get_next()

    return (batched_iter.initializer, src_input_ids, None, None, 
        src_seq_lens, None)

  def _get_iterator(self,
        hparams,
        num_parallel_calls=4,
        output_buffer_size=None,
        reshuffle_each_iteration=True):
    if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
      src_file = ".".join([hparams.train_prefix, hparams.src])
      tgt_file = ".".join([hparams.train_prefix, hparams.tgt])
      src_max_len = hparams.src_max_len
      tgt_max_len = hparams.tgt_max_len
    elif self.mode == tf.contrib.learn.ModeKeys.EVAL:
      src_file = ".".join([hparams.dev_prefix, hparams.src])
      tgt_file = ".".join([hparams.dev_prefix, hparams.tgt])
      src_max_len = hparams.src_max_len_infer
      tgt_max_len = hparams.tgt_max_len_infer
    else:
      pass

    src_dataset = tf.data.TextLineDataset(src_file)
    tgt_dataset = tf.data.TextLineDataset(tgt_file)  
    src_vocab_table = self.src_vocab_table
    tgt_vocab_table = self.tgt_vocab_table
    batch_size = hparams.batch_size
    sos = hparams.sos
    eos = hparams.eos
    random_seed = hparams.random_seed
    num_buckets = hparams.num_buckets
    src_eos_id = self.src_eos_id
    tgt_sos_id = self.tgt_sos_id
    tgt_eos_id = self.tgt_eos_id
    if not output_buffer_size:
      output_buffer_size = batch_size * 1000
    
    src_tgt_dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))

    src_tgt_dataset = src_tgt_dataset.shuffle(
        output_buffer_size, random_seed, reshuffle_each_iteration)

    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: (
            tf.string_split([src]).values, tf.string_split([tgt]).values),
        num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

    src_tgt_dataset = src_tgt_dataset.filter(
        lambda src, tgt: tf.logical_and(tf.size(src) > 0, tf.size(tgt) > 0))

    if src_max_len:
      src_tgt_dataset = src_tgt_dataset.map(
          lambda src, tgt: (src[:src_max_len], tgt),
          num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
    if tgt_max_len:
      src_tgt_dataset = src_tgt_dataset.map(
          lambda src, tgt: (src, tgt[:tgt_max_len]),
          num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: (tf.cast(src_vocab_table.lookup(src), tf.int32),
                          tf.cast(tgt_vocab_table.lookup(tgt), tf.int32)),
        num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: (src,
                          tf.concat(([tgt_sos_id], tgt), 0),
                          tf.concat((tgt, [tgt_eos_id]), 0)),
        num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt_in, tgt_out: (
            src, tgt_in, tgt_out, tf.size(src), tf.size(tgt_in)),
        num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

    def batching_func(x):
      return x.padded_batch(
          batch_size,
          padded_shapes=(
              tf.TensorShape([None]),
              tf.TensorShape([None]),
              tf.TensorShape([None]),
              tf.TensorShape([]),
              tf.TensorShape([])),
          padding_values=(
              src_eos_id,
              tgt_eos_id,
              tgt_eos_id,
              0,
              0))

    if num_buckets > 1:
      def key_func(unused_1, unused_2, unused_3, src_len, tgt_len):
        if src_max_len:
          bucket_width = (src_max_len + num_buckets - 1) // num_buckets
        else:
          bucket_width = 10

        bucket_id = tf.maximum(src_len // bucket_width, tgt_len // bucket_width)
        return tf.to_int64(tf.minimum(num_buckets, bucket_id))

      def reduce_func(unused_key, windowed_data):
        return batching_func(windowed_data)

      batched_dataset = src_tgt_dataset.apply(
          tf.contrib.data.group_by_window(
              key_func=key_func, reduce_func=reduce_func, window_size=batch_size))
    else:
      batched_dataset = batching_func(src_tgt_dataset)

    batched_iter = batched_dataset.make_initializable_iterator()
    (src_input_ids, tgt_input_ids, tgt_output_ids, src_seq_lens,
    tgt_seq_lens) = (batched_iter.get_next())

    return (batched_iter.initializer, src_input_ids, tgt_input_ids, 
        tgt_output_ids, src_seq_lens, tgt_seq_lens)


def get_max_time(dataset):
  return tf.reduce_max(dataset.tgt_seq_lens)


def get_word_count(dataset):
  return tf.reduce_sum(dataset.src_seq_lens) + tf.reduce_sum(dataset.tgt_seq_lens)


def get_predict_count(dataset):
  return tf.reduce_sum(dataset.tgt_seq_lens)
