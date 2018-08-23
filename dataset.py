from abc import abstractproperty

import tensorflow as tf

UNK = "<unk>"
UNK_ID = 0


class Seq2SeqDataset(object):
  """Base class for generating a dataset to be used by a seq2seq model.

  Subclasses must override the constructor and implement the `mode` abstract
  property.
  """
  def __init__(self,
               batch_size,
               src_vocab_size,
               tgt_vocab_size,
               src_max_len,
               sos='<s>',
               eos='</s>'):
    """Constructor.

    Args:
      batch_size: int scalar, batch size.
      src_vocab_size: int scalar, num of symbols in source vocabulary.
      tgt_vocab_size: int scalar, num of symbols in target vocabulary.
      src_max_len: int scalar or None, used to truncate source sequence to a 
        max length. If None, no truncating is performed. 
      sos: string scalar, the start-of-sentence marker.
      eos: string scalar, the end-of-sentence marker.
    """
    self._batch_size = batch_size
    self._src_vocab_size = src_vocab_size
    self._tgt_vocab_size = tgt_vocab_size
    self._src_max_len = src_max_len
    self._sos = sos
    self._eos = eos

    self._get_tensor_dict_scope = 'GetTensorDict'
    self._assert_equal_ops = list()

  @property
  def src_vocab_size(self):
    return self._src_vocab_size

  @property
  def tgt_vocab_size(self):
    return self._tgt_vocab_size

  @property
  def iterator_initializer(self):
    """Returns the initializer of the initializable iterator of dataset."""
    return self._iterator_initializer

  @abstractproperty
  def mode(self):
    """Returns a string scalar indicating the mode of dataset (train, eval or 
    infer).
    """
    pass

  def _get_vocab_tables(self, src_vocab_file, tgt_vocab_file):
    """Builds the vocabulary tables for looking up source and target sequence. 
    Vocabulary table is a dict mapping from symbols (string tensor) to symbol 
    indices (int tensor), where symbols are defined as elements that vocabulary
    is made up of. Has the side effect of setting `self._assert_equal_ops`.

    Args:
      src_vocab_file: string scalar, the path to the source vocabulary file,
        where each line contains a single symbol. 
      tgt_vocab_file: string scalar, the path to the target vocabulary file,
        where each line contains a single symbol. 

    Returns:
      src_vocab_table: a HashTable instance, the source vocabulary.
      tgt_vocab_table: a HashTable instance, the target vocabulary.
    """
    src_vocab_table = tf.contrib.lookup.index_table_from_file(
        src_vocab_file, default_value=UNK_ID)
    tgt_vocab_table = tf.contrib.lookup.index_table_from_file(
        tgt_vocab_file, default_value=UNK_ID)

    self._assert_equal_ops = [
        tf.assert_equal(src_vocab_table.size(),
            tf.constant(self._src_vocab_size, dtype=tf.int64)),
        tf.assert_equal(tgt_vocab_table.size(),
            tf.constant(self._tgt_vocab_size, dtype=tf.int64))]

    return src_vocab_table, tgt_vocab_table

  def _get_tgt_marker_indices(self, tgt_vocab_table):
    """Returns target sequence start-of-sentence (sos) and end-of-sentence
    (eos) indices in the vocabulary. Makes sure that the static vocabulary 
    sizes passed to the constructor are equal to the dynamic sizes by 
    running the assertion ops `self._assert_equal_ops` first.

    Args:
      tgt_vocab_table: a HashTable instance, returned by `_get_vocab_tables`.
      
    Returns:
      tgt_sos_id: an int scalar tensor, the index of sos in target vocabulary.
      tgt_eos_id: an int scalar tensor, the index of eos in target vocabulary.
    """
    with tf.control_dependencies(self._assert_equal_ops):
      tgt_sos_id = tf.to_int32(tgt_vocab_table.lookup(tf.constant(self._sos)))
      tgt_eos_id = tf.to_int32(tgt_vocab_table.lookup(tf.constant(self._eos)))
    return tgt_sos_id, tgt_eos_id

  def _get_tensor_dict_train_eval(self,
                                  src_file_list,
                                  tgt_file_list,
                                  src_vocab_file,
                                  tgt_vocab_file):
    """Generates tensor dict for training and evaluation.

    Args:
      src_file_list: a list of string scalars, the paths to the source sequence
        text files. 
      tgt_file_list: a list of string scalars, the paths to the corresponding
        target sequence text files.
      src_vocab_file: string scalar, the path to the source vocabulary file,
        where each line contains a single symbol. 
      tgt_vocab_file: string scalar, the path to the target vocabulary file,
        where each line contains a single symbol. 

    Returns:
      tensor_dict: a dict mapping from tensor names to tensors
        {'src_input_ids': [batch, max_time_src], the indices of source sequence
            symbols in a batch.
         'tgt_input_ids': [batch, max_time_tgt], the indices of target input
            sequence symbols in a batch.
         'tgt_output_ids': [batch, max_time_tgt], the indices of target output
            sequence symbols in a batch.
         'src_seq_lens': [batch], the lengths of unpadded source sequences in
            a batch.
         'tgt_seq_lens': [batch], the lengths of unpadded target sequences in
            a batch.}
    """
    if self.mode == tf.contrib.learn.ModeKeys.INFER:
      raise ValueError('Must be in train or eval mode when calling',
          '`_get_tensor_dict_train_eval`.')
    src_vocab_table, tgt_vocab_table = self._get_vocab_tables(src_vocab_file,
                                                              tgt_vocab_file)
    tgt_sos_id, tgt_eos_id = self._get_tgt_marker_indices(tgt_vocab_table)
    src_dataset = tf.data.TextLineDataset(src_file_list)
    tgt_dataset = tf.data.TextLineDataset(tgt_file_list)
    src_tgt_dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))

    if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
      src_tgt_dataset = src_tgt_dataset.repeat(None)
      if self._shuffle_buffer_size is not None:
        src_tgt_dataset = src_tgt_dataset.shuffle(
            self._shuffle_buffer_size, self._random_seed)

    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: (
            tf.string_split([src]).values, tf.string_split([tgt]).values))
    src_tgt_dataset = src_tgt_dataset.filter(
        lambda src, tgt: tf.logical_and(tf.size(src) > 0, tf.size(tgt) > 0))
    if self._src_max_len:
      src_tgt_dataset = src_tgt_dataset.map(
          lambda src, tgt: (src[:self._src_max_len], tgt))
    if self._tgt_max_len:
      src_tgt_dataset = src_tgt_dataset.map(
          lambda src, tgt: (src, tgt[:self._tgt_max_len]))
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: (tf.to_int32(src_vocab_table.lookup(src)),
                          tf.to_int32(tgt_vocab_table.lookup(tgt))))
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: (src,
                          tf.concat([[tgt_sos_id], tgt], 0),
                          tf.concat([tgt, [tgt_eos_id]], 0)))
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt_in, tgt_out: (
            src, tgt_in, tgt_out, tf.size(src), tf.size(tgt_in)))

    def batching_fn(dataset):
      return dataset.padded_batch(
          self._batch_size,
          padded_shapes=dataset.output_shapes)

    if self._num_buckets > 1:
      # mapper
      def key_fn(*args):
        src_len, tgt_len = args[-2], args[-1]
        bucket_width = 10
        if self._src_max_len:
          bucket_width = (self._src_max_len + self._num_buckets - 1
              ) // self._num_buckets
        bucket_id = tf.maximum(src_len, tgt_len) // bucket_width
        return tf.to_int64(tf.minimum(self._num_buckets, bucket_id))
      # reducer
      def reduce_fn(_, windowed_dataset):
        return batching_fn(windowed_dataset)

      batched_dataset = src_tgt_dataset.apply(
          tf.contrib.data.group_by_window(key_func=key_fn,
                                          reduce_func=reduce_fn,
                                          window_size=self._batch_size))
    else:
      batched_dataset = batching_fn(src_tgt_dataset)

    iterator = batched_dataset.make_initializable_iterator()
    (src_input_ids, tgt_input_ids, tgt_output_ids, src_seq_lens, tgt_seq_lens
        ) = iterator.get_next()

    tensor_dict = {'src_input_ids': src_input_ids,
                   'tgt_input_ids': tgt_input_ids,
                   'tgt_output_ids': tgt_output_ids,
                   'src_seq_lens': src_seq_lens,
                   'tgt_seq_lens': tgt_seq_lens}
    self._iterator_initializer = iterator.initializer 
    return tensor_dict   

  def _get_tensor_dict_infer(self,
                             src_file_list,
                             src_vocab_file,
                             tgt_vocab_file):
    """Generates tensor dict for inference.

    Args:
      src_file_list: a list of string scalars, the paths to the source sequence
        text files.
      src_vocab_file: string scalar, the path to the source vocabulary file,
        where each line contains a single symbol.
      tgt_vocab_file: string scalar, the path to the target vocabulary file,
        where each line contains a single symbol.

    Returns:
      tensor_dict: a dict mapping from tensor names to tensors
        {'src_input_ids': [batch, max_time_src], the indices of source sequence
            symbols in a batch.
         'src_seq_lens': [batch], the lengths of unpadded source sequences in
            a batch.
         'tgt_sos_id': an int scalar tensor, the index of sos in target
            vocabulary.
         'tgt_eos_id': an int scalar tensor, the index of sos in target
            vocabulary.}
    """
    if self.mode != tf.contrib.learn.ModeKeys.INFER:
      raise ValueError('Must be in infer mode when calling',
          '`_get_tensor_dict_infer`.')
    src_dataset = tf.data.TextLineDataset(src_file_list) 
    src_vocab_table, tgt_vocab_table = self._get_vocab_tables(src_vocab_file,
                                                              tgt_vocab_file)
    tgt_sos_id, tgt_eos_id = self._get_tgt_marker_indices(tgt_vocab_table)

    src_dataset = src_dataset.map(lambda src: tf.string_split([src]).values)
    if self._src_max_len:
      src_dataset = src_dataset.map(lambda src: src[:self._src_max_len])
    src_dataset = src_dataset.map(
        lambda src: tf.to_int32(src_vocab_table.lookup(src)))
    src_dataset = src_dataset.map(lambda src: (src, tf.size(src)))

    def batching_fn(dataset):
      return dataset.padded_batch(
          self._batch_size,
          padded_shapes=dataset.output_shapes)

    batched_dataset = batching_fn(src_dataset)
    iterator = batched_dataset.make_initializable_iterator()
    src_input_ids, src_seq_lens = iterator.get_next()

    tensor_dict = {'src_input_ids': src_input_ids,
                   'src_seq_lens': src_seq_lens,
                   'tgt_sos_id': tgt_sos_id,
                   'tgt_eos_id': tgt_eos_id}
    self._iterator_initializer = iterator.initializer
    return tensor_dict


class TrainerSeq2SeqDataset(Seq2SeqDataset):
  """Dataset for seq2seq model trainer."""
  def __init__(self,
               batch_size,
               src_vocab_size,
               tgt_vocab_size,
               shuffle_buffer_size=10000,
               num_buckets=5,
               src_max_len=50,
               tgt_max_len=50,
               sos='<s>',
               eos='</s>',
               random_seed=0):
    """Constructor.

    Args:
      batch_size: int scalar, batch size.
      src_vocab_size: int scalar, num of symbols in source vocabulary.
      tgt_vocab_size: int scalar, num of symbols in target vocabulary.
      shuffle_buffer_size: int scalar, buffer size for shuffling the dataset.
        Must be large enough to ensure sufficiently randomized dataset. If None,
        no shuffling is performed.
      num_buckets: int scalar, the num of buckets containing sequences within 
        different length ranges. If 1, no bucketing is performed.
      src_max_len: int scalar or None, used to truncate source sequence to a 
        max length. If None, no truncating is performed. 
      tgt_max_len: int scalar or None, used to truncate target sequence to a 
        max length. If None, no truncating is performed. 
      sos: string scalar, the start-of-sentence marker.
      eos: string scalar, the end-of-sentence marker.
      random_seed: int scalar, random seed.
    """
    super(TrainerSeq2SeqDataset, self).__init__(
        batch_size, src_vocab_size, tgt_vocab_size, src_max_len, sos, eos)
    self._shuffle_buffer_size = shuffle_buffer_size
    self._num_buckets = num_buckets
    self._tgt_max_len = tgt_max_len
    self._random_seed = random_seed

  @property
  def mode(self):
    return tf.contrib.learn.ModeKeys.TRAIN

  def get_tensor_dict(self,
                      src_file_list,
                      tgt_file_list,
                      src_vocab_file,
                      tgt_vocab_file,
                      scope=None):
    """Generates a tensor dict to be used by `Trainer`.

    Args:
      src_file_list: a list of string scalars, the paths to the source sequence
        text files. 
      tgt_file_list: a list of string scalars, the paths to the corresponding
        target sequence text files.
      src_vocab_file: string scalar, the path to the source vocabulary file,
        where each line contains a single symbol. 
      tgt_vocab_file: string scalar, the path to the target vocabulary file, 
        where each line contains a single symbol. 
      scope: string scalar or None, name scope.

    Returns:
      tensor_dict: a dict mapping from tensor names to tensors
        {'src_input_ids': [batch, max_time_src], the indices of source sequence
            symbols in a batch.
         'tgt_input_ids': [batch, max_time_tgt], the indices of target input
            sequence symbols in a batch.
         'tgt_output_ids': [batch, max_time_tgt], the indices of target output
            sequence symbols in a batch.
         'src_seq_lens': [batch], the lengths of unpadded source sequences in
            a batch.
         'tgt_seq_lens': [batch], the lengths of unpadded target sequences in
            a batch.}
    """
    with tf.name_scope(scope, self._get_tensor_dict_scope, values=[
        src_file_list, tgt_file_list, src_vocab_file, tgt_vocab_file]):
      return self._get_tensor_dict_train_eval(
          src_file_list, tgt_file_list, src_vocab_file, tgt_vocab_file)


class EvaluatorSeq2SeqDataset(Seq2SeqDataset):
  """Dataset for seq2seq model evaluator."""
  def __init__(self,
               batch_size,
               src_vocab_size,
               tgt_vocab_size,
               num_buckets=1,
               src_max_len=None,
               tgt_max_len=None,
               sos='<s>',
               eos='</s>'):
    """Constructor.

    Args:
      batch_size: int scalar, batch size.
      src_vocab_size: int scalar, num of symbols in source vocabulary.
      tgt_vocab_size: int scalar, num of symbols in target vocabulary.
      num_buckets: int scalar, the num of buckets containing sequences within 
        different length ranges. If 1, no bucketing is performed.
      src_max_len: int scalar or None, used to truncate source sequence to a 
        max length. If None, no truncating is performed. 
      tgt_max_len: int scalar or None, used to truncate target sequence to a 
        max length. If None, no truncating is performed. 
      sos: string scalar, the start-of-sentence marker.
      eos: string scalar, the end-of-sentence marker.
    """
    super(EvaluatorSeq2SeqDataset, self).__init__(
        batch_size, src_vocab_size, tgt_vocab_size, src_max_len, sos, eos)
    self._num_buckets = num_buckets
    self._tgt_max_len = tgt_max_len

  @property
  def mode(self):
    return tf.contrib.learn.ModeKeys.EVAL

  def get_tensor_dict(self,
                      src_file_list,
                      tgt_file_list,
                      src_vocab_file,
                      tgt_vocab_file,
                      scope=None):
    """Generates a tensor dict to be used by `Evaluator`.

    Args:
      src_file_list: a list of string scalars, the paths to the source sequence
        text files. 
      tgt_file_list: a list of string scalars, the paths to the corresponding
        target sequence text files.
      src_vocab_file: string scalar, the path to the source vocabulary file,
        where each line contains a single symbol. 
      tgt_vocab_file: string scalar, the path to the target vocabulary file, 
        where each line contains a single symbol. 
      scope: string scalar or None, name scope.

    Returns:
      tensor_dict: a dict mapping from tensor names to tensors
        {'src_input_ids': [batch, max_time_src], the indices of source sequence
            symbols in a batch.
         'tgt_input_ids': [batch, max_time_tgt], the indices of target input
            sequence symbols in a batch.
         'tgt_output_ids': [batch, max_time_tgt], the indices of target output
            sequence symbols in a batch.
         'src_seq_lens': [batch], the lengths of unpadded source sequences in
            a batch.
         'tgt_seq_lens': [batch], the lengths of unpadded target sequences in
            a batch.}
    """
    with tf.name_scope(scope, self._get_tensor_dict_scope, values=[
        src_file_list, tgt_file_list, src_vocab_file, tgt_vocab_file]):
      return self._get_tensor_dict_train_eval(
          src_file_list, tgt_file_list, src_vocab_file, tgt_vocab_file)


class InferencerSeq2SeqDataset(Seq2SeqDataset):
  """Dataset for seq2seq model inferencer."""

  @property
  def mode(self):
    return tf.contrib.learn.ModeKeys.INFER

  def get_tensor_dict(self,
                      src_file_list,
                      src_vocab_file,
                      tgt_vocab_file,
                      scope=None):
    """Generates a tensor dict to be used by `Inferencer`.

    Args:
      src_file_list: a list of string scalars, the paths to the source sequence
        text files.
      src_vocab_file: string scalar, the path to the source vocabulary file,
        where each line contains a single symbol.
      tgt_vocab_file: string scalar, the path to the target vocabulary file, 
        where each line contains a single symbol.
      scope: string scalar or None, name scope.

    Returns:
      tensor_dict: a dict mapping from tensor names to tensors
        {'src_input_ids': [batch, max_time_src], the indices of source sequence
            symbols in a batch.
         'src_seq_lens': [batch], the lengths of unpadded source sequences in
            a batch.
         'tgt_sos_id': an int scalar tensor, the index of sos in target
            vocabulary.
         'tgt_eos_id': an int scalar tensor, the index of sos in target
            vocabulary.}
    """
    with tf.name_scope(scope, self._get_tensor_dict_scope,
        values=[src_file_list, src_vocab_file, tgt_vocab_file]):
      return self._get_tensor_dict_infer(
          src_file_list, src_vocab_file, tgt_vocab_file)

  def get_rev_tgt_vocab_table(self, tgt_vocab_file, scope=None):
    """Builds the reverse vocabulary table for target sequence.
    Reverse vocabulary table is a table mapping from symbol indices to symbols.

    Args:
      tgt_vocab_file: string scalar or None, the path to the target vocabulary 
        file, where each line contains a single symbol. 
      scope: string scalar or None, name scope.

    Returns:
      rev_tgt_vocab_table: a HashTable instance, the reverse target vocabulary.
    """
    with tf.name_scope(scope, 'ReverseTgtVocabTable', values=[tgt_vocab_file]):
      rev_tgt_vocab_table = tf.contrib.lookup.index_to_string_table_from_file(
          tgt_vocab_file, default_value=UNK)
      return rev_tgt_vocab_table
 
