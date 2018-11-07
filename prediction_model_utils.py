import tensorflow as tf

VOCAB_SIZE_THRESHOLD_CPU = 50000


def _get_embed_device(vocab_size):
  """Returns the name of the device to place embedding matrix variables, 
  based on vocabulary size.
  """
  if vocab_size > VOCAB_SIZE_THRESHOLD_CPU:
    return "/cpu:0"
  else:
    return "/gpu:0"


def create_embedding(vocab_size, embed_size, name, dtype=tf.float32):
  """Creates embedding matrix [vocab_size, embed_size] for a vocabulary.
  Depending on the vocabulary size, the created variables are pinned to
  either GPU or CPU.

  Args:
    vocab_size: int scalar, num of symbols in vocabulary.
    embed_size: int scalar, size of embedding vector.
    name: string scalar, name of the embedding matrix variable.
    dtype: data type of embedding matrix. Defaults to tf.float32.

  Returns:
    embedding: float tensor with shape [vocab_size, embed_size]
  """
  with tf.device(_get_embed_device(vocab_size)):
    embedding = tf.get_variable(name, [vocab_size, embed_size], dtype)
    return embedding


def create_encoder_and_decoder_embeddings(src_vocab_size,
                                          tgt_vocab_size,
                                          src_embed_size,
                                          tgt_embed_size,
                                          share_vocab=False,
                                          scope=None):
  """Creates embeddings for encoder (source) and decoder (target).

  Args:
    src_vocab_size: int scalar, num of symbols in the source vocabulary.
    tgt_vocab_size: int scalar, num of symbols in the target vocabulary.
    src_embed_size: int scalar, size of embedding vector for source.
    tgt_embed_size: int scalar, size of embedding vector for target.
    share_vocab: bool scalar, if True, source and target vocabularies are 
      assumed to be the same, and only one embedding matrix is created and 
      shared by both. Defaults to False.
    scope: string scalar, scope name.

  Returns:
    encoder_embedding: float tensor with shape [src_vocab_size, num_units]
      containing the embeddings of symbols in source vocabulary. 
    decoder_embedding: float tensor with shape [tgt_vocab_size, num_units]
      containing the embeddings of symbols in target vocabulary.
  """
  with tf.variable_scope(scope, 'embeddings', 
      [src_vocab_size, tgt_vocab_size, src_embed_size, tgt_embed_size]):
    if share_vocab:
      encoder_embedding = decoder_embedding = create_embedding(
          src_vocab_size, src_embed_size, name='shared_embedding')
    else:
      with tf.variable_scope('encoder'):
        encoder_embedding = create_embedding(src_vocab_size, src_embed_size,
            name='embedding_encoder')
      with tf.variable_scope('decoder'):
        decoder_embedding = create_embedding(tgt_vocab_size, tgt_embed_size,
            name='embedding_decoder')
  return encoder_embedding, decoder_embedding
  

def create_single_cell(unit_type='lstm',
                       num_units=128,
                       forget_bias=1.0,
                       keep_prob=0.5,
                       residual_connection=False):
  """Creates a single RNN Cell.

  Args:
    unit_type: string scalar, the type of RNN cell ('lstm', 'gru', etc.).
    num_units: int scalar, the num of units in an RNN Cell.
    forget_bias: float scalar, forget bias in LSTM Cell. Defaults to 1.0.
    keep_prob: float scalar, dropout rate equals `1 - keep_prob`.
    residual_connection: bool scalar, whether to add residual connection linking
      the input and output of a RNN Cell.

  Returns:
    an RNN Cell instance.
  """
  if unit_type == 'lstm':
    single_cell = tf.nn.rnn_cell.LSTMCell(
        num_units=num_units, forget_bias=forget_bias, name='basic_lstm_cell')
  elif unit_type == 'gru':
    single_cell = tf.contrib.rnn.GRUCell(num_units)
  elif unit_type == 'nas':
    single_cell = tf.contrib.rnn.NASCell(num_units)
  else:
    raise ValueError('Unknown RNN unit type: {}'.format(unit_type))

  if keep_prob < 1.0:
    single_cell = tf.contrib.rnn.DropoutWrapper(
        cell=single_cell, input_keep_prob=keep_prob)
  if residual_connection:
    single_cell = tf.contrib.rnn.ResidualWrapper(single_cell)
  return single_cell


def create_cell_list(unit_type,
                     num_units,
                     forget_bias,
                     keep_prob,
                     num_layers,
                     num_res_layers):
  """Creates a list of RNN Cells for a multi-layer RNN.

  Args:
    unit_type: string scalar, the type of RNN cell ('lstm', 'gru', etc.).
    num_units: int scalar, the num of units in an RNN Cell.
    forget_bias: float scalar, forget bias in LSTM Cell. Defaults to 1.0.
    keep_prob: float scalar, dropout rate equals `1 - keep_prob`.
    num_layers: int scalar, the num of layers in a multi-layer RNN.
    num_res_layers: int scalar, the num of layers in a multi-layer RNN with 
      residual connections.

  Returns:
    a list of RNN Cell instances. 
  """
  cell_list = [create_single_cell(
      unit_type,
      num_units,
      forget_bias,
      keep_prob,
      residual_connection=(i>=num_layers-num_res_layers))
          for i in range(num_layers)]
  return cell_list


def create_rnn_cell(unit_type,
                    num_units,
                    forget_bias,
                    keep_prob,
                    num_layers,
                    num_res_layers):
  """Creates an RNN Cell for a single layer or multi-layer RNN.

  Args:
    unit_type: string scalar, the type of RNN cell ('lstm', 'gru', etc.).
    num_units: int scalar, the num of units in an RNN Cell.
    forget_bias: float scalar, forget bias in LSTM Cell. Defaults to 1.0.
    keep_prob: float scalar, dropout rate equals `1 - keep_prob`.
    num_layers: int scalar, the num of layers in a multi-layer RNN.
    num_res_layers: int scalar, the num of layers in a multi-layer RNN with 
      residual connections.

  Returns:
    an RNN Cell instance.
  """
  cell_list = create_cell_list(unit_type,
                               num_units,
                               forget_bias,
                               keep_prob,
                               num_layers,
                               num_res_layers)
  if len(cell_list) == 1:
    return cell_list[0]
  else:
    return tf.contrib.rnn.MultiRNNCell(cell_list)

