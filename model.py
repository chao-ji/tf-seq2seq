"""Defines Seq2Seq model augmented with attention mechanism in tf.keras API."""
import numpy as np
import tensorflow as tf

import utils
from commons.tokenization import SOS_ID
from commons.tokenization import EOS_ID
from commons.beam_search import NEG_INF
from commons.layers import EmbeddingLayer
from commons import beam_search


class Encoder(tf.keras.layers.Layer):
  """The Encoder that consists of a bidirectional one-layer LSTM RNN."""
  def __init__(self, hidden_size, dropout_rate):
    """Constructor.

    Args:
      hidden_size: int scalar, the hidden size of continuous representation. 
      dropout_rate: float scalar, dropout rate for the Dropout layers.
    """
    super(Encoder, self).__init__()
    self._hidden_size = hidden_size
    self._dropout_rate = dropout_rate

    self._recurrent_layer = tf.keras.layers.Bidirectional(tf.keras.layers.RNN(
        [tf.keras.layers.LSTMCell(self._hidden_size, dropout=dropout_rate)],
        return_sequences=True,
        return_state=True))

  def call(self, src_token_embeddings, padding_mask, training=False):
    """Computes the output of the encoder RNN layers.

    Args:
      src_token_embeddings: float tensor of shape [batch_size, src_seq_len, 
        hidden_size], the embeddings of source sequence tokens.
      padding_mask: float tensor of shape [batch_size, src_seq_len], populated 
        with either 0 (for tokens to keep) or 1 (for tokens to be masked).
      training: bool scalar, True if in training mode. 

    Returns:
      encoder_outputs: float tensor of shape [batch_size, src_seq_len, 
        hidden_size*2], the encoded source sequences to be used as reference.
      fw_states: a length-2 list of tensors of shape [[batch_size, hidden_size], 
        [batch_size, hidden_size]], the "hidden" and "cell" state of the forward
        going LSTM layer.
      bw_states: a length-2 list of tensors of shape [[batch_size, hidden_size],
        [batch_size, hidden_size]], the "hidden" and "cell" state of the 
        backward going LSTM layer.
    """
    encoder_outputs, fw_states, bw_states = self._recurrent_layer(
        src_token_embeddings, 
        mask=1 - padding_mask,
        training=training)
    return encoder_outputs, fw_states, bw_states


class Decoder(tf.keras.layers.Layer):
  """Decoder that consists of a stacked 2-layer LSTM augmented with attention 
  mechanism.
  """
  def __init__(self, hidden_size, decoder_cell):
    """Constructor.

    Args:
      hidden_size: int scalar, the hidden size of continuous representation. 
      decoder_cell: an instence of DecoderCell, the decoder cell.
    """
    super(Decoder, self).__init__()
    self._hidden_size = hidden_size
    self._decoder_cell = decoder_cell

    self._recurrent_layer = tf.keras.layers.RNN(
        decoder_cell, return_sequences=True)

  def call(self, 
           tgt_token_embeddings, 
           fw_states, 
           bw_states, 
           encoder_outputs, 
           padding_mask, 
           training=False):
    """Computes the output of the decoder RNN layers.

    Args:
      tgt_token_embeddings: float tensor of shape [batch_size, tgt_seq_len, 
        hidden_size], the embeddings of target sequence tokens. 
      fw_states: a length-2 list of tensors of shape [[batch_size, hidden_size], 
        [batch_size, hidden_size]], the "hidden" and "cell" state of the forward
        going LSTM layer.
      bw_states: a length-2 list of tensors of shape [[batch_size, hidden_size],
        [batch_size, hidden_size]], the "hidden" and "cell" state of the 
        backward going LSTM layer.
      encoder_outputs: float tensor of shape [batch_size, src_seq_len, 
        hidden_size*2], the encoded source sequences to be used as reference.
      padding_mask: float tensor of shape [batch_size, src_seq_len], 
        populated with either 0 (for tokens to keep) or 1 (for tokens to be 
        masked).
      training: bool scalar, True if in training mode.
        
    Returns:
      decoder_outputs: float tensor of shape [batch_size, tgt_seq_len, 
        hidden_size], the sequences in continuous representation.
    """
    decoder_outputs = self._recurrent_layer(
        tgt_token_embeddings, 
        [fw_states,
         bw_states,
         tf.zeros((tgt_token_embeddings.shape[0], self._hidden_size), 
             dtype='float32')],
        [encoder_outputs, padding_mask], 
        training=training)

    return decoder_outputs


class BahdanauAttention(tf.keras.layers.Layer):
  """Bahdanau-style additive attention mechanism."""
  def __init__(self, hidden_size):
    """Constructor.

    Args:
      hidden_size: int scalar, the hidden size of continuous representation.
    """
    super(BahdanauAttention, self).__init__()
    self._hidden_size = hidden_size

    self._dense_query = tf.keras.layers.Dense(self._hidden_size,
        use_bias=False,
        kernel_initializer='glorot_uniform')
    self._dense_memory = tf.keras.layers.Dense(self._hidden_size,
        use_bias=False,
        kernel_initializer='glorot_uniform')
    self._dense_score = tf.keras.layers.Dense(1, 
        use_bias=False,
        kernel_initializer='glorot_uniform')
    self._dense_attention = tf.keras.layers.Dense(self._hidden_size,
        use_bias=False,
        kernel_initializer='glorot_uniform')

  def call(self, query, encoder_outputs, padding_mask):
    """Computes the outputs of Bahdanau-style attention mechanism.

    Args:
      query: float tensor of shape [batch_size, hidden_size], the embeddings
        of tokens at a single decoding step.
      encoder_outputs: float tensor of shape [batch_size, src_seq_len, 
        hidden_size*2], the encoded source sequences to be used as reference.
      padding_mask: float tensor of shape [batch_size, src_seq_len], populated 
        with either 0 (for tokens to keep) or 1 (for tokens to be masked).

    Returns:
      attention_outputs: float tensor of shape [batch_size, hidden_size], the
        outputs of attention mechanism.
      attention_weights: float tensor of shape [batch_size, src_seq_len], the
        amount of attention to tokens in source sequences.
    """
    reference = self._dense_memory(encoder_outputs)

    attention_weights = tf.squeeze(self._dense_score(tf.nn.tanh(
        self._dense_query(tf.expand_dims(query, axis=1)) + 
        reference)), axis=2)
    attention_weights += padding_mask * NEG_INF
    attention_weights = tf.nn.softmax(attention_weights, axis=1)

    context = tf.reduce_sum(
        tf.expand_dims(attention_weights, axis=2) * reference, axis=1)
    attention_outputs = self._dense_attention(
        tf.concat([query, context], axis=1))
    return attention_outputs, attention_weights


class LuongAttention(tf.keras.layers.Layer):
  """Luong-style multiplicative attention mechanism."""
  def __init__(self, hidden_size):
    """Constructor.

    Args:
      hidden_size: int scalar, the hidden size of continuous representation.
    """
    super(LuongAttention, self).__init__()
    self._hidden_size = hidden_size

    self._dense_memory = tf.keras.layers.Dense(
        self._hidden_size, 
        use_bias=False,
        kernel_initializer='glorot_uniform')
    self._dense_attention = tf.keras.layers.Dense(
        self._hidden_size, 
        use_bias=False,
        kernel_initializer='glorot_uniform')

  def call(self, query, encoder_outputs, padding_mask):
    """Computes the outputs of Luong-style attention mechanism.

    Args:
      query: float tensor of shape [batch_size, hidden_size], the embeddings
        of tokens at a single decoding step.
      encoder_outputs: float tensor of shape [batch_size, src_seq_len, 
        hidden_size*2], the encoded source sequences to be used as reference.
      padding_mask: float tensor of shape [batch_size, src_seq_len], populated 
        with either 0 (for tokens to keep) or 1 (for tokens to be masked).

    Returns:
      attention_outputs: float tensor of shape [batch_size, hidden_size], the
        outputs of attention mechanism.
      attention_weights: float tensor of shape [batch_size, src_seq_len], the
        amount of attention to tokens in source sequences.
    """
    batch_size, hidden_size = query.shape

    # [batch_size, src_seq_len, hidden_size]
    reference = self._dense_memory(encoder_outputs)

    # [batch_size, src_seq_len]
    attention_weights = tf.reduce_sum(
        reference * tf.expand_dims(query, axis=1), axis=2)
    attention_weights += padding_mask * NEG_INF
    attention_weights = tf.nn.softmax(attention_weights, axis=1)

    # [batch_size, 2*hidden_size]
    context = tf.reduce_sum(tf.expand_dims(attention_weights, axis=2)
        * encoder_outputs, axis=1)

    # [batch_size, hidden_size]
    attention_outputs = self._dense_attention(
        tf.concat([query, context], axis=1))

    return attention_outputs, attention_weights


class DecoderCell(tf.keras.layers.AbstractRNNCell):
  """Decoder cell that consists of a stacked 2-layer LSTM cell augmented with
  attention mechanism.
  """
  def __init__(
      self, hidden_size, dropout_rate, attention_model='luong', **kwargs):
    """Constructor.

    Args:
      hidden_size: int scalar, the hidden size of continuous representation. 
      dropout_rate: float scalar, dropout rate for the Dropout layers.
      kwargs: dict, additional optional keyword arguments.
    """
    super(DecoderCell, self).__init__(**kwargs)
    self._hidden_size = hidden_size

    self._dropout_layer = tf.keras.layers.Dropout(dropout_rate)
    self._attention = (LuongAttention(hidden_size) if attention_model == 'luong'
                       else BahdanauAttention(hidden_size))

  @property
  def state_size(self):
    """Returns the sizes of the state tensors."""
    return [(self._hidden_size, self._hidden_size),
            (self._hidden_size, self._hidden_size),
            (self._hidden_size,)]

  def build(self, input_shape):
    """Creates weights of stacked 2-layer LSTM cell.

    Args:
      input_shape: unused. 
    """
    weights1 = self._build_layer(self._hidden_size*2, self._hidden_size, 1)
    weights2 = self._build_layer(self._hidden_size, self._hidden_size, 2)

    self._kernel_inputs = [weights1[0], weights2[0]]
    self._kernel_recurrent = [weights1[1], weights2[1]]
    self._biases = [weights1[2], weights2[2]]
    
  def _build_layer(self, input_size, hidden_size, layer_index):
    """Creates weights of a single LSTM cell.

    Args:
      input_size: int scalar, hidden size of the input token embedding.
      hidden_size: int scalar, hidden size of the recurrent states.
      layer_index: int scalar, index of the layer (1 or 2).

    Returns:
      kernel_inputs: float tensor of shape [input_size, hidden_size * 4], 
        the LSTM kernel for the input tensor.
      kernel_recurrent: float tensor of shape [hidden_size, hidden_size * 4],
        the LSTM kernel for the recurrent tensor.
      biases: float tensor of shape [hidden_size * 4], the biases for the LSTM 
        layer.
    """
    kernel_inputs = self.add_weight(
        shape=(input_size, self._hidden_size*4),
        initializer='glorot_uniform',
        name='kernel_inputs_%d' % layer_index)
    kernel_recurrent = self.add_weight(
        shape=(hidden_size, self._hidden_size*4),
        initializer='glorot_uniform',
        name='kernel_recurrent_%d' % layer_index)

    biases_value = np.zeros((self._hidden_size*4))
    biases_value[self._hidden_size:self._hidden_size*2] = 1
    biases = self.add_weight(
        shape=self._hidden_size*4, 
        initializer=tf.keras.initializers.Constant(biases_value),
        name='biases_%d' % layer_index)

    return kernel_inputs, kernel_recurrent, biases

  def _run_single_layer(self, inputs, h, c, layer_index):
    """Runs the forward pass of a single LSTM layer.

    Args:
      inputs: float tensor of shape [batch_size, input_size], the input tensor.
      h: float tensor of shape [batch_size, hidden_size], the "hidden" state
        of LSTM.
      c: float tensor of shape [batch_size, hidden_size], the "cell" state of 
        LSTM.
      layer_index: int scalar, index of the layer (1 or 2). 

    Returns:
      h: float tensor of shape [batch_size, hidden_size], the updated "hidden" 
        state of LSTM.
      c: float tensor of shape [batch_size, hidden_size], the updated "cell" 
        state of LSTM.
    """
    logits = tf.matmul(inputs, self._kernel_inputs[layer_index]) + tf.matmul(
        h, self._kernel_recurrent[layer_index]) + self._biases[layer_index]
    i, f, j, o = tf.split(logits, 4, axis=1)
    c = tf.sigmoid(f) * c + tf.sigmoid(i) * tf.tanh(j)
    h = tf.sigmoid(o) * tf.tanh(c)
    return h, c

  def call(self, inputs, states, constants, cache=None):
    """Computes the outputs of decoder cell.

    Args:
      inputs: float tensor of shape [batch_size, hidden_size], the input tensor.
      states: a 3-tuple of float tensors of shape
        ([[batch_size, hidden_size], [batch_size, hidden_size]],
         [[batch_size, hidden_size], [batch_size, hidden_size]],
         [batch_size, hidden_size]), the "hidden" and "cell" state of layer 1
        and layer 2 of the stacked LSTM cells, plus the attention outputs.
      constants: a 2-tuple of float tensors of shape
        ([batch_size, src_seq_len, 2*hidden_size], [batch_size, src_seq_len]),
        the encoder outputs and source sequence padding mask.
      cache: (Optional) a dict with `tgt_src_attention` as key, mapping to a 
        tensor of shape [batch_size, tgt_seq_len, src_seq_len] as value. Used in 
        inference mode only.

    Returns:
      outputs: float tensor of shape [batch_size, hidden_size], the decoder cell
        outputs (which is also the attention outputs).
      states_list: a 3-tuple of float tensors of shape
        ([[batch_size, hidden_size], [batch_size, hidden_size]],
         [[batch_size, hidden_size], [batch_size, hidden_size]],
         [batch_size, hidden_size]), the updated "hidden" and "cell" state of 
        layer 1 and layer 2 of the stacked LSTM layers, plus the attention 
        outputs. 
    """
    encoder_outputs, padding_mask = constants
    inputs = tf.concat([inputs, states[2]], axis=1)

    h1, c1 = self._run_single_layer(inputs, states[0][0], states[0][1], 0)
    h1 = self._dropout_layer(h1)
    h2, c2 = self._run_single_layer(h1, states[1][0], states[1][1], 1) 
    h2 = self._dropout_layer(h2) 

    outputs, weights = self._attention(h2, encoder_outputs, padding_mask)
    if cache is not None:
      cache['tgt_src_attention'] = tf.concat(
          [cache['tgt_src_attention'], tf.expand_dims(weights, axis=1)], axis=1)
    return outputs, ([h1, c1], [h2, c2], outputs)
    

class Seq2SeqModel(tf.keras.Model):
  """Seq2Seq model augmented with attention mechanism.

  The model implements methods `call` and `transduce`, where
    - `call` is invoked in training mode, taking as input BOTH the source and 
      target token ids, and returning the estimated logits for the target token 
      ids.
    - `transduce` is invoked in inference mode, taking as input the source token 
      ids ONLY, and outputting the token ids of the decoded target sequences 
      using beam search. 
  """
  def __init__(self, 
               vocab_size, 
               hidden_size,
               attention_model='luong',
               dropout_rate=0.2,
               extra_decode_length=50,
               beam_width=4,
               alpha=0.6):
    """Constructor.

    Args:
      vocab_size: int scalar, num of subword tokens (including SOS/PAD and EOS) 
        in the vocabulary.
      hidden_size: int scalar, the hidden size of continuous representation.
      dropout_rate: float scalar, dropout rate for the Dropout layers.
      extra_decode_length: int scalar, the max decode length would be the sum of
        `tgt_seq_len` and `extra_decode_length`.
      beam_width: int scalar, beam width for beam search.
      alpha: float scalar, the parameter for length normalization used in beam 
        search.      
    """
    super(Seq2SeqModel, self).__init__()
    self._vocab_size = vocab_size
    self._hidden_size = hidden_size
    self._attention_model = attention_model
    self._dropout_rate = dropout_rate
    self._extra_decode_length = extra_decode_length
    self._beam_width = beam_width
    self._alpha = alpha 

    self._encoder = Encoder(hidden_size, dropout_rate)
    self._decoder_cell = DecoderCell(hidden_size, dropout_rate, attention_model)
    self._decoder = Decoder(
        hidden_size, self._decoder_cell)
    self._embedding_logits_layer = EmbeddingLayer(vocab_size, hidden_size)

  def call(self, src_token_ids, tgt_token_ids):
    """Takes as input the source and target token ids, and returns the estimated
    logits for the target sequences. Note this function should be called in 
    training mode only.

    Args:
      src_token_ids: int tensor of shape [batch_size, src_seq_len], token ids
        of source sequences.
      tgt_token_ids: int tensor of shape [batch_size, tgt_seq_len], token ids 
        of target sequences.

    Returns:
      logits: float tensor of shape [batch_size, tgt_seq_len, vocab_size].
    """
    padding_mask = utils.get_padding_mask(src_token_ids)

    src_token_embeddings = self._embedding_logits_layer(
        src_token_ids, 'embedding')
    tgt_token_embeddings = self._embedding_logits_layer(
        tgt_token_ids, 'embedding')

    encoder_outputs, fw_states, bw_states = self._encoder(
        src_token_embeddings, padding_mask, training=True)
    decoder_outputs = self._decoder(tgt_token_embeddings,
                                    fw_states,
                                    bw_states,
                                    encoder_outputs,
                                    padding_mask,
                                    training=True)

    logits = self._embedding_logits_layer(decoder_outputs, 'logits') 
    return logits

  def transduce(self, src_token_ids):
    """Takes as input the source token ids only, and outputs the token ids of 
    the decoded target sequences using beam search. Note this function should be 
    called in inference mode only.

    Args:
      src_token_ids: int tensor of shape [batch_size, src_seq_len], token ids
        of source sequences.

    Returns:
      decoded_ids: int tensor of shape [batch_size, decoded_seq_len], the token
        ids of the decoded target sequences using beam search.
      scores: float tensor of shape [batch_size], the scores (length-normalized 
        log-probs) of the decoded target sequences.
      tgt_src_attention: a list of `decoder_stack_size` float tensor of shape 
        [batch_size, num_heads, decoded_seq_len, src_seq_len], target-to-source 
        attention weights.
      tgt_src_attention: float tensor of shape [batch_size, tgt_seq_len, 
        src_seq_len], the target-to-source attention weights.
    """
    batch_size, src_seq_len = src_token_ids.shape
    hidden_size = self._hidden_size
    max_decode_length = src_seq_len + self._extra_decode_length
    decoding_fn = self._build_decoding_fn()

    src_token_embeddings = self._embedding_logits_layer(
        src_token_ids, 'embedding')
    padding_mask = utils.get_padding_mask(src_token_ids)
    encoder_outputs, fw_states, bw_states = self._encoder(
        src_token_embeddings, padding_mask, training=False)
    decoding_cache = {'fw_states': fw_states,
                      'bw_states': bw_states,
                      'attention_states': tf.zeros((batch_size, hidden_size)),
                      'encoder_outputs': encoder_outputs,
                      'padding_mask': padding_mask,
                      'tgt_src_attention': tf.zeros((batch_size, 0, src_seq_len))
                      }
    sos_ids = tf.ones([batch_size], dtype='int32') * SOS_ID

    bs = beam_search.BeamSearch(decoding_fn,
                                self._vocab_size,
                                batch_size,
                                self._beam_width,
                                self._alpha,
                                max_decode_length,
                                EOS_ID)

    decoded_ids, scores, decoding_cache = bs.search(sos_ids, decoding_cache)           

    tgt_src_attention = decoding_cache['tgt_src_attention'].numpy()[:, 0]

    decoded_ids = decoded_ids[:, 0, 1:]
    scores = scores[:, 0]
    return decoded_ids, scores, tgt_src_attention

  def _build_decoding_fn(self):
    """Builds the decoding function that will be called in beam search.

    The function steps through the proposed token ids one at a time, and 
    generates the logits of next token id over the vocabulary.

    Returns:
      decoding_fn: a callable that outputs the logits of the next decoded token
        ids.
    """
    def decoding_fn(decoder_input, cache, **kwargs):
      """Computes the logits of the next decoded token ids.

      Args:
        decoder_input: int tensor of shape [batch_size * beam_width, 1], decoder
          input.
        cache: dict of entries 
          'fw_states': a length-2 list of tensors of shape 
            [[batch_size*beam_width, hidden_size], 
             [batch_size*beam_width, hidden_size]]
          'bw_states': a length-2 list of tensors of shape 
            [[batch_size*beam_width, hidden_size], 
             [batch_size*beam_width, hidden_size]]
          'attention_states': float tensor of shape 
            [batch_size*beam_width, hidden_size] 
          'encoder_outputs': float tensor of shape 
            [batch_size*beam_width, src_seq_len, hidden_size*2]
          'padding_mask': float tensor of shape 
            [batch_size*beam_width, src_seq_len].
          'tgt_src_attention': float tensor of shape 
            [batch_size*beam_width, seq_len, src_seq_len].
        kwargs: dict, storing (optional) additional keyword arguments.

      Returns:
        logits: float tensor of shape [batch_size * beam_width, vocab_size].
        cache: a dict with the same structure as the input `cache`, except that
          the shapes of the value of `tgt_src_attention` is 
          [batch_size*beam_width, seq_len+1, src_seq_len].
      """
      fw_states = cache['fw_states']
      bw_states = cache['bw_states']
      attention_states = cache['attention_states']
      encoder_outputs = cache['encoder_outputs']
      padding_mask = cache['padding_mask']

      # [batch_size * beam_width, 1, hidden_size]
      decoder_input = self._embedding_logits_layer(decoder_input, 'embedding')
      # [batch_size * beam_width, hidden_size]
      decoder_input = tf.squeeze(decoder_input, axis=1)

      decoder_outputs, states = self._decoder_cell(
          decoder_input, 
          (fw_states, bw_states, attention_states),
          (encoder_outputs, padding_mask),
          cache=cache)
      fw_states, bw_states, attention_states = states

      decoder_outputs = tf.expand_dims(decoder_outputs, axis=1)
      logits = self._embedding_logits_layer(decoder_outputs, 'logits')
      logits = tf.squeeze(logits, axis=1)

      cache['fw_states'] = fw_states
      cache['bw_states'] = bw_states
      cache['attention_states'] = attention_states
      return logits, cache
    return decoding_fn
