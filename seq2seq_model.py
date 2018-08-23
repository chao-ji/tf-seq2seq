from abc import ABCMeta
from abc import abstractmethod

import tensorflow as tf

import prediction_model_utils as utils


class Seq2SeqPredictionModel(object):
  """Runs the source input sequences (and target input sequences in train and
  eval mode) through the forward pass of a seq-to-seq model to obtain prediction
  logits (in train and eval mode) or decoded symbol indices (in infer mode).

  Must implement `predict_logits` (in train and eval mode) and `predict_indices`
  (in infer mode) methods by subclasses.
  """

  __metaclass__ = ABCMeta

  def __init__(self,
               unit_type,
               num_units,
               forget_bias,
               keep_prob,
               encoder_type,
               time_major,
               share_vocab,
               is_inferring,
               initializer,
               src_vocab_size,
               tgt_vocab_size,
               num_encoder_layers,
               num_encoder_res_layers,
               num_decoder_layers,
               num_decoder_res_layers):
    """Constructor.

    Args:
      unit_type: string scalar, the type of RNN cell ('lstm', 'gru', etc.).
      num_units: int scalar, the num of units in an RNN Cell.
      forget_bias: float scalar, forget bias in LSTM Cell. Defaults to 1.0.
      keep_prob: float scalar, dropout rate equals 1 - `keep_prob`.
      encoder_type: string scalar, 'uni' (unidirectional RNN as encoder) or 
        'bi' (bidirectional RNN as encode).
      time_major: bool scalar, whether the output tensors are in time major 
        format (the first two axes are `max_time` and `batch`) or not (the 
        first two axes are `batch` and `max_time`).
      share_vocab: bool scalar, if True, source and target vocabularies are 
        assumed to be the same, and only one embedding matrix is created and 
        shared by both. Defaults to False.
      is_inferring: bool scalar, whether prediction model is in infer mode.
      initializer: weights initializer.
      src_vocab_size: int scalar, num of symbols in source vocabulary.
      tgt_vocab_size: int scalar, num of symbols in target vocabulary.
      num_encoder_layers: int scalar, the num of layers in the encoding RNN.
      num_encoder_res_layers: int scalar, the num of layers in the encoding RNN
         with residual connections.
      num_decoder_layers: int scalar, the num of layers in the decoding RNN.
      num_decoder_res_layers: int scalar, the num of layers in the decoding RNN
         with residual connections.
    """
    self._unit_type = unit_type
    self._num_units = num_units
    self._forget_bias = forget_bias
    self._keep_prob = keep_prob
    self._encoder_type = encoder_type
    self._time_major = time_major
    self._share_vocab = share_vocab
    self._is_inferring = is_inferring
    self._initializer = initializer
    self._src_vocab_size = src_vocab_size
    self._tgt_vocab_size = tgt_vocab_size
    self._num_encoder_layers = num_encoder_layers
    self._num_encoder_res_layers = num_encoder_res_layers
    self._num_decoder_layers = num_decoder_layers
    self._num_decoder_res_layers = num_decoder_res_layers

    self._global_scope = 'dynamic_seq2seq'
    with tf.variable_scope('decoder/output_projection'):
      self._output_layer = tf.layers.Dense(
        self._tgt_vocab_size, use_bias=False, name='output_projection') 

  @property 
  def time_major(self):
    return self._time_major

  @abstractmethod
  def predict_logits(self, *args, **kwargs):
    """Abstract method to be implemented by subclass Trainer and Evaluator.

    Creates logits tensor to be used to create loss tensor.
    """
    pass

  @abstractmethod
  def predict_indices(self, *args, **kwargs):
    """Abstract method to be implemented by subclass Inferencer.

    Creates symbol indices tensor. Symbol indices contain the decoded 
    sequence of symbol ids in the target vocabulary.
    """
    pass

  def _create_embeddings(self):
    """Returns the embeddings for encoder (source) and decoder (target). 
    The gradients will be backpropagated to the embeddings at training time. 

    Returns:
      encoder_embedding: float tensor with shape [src_vocab_size, num_units]
        containing the embeddings of symbols in source vocabulary. 
      decoder_embedding: float tensor with shape [tgt_vocab_size, num_units]
        containing the embeddings of symbols in target vocabulary.
    """
    (encoder_embedding, decoder_embedding
        ) = utils.create_encoder_and_decoder_embeddings(
            src_vocab_size=self._src_vocab_size,
            tgt_vocab_size=self._tgt_vocab_size,
            src_embed_size=self._num_units,
            tgt_embed_size=self._num_units,
            share_vocab=self._share_vocab)
    return encoder_embedding, decoder_embedding

  def _build_encoder(self,
                     src_input_ids,
                     src_seq_lens,
                     encoder_embedding,
                     scope=None):
    """Runs the source input sequences through the encoder to obtain encoder
    outputs and encoder states to be passed to decoder.

    Args:
      src_input_ids: int tensor with shape [batch, max_time_src], the indices
        of source sequence symbols in a batch.
      src_seq_lens: int tensor with shape [batch], the lengths of unpadded 
        source sequences in `src_input_ids`.
      encoder_embedding: float tensor with shape [src_vocab_size, num_units]
        containing the embeddings of symbols in source vocabulary.
      scope: string scalar, scope name.

    Returns:
      encoder_outputs: float tensor with shape
        [batch, max_time, num_units]/[max_time, batch, num_units] (time_major
        is False/True) for unidirectional RNN, or
        [batch, max_time, 2 * num_units]/[max_time, batch, 2 * num_units]
        (time_major is False/True) for bidirectional RNN.
      encoder_states: a list of `num_encoder_layers` state_tuple instances, 
        where each state_tuple contains a cell state and a hidden state tensor,
        both with shape [batch, num_units].
    """
    with tf.variable_scope(scope, 'encoder', 
        [src_input_ids, src_seq_lens, encoder_embedding]):
      if self._time_major:
        src_input_ids = tf.transpose(src_input_ids)

      inputs = tf.nn.embedding_lookup(encoder_embedding, src_input_ids)
      if self._encoder_type == 'uni':
        encoder_outputs, encoder_states = self._build_unidirectional_encoder(
            inputs, src_seq_lens)
      elif self._encoder_type == 'bi':
        encoder_outputs, encoder_states = self._build_bidirectional_encoder(
            inputs, src_seq_lens)
      else:
        raise ValueError('Unkown encoder type: {}'.format(self._encoder_type))
    return encoder_outputs, encoder_states

  def _build_unidirectional_encoder(self, inputs, src_seq_lens):
    """Builds a unidirectional RNN as encoder.

    Args:
      inputs: float tensor with shape [batch, max_time, num_units] or
        [max_time, batch, num_units], the embedding inputs to the encoder.
      src_seq_lens: int tensor with shape [batch], the lengths of unpadded 
        source sequences in a batch.

    Returns:
      outputs: float tensor with shape [batch, max_time, num_units] or
        [max_time, batch, num_units], the encoder outputs.
      states: a list of `num_encoder_layers` state_tuple instances, where
        each state_tuple contains a cell state and a hidden state tensor,
        both with shape [batch, num_units].
    """
    cell = self._build_encoder_cell(
        self._num_encoder_layers, self._num_encoder_res_layers)
    outputs, states = tf.nn.dynamic_rnn(cell,
                                        inputs,
                                        sequence_length=src_seq_lens,
                                        time_major=self._time_major,
                                        dtype=tf.float32,
                                        swap_memory=True)
    return outputs, states

  def _build_bidirectional_encoder(self, inputs, src_seq_lens):
    """Builds a bidirectional RNN as encoder.

    Args:
      inputs: float tensor with shape [batch, max_time, num_units] or
        [max_time, batch, num_units], the embedding inputs to the encoder.
      src_seq_lens: int tensor with shape [batch], the lengths of unpadded 
        source sequences in a batch.

    Returns:
      outputs: float tensor with shape [batch, max_time, 2 * num_units] or
        [max_time, batch, 2 * num_units], the encoder outputs.
      states: a list of `num_encoder_layers` state_tuple instances, where
        each state_tuple contains a cell state and a hidden state tensor,
        both with shape [batch, num_units].
    """
    num_bi_layers = self._num_encoder_layers // 2
    num_bi_res_layers = self._num_encoder_res_layers // 2
    cell_fw = self._build_encoder_cell(num_bi_layers, num_bi_res_layers)
    cell_bw = self._build_encoder_cell(num_bi_layers, num_bi_res_layers)
    # bi_outputs.shape: 
    #   ([batch, max_time, num_units], [batch, max_time, num_units]) or 
    #   ([max_time, batch, num_units], [max_time, batch, num_units])
    # bi_states.shape: (
    #   [state_tuple(c=[batch, num_units], h=[batch, num_units])]
    #     * num_bi_layers   ==> forward direction  ,
    #   [state_tuple(c=[batch, num_units], h=[batch, num_units])]
    #     * num_bi_layers   ==> backward direction
    #   )
    bi_outputs, bi_states = tf.nn.bidirectional_dynamic_rnn(
        cell_fw,
        cell_bw,
        inputs,
        sequence_length=src_seq_lens,
        time_major=self._time_major,
        dtype=tf.float32,
        swap_memory=True)
    outputs = tf.concat(bi_outputs, -1)

    if num_bi_layers == 1:
      states = bi_states
    else:
      states = []
      # interleave forward and backward state_tuples, but total num of 
      # state_tuples is still `num_encoder_layers`, the same as output of 
      # `_unidirectional_encode`.
      for l in range(num_bi_layers):
        states.append(bi_states[0][l])
        states.append(bi_states[1][l])
      states = tuple(states)
    return outputs, states

  def _build_encoder_cell(self, num_layers, num_res_layers):
    """Builds RNN Cell for encoder.

    Args:
      num_layers: int scalar, the num of layers in a multi-layer RNN.
      num_res_layers: int scalar, the num of layers in a multi-layer RNN with 
        residual connections.

    Returns:
      an RNN cell instance.
    """
    return utils.create_rnn_cell(self._unit_type, self._num_units,
        self._forget_bias, self._keep_prob, num_layers, num_res_layers)

  def _build_decoder_cell(self, num_layers, num_res_layers):
    """Builds RNN Cell for decoder.

    Args:
      num_layers: int scalar, the num of layers in a multi-layer RNN.
      num_res_layers: int scalar, the num of layers in a multi-layer RNN with 
        residual connections.

    Returns:
      an RNN cell instance.
    """
    return utils.create_rnn_cell(self._unit_type, self._num_units,
        self._forget_bias, self._keep_prob, num_layers, num_res_layers)

  def _create_logits(self,
                     tgt_input_ids,
                     tgt_seq_lens,
                     decoder_embedding,
                     cell,
                     decoder_init_state,
                     scope=None):
    """Creates logits tensor.

    Args:
      tgt_input_ids: int tensor with shape [batch, max_time_tgt], the indices
        of target input sequence symbols in a batch.
      tgt_seq_lens: int tensor with shape [batch], the lengths of unpadded 
        target sequences in `tgt_input_ids`. 
      decoder_embedding: float tensor with shape [tgt_vocab_size, num_units]
        containing the embeddings of symbols in target vocabulary.
      cell: RNN Cell instance returned by `self._build_decoder_cell`, maybe
        additionaly wrapped by `self._wrap_decoder_cell`.
      decoder_init_state: a list of state_tuples containing a cell state and
        a hidden state tensor, Or an instance of AttentionWrapperState that 
        wraps `encoder_states` with attention data.
      scope: string scalar, scope name.

    Returns:
      logits: float tensor with shape [max_tgt_time, batch, tgt_vocab_size]/
        [batch, max_tgt_time, tgt_vocab_size], the prediction logits.
    """
    if self._time_major:
      tgt_input_ids = tf.transpose(tgt_input_ids)
    inputs = tf.nn.embedding_lookup(decoder_embedding, tgt_input_ids)

    helper = tf.contrib.seq2seq.TrainingHelper(
        inputs, tgt_seq_lens, time_major=self._time_major)

    decoder = tf.contrib.seq2seq.BasicDecoder(
        cell, helper, decoder_init_state, output_layer=self._output_layer)

    outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
        decoder,
        output_time_major=self._time_major,
        swap_memory=True,
        scope=scope)

    logits = outputs.rnn_output

    return logits

  def _create_indices(self,
                      tgt_sos_id,
                      tgt_eos_id,
                      batch_size,
                      decoder_embedding,
                      cell,
                      decoder_init_state,
                      beam_width,
                      length_penalty_weight,
                      sampling_temperature,
                      maximum_iterations,
                      random_seed,
                      scope=None):
    """Creates symbol indices tensor.

    This function generates decoded sequence of symbol indices, given the 
    initial decoder state in which the source input sequence is encoded. It
    uses either a basic decoder, which generates the next symbol index one
    at a time (with greedy or sampling strategy), or a beam search decoder, 
    which generates the whole sequence of symbol indices all at once.

    Args:
      tgt_sos_id: an int scalar tensor, the index of sos in target vocabulary.
      tgt_eos_id: an int scalar tensor, the index of eos in target vocabulary.
      batch_size: int scalar tensor, batch size.
      decoder_embedding: float tensor with shape [tgt_vocab_size, num_units]
        containing the embeddings of symbols in target vocabulary.
      cell: RNN Cell instance returned by `self._build_decoder_cell`, maybe
        additionaly wrapped by `self._wrap_decoder_cell`. 
      decoder_init_state: a list of state_tuples containing a cell state and
        a hidden state tensor, Or an instance of AttentionWrapperState that 
        wraps `encoder_states` with attention data.
      beam_width: int scalar, width for beam seach.
      length_penalty_weight: float scalar, length penalty weight for beam 
        search.
      sampling_temperature: float scalar > 0.0, sampling temperature for 
        sampling decoder. 
      maximum_iterations: int scalar, max num of iterations for dynamic
        decoding.
      random_seed: int scalar, random seed for sampling decoder. 
      scope: string scalar, scope name.

    Returns:
      indices: int tensor with shape [batch, max_time]/[max_time, batch] for
        greedy and sampling decoder, Or
        [batch, max_time, beam_width]/[max_time, batch, beam_width] for 
        beam search decoder, the sampled ids of decoded sequence of symbols 
        in target vocabulary.
    """
    start_tokens = tf.fill([batch_size], tgt_sos_id)
    end_token = tgt_eos_id

    if beam_width > 0:
      decoder = tf.contrib.seq2seq.BeamSearchDecoder(
          cell=cell,
          embedding=decoder_embedding,
          start_tokens=start_tokens,
          end_token=end_token,
          initial_state=decoder_init_state,
          beam_width=beam_width,
          output_layer=self._output_layer,
          length_penalty_weight=length_penalty_weight)
    else:
      if sampling_temperature > 0.0:
        helper = tf.contrib.seq2seq.SampleEmbeddingHelper(
            decoder_embedding,
            start_tokens,
            end_token,
            softmax_temperature=sampling_temperature,
            seed=random_seed)
      else:
        helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
            decoder_embedding, start_tokens, end_token)

      decoder = tf.contrib.seq2seq.BasicDecoder(
          cell, helper, decoder_init_state, output_layer=self._output_layer)
  
    outputs, states, _ = tf.contrib.seq2seq.dynamic_decode(
        decoder,
        maximum_iterations=maximum_iterations,
        output_time_major=self._time_major,
        swap_memory=True,
        scope=scope)

    if beam_width > 0:
      indices = outputs.predicted_ids
    else:
      indices = outputs.sample_id
    return indices, states

