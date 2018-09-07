import tensorflow as tf

from seq2seq_model import Seq2SeqPredictionModel


class VanillaSeq2SeqPredictionModel(Seq2SeqPredictionModel):
  """The Vanilla Seq2Seq model without attention mechanism. Simply uses
  encoder states to initialize decoder states.
  """
  def predict_logits(self,
                     src_input_ids,
                     src_seq_lens,
                     tgt_input_ids,
                     tgt_seq_lens,
                     scope=None):
    """Creates logits tensor to be used to create loss tensor.

    Args:
      src_input_ids: int tensor with shape [batch, max_time_src], the indices
        of source sequence symbols in a batch.
      src_seq_lens: int tensor with shape [batch], the lengths of unpadded 
        source sequences in `src_input_ids`.
      tgt_input_ids: int tensor with shape [batch, max_time_tgt], the indices
        of target input sequence symbols in a batch.
      tgt_seq_lens: int tensor with shape [batch], the lengths of unpadded 
        target sequences in `tgt_input_ids`.
      scope: string scalar, scope name.

    Returns:
      logits: float tensor with shape [max_tgt_time, batch, tgt_vocab_size]/
        [batch, max_tgt_time, tgt_vocab_size], the prediction logits.
      batch_size: int scalar tensor, num of sequence in a batch. Used to compute
        training perplexity.
    """
    if self._is_inferring:
      raise ValueError('Model must NOT be in inferring mode when calling',
          '`predict_logits`.')
    encoder_embedding, decoder_embedding = self._create_embeddings()
    with tf.variable_scope(scope, self._global_scope, 
        [src_input_ids, src_seq_lens, tgt_input_ids, tgt_seq_lens],
        initializer=self._initializer):
      encoder_outputs, encoder_states = self._build_encoder(
          src_input_ids, src_seq_lens, encoder_embedding)

      with tf.variable_scope('decoder') as scope:
        batch_size = tf.size(src_seq_lens)

        cell = self._build_decoder_cell(
            self._num_decoder_layers, self._num_decoder_res_layers)
        decoder_init_state = encoder_states
        logits = self._create_logits(tgt_input_ids, tgt_seq_lens, 
            decoder_embedding, cell, decoder_init_state, scope)
        return logits, batch_size

  def predict_indices(self,
                      src_input_ids,
                      src_seq_lens,
                      tgt_sos_id,
                      tgt_eos_id,
                      beam_width=10,
                      length_penalty_weight=0.0,
                      sampling_temperature=1.0,
                      maximum_iterations=None,
                      random_seed=0,
                      scope=None):
    """Creates symbol indices tensor. Symbol indices contain the decoded 
    sequence of symbol ids in the target vocabulary.

    Args:
      src_input_ids: int tensor with shape [batch, max_time_src], the indices 
        of source sequence symbols in a batch.
      src_seq_lens: int tensor with shape [batch], the lengths of unpadded 
        source sequences in `src_input_ids`.
      tgt_sos_id: an int scalar tensor, the index of sos in target vocabulary.
      tgt_eos_id: an int scalar tensor, the index of eos in target vocabulary.
      beam_width: int scalar, width for beam seach.
      length_penalty_weight: float scalar, length penalty weight for beam 
        search. Disabled with 0.0
      sampling_temperature: float scalar > 0.0, value to divide the logits by
        before computing the softmax. Larger values (above 1.0) result in more
        random samples, while smaller values push the sampling distribution 
        towards the argmax. 
      maximum_iterations: int scalar or None, max num of iterations for dynamic
        decoding.
      random_seed: int scalar, random seed for sampling decoder. 
      scope: string scalar, scope name.

    Returns:
      indices: int tensor with shape [batch, max_time_tgt]/[max_time_tgt, batch] 
        for greedy and sampling decoder, Or
        [batch, max_time_tgt, beam_width]/[max_time_tgt, batch, beam_width] for 
        beam search decoder, the sampled ids of decoded sequence of symbols 
        in target vocabulary.
      alignment: tf.no_op, DUMMY operation.
    """
    if not self._is_inferring:
      raise ValueError('Model must be in inferring mode when calling',
          '`predict_indices`.')
    encoder_embedding, decoder_embedding = self._create_embeddings()
    with tf.variable_scope(scope, self._global_scope, 
        [src_input_ids, src_seq_lens, tgt_sos_id, tgt_eos_id],
        initializer=self._initializer):
      encoder_outputs, encoder_states = self._build_encoder(
          src_input_ids, src_seq_lens, encoder_embedding)

      with tf.variable_scope('decoder') as scope:
        batch_size = tf.size(src_seq_lens)

        cell = self._build_decoder_cell(
            self._num_decoder_layers, self._num_decoder_res_layers)
        decoder_init_state = encoder_states
        if beam_width > 0:
          decoder_init_state = tf.contrib.seq2seq.tile_batch(
              encoder_states, beam_width)
        maximum_iterations = (maximum_iterations or 
            2 * tf.reduce_max(src_seq_lens))
        indices, alignment = self._create_indices(
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
            scope)
        return indices, alignment


class AttentionSeq2SeqPredictionModel(Seq2SeqPredictionModel):
  """The Seq2Seq model with attention mechanism."""
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
               num_decoder_res_layers,
               attention_type='scaled_luong',
               output_attention=True):
    """Constructor.

    Args:
      unit_type: string scalar, the type of RNN cell ('lstm', 'gru', etc.).
      num_units: int scalar, the num of units in an RNN Cell.
      forget_bias: float scalar, forget bias in LSTM Cell. Defaults to 1.0.
      keep_prob: float scalar, dropout rate equals `1 - keep_prob`.
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
      attention_type: string scalar, valid values are 'luong', 'scaled_luong',
        'bahdanau', and 'normed_bahdanau'.
      output_attention: bool scalar, If True, the output at each time step is 
        the attention value. This is the behavior of Luong-style attention 
        mechanisms. If False, the output at each time step is the output of 
        cell. This is the behavior of Bhadanau-style attention mechanisms. In
        both cases, the attention tensor is propagated to the next time step 
        via the state and is used there. This flag only controls whether the 
        attention mechanism is propagated up to the next cell in an RNN stack
        or to the top RNN output.
    """
    super(AttentionSeq2SeqPredictionModel, self).__init__(
        unit_type=unit_type,
        num_units=num_units,
        forget_bias=forget_bias,
        keep_prob=keep_prob,
        encoder_type=encoder_type,
        time_major=time_major,
        share_vocab=share_vocab,
        is_inferring=is_inferring,
        initializer=initializer,
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        num_encoder_layers=num_encoder_layers,
        num_encoder_res_layers=num_encoder_res_layers,
        num_decoder_layers=num_decoder_layers,
        num_decoder_res_layers=num_decoder_res_layers)
    self._attention_type = attention_type
    self._output_attention = output_attention

  def predict_logits(self,
                     src_input_ids,
                     src_seq_lens,
                     tgt_input_ids,
                     tgt_seq_lens,
                     scope=None):
    """Creates logits tensor to be used to create loss tensor.

    Args:
      src_input_ids: int tensor with shape [batch, max_time_src], the indices
        of source sequence symbols in a batch.
      src_seq_lens: int tensor with shape [batch], the lengths of unpadded 
        source sequences in `src_input_ids`.
      tgt_input_ids: int tensor with shape [batch, max_time_tgt], the indices
        of target input sequence symbols in a batch.
      tgt_seq_lens: int tensor with shape [batch], the lengths of unpadded 
        target sequences in `tgt_input_ids`.
      scope: string scalar, scope name.

    Returns:
      logits: float tensor with shape [max_tgt_time, batch, tgt_vocab_size]/
        [batch, max_tgt_time, tgt_vocab_size], the prediction logits.
      batch_size: int scalar tensor, num of sequence in a batch. Used to compute
        training perplexity.
    """
    if self._is_inferring:
      raise ValueError('Model must NOT be in inferring mode when calling',
          '`predict_logits`.')
    encoder_embedding, decoder_embedding = self._create_embeddings()
    with tf.variable_scope(scope, self._global_scope,
        [src_input_ids, src_seq_lens, tgt_input_ids, tgt_seq_lens],
        initializer=self._initializer):
      encoder_outputs, encoder_states = self._build_encoder(
          src_input_ids, src_seq_lens, encoder_embedding)

      with tf.variable_scope('decoder') as scope:
        batch_size = tf.size(src_seq_lens)

        cell = self._build_decoder_cell(
            self._num_decoder_layers, self._num_decoder_res_layers)
        cell = self._wrap_decoder_cell(
            cell, encoder_outputs, src_seq_lens)
        decoder_init_state = self._get_decoder_init_states(
            encoder_states, cell, batch_size)
        logits = self._create_logits(tgt_input_ids, tgt_seq_lens, 
            decoder_embedding, cell, decoder_init_state, scope)
        return logits, batch_size

  def predict_indices(self,
                      src_input_ids,
                      src_seq_lens,
                      tgt_sos_id,
                      tgt_eos_id,
                      beam_width=10,
                      length_penalty_weight=0.0,
                      sampling_temperature=1.0,
                      maximum_iterations=None,
                      random_seed=0,
                      scope=None):
    """Creates symbol indices tensor. Symbol indices contain the decoded 
    sequence of symbol ids in the target vocabulary.

    Args:
      src_input_ids: int tensor with shape [batch, max_time_src], the indices 
        of source sequence symbols in a batch.
      src_seq_lens: int tensor with shape [batch], the lengths of unpadded 
        source sequences in `src_input_ids`.
      tgt_sos_id: an int scalar tensor, the index of sos in target vocabulary.
      tgt_eos_id: an int scalar tensor, the index of eos in target vocabulary.
      beam_width: int scalar, width for beam seach.
      length_penalty_weight: float scalar, length penalty weight for beam 
        search. Disabled with 0.0
      sampling_temperature: float scalar > 0.0, value to divide the logits by
        before computing the softmax. Larger values (above 1.0) result in more
        random samples, while smaller values push the sampling distribution 
        towards the argmax. 
      maximum_iterations: int scalar or None, max num of iterations for dynamic
        decoding.
      random_seed: int scalar, random seed for sampling decoder. 
      scope: string scalar, scope name.

    Returns:
      indices: int tensor with shape [batch, max_time_tgt]/[max_time_tgt, batch] 
        for greedy and sampling decoder, Or
        [batch, max_time_tgt, beam_width]/[max_time_tgt, batch, beam_width] for 
        beam search decoder, the sampled ids of decoded sequence of symbols 
        in target vocabulary.
      alignment: float tensor with shape [max_time_tgt, K, max_time_src], where
        max_time_tgt = the maximum length of decoded sequences over a batch,
        K = batch_size (not in beam-search mode) or batch_size * beam_width (
        with batch_size being the first axis, in beam-search mode), holding the 
        alignment scores of each target symbol w.r.t each input source symbol.
    """
    if not self._is_inferring:
      raise ValueError('Model must be in inferring mode when calling',
          '`predict_indices`.')
    encoder_embedding, decoder_embedding = self._create_embeddings()
    with tf.variable_scope(scope, self._global_scope,
        [src_input_ids, src_seq_lens, tgt_sos_id, tgt_eos_id],
        initializer=self._initializer):
      encoder_outputs, encoder_states = self._build_encoder(
          src_input_ids, src_seq_lens, encoder_embedding)

      with tf.variable_scope('decoder') as scope:
        batch_size = tf.size(src_seq_lens)

        cell = self._build_decoder_cell(
            self._num_decoder_layers, self._num_decoder_res_layers)
        cell = self._wrap_decoder_cell(
            cell, encoder_outputs, src_seq_lens, beam_width)
        decoder_init_state = self._get_decoder_init_states(
            encoder_states, cell, batch_size, beam_width)
        maximum_iterations = (maximum_iterations or
            2 * tf.reduce_max(src_seq_lens))
        indices, alignment = self._create_indices(
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
            scope)
        return indices, alignment

  def _wrap_decoder_cell(self,
                         cell,
                         encoder_outputs,
                         src_seq_lens,
                         beam_width=0):
    """Wraps decoder cell with attention mechanism.

    The argument `beam_width` has a default value and is ignored when called by
    `predict_logits` (i.e. `self._inferring` is False).

    Args:
      cell: an RNN Cell, decoder cell returned by `self._build_decoder_cell`.
      encoder_outputs: float tensor with shape
        [batch, max_time_src, num_units]/[max_time_src, batch, num_units] 
        (time_major is False/True) for unidirectional RNN, or
        [batch, max_time_src, 2*num_units]/[max_time_src, batch, 2*num_units]
        (time_major is False/True) for bidirectional RNN.
      src_seq_lens: int tensor with shape [batch], the lengths of unpadded 
        source sequences in `src_input_ids`.
      beam_width: int scalar, width for beam seach.

    Returns:
      wrapped_cell: an RNN Cell instance wrapped with attention mechanism.
    """
    memory = encoder_outputs

    # make sure `memory` is in batch major to be batch-tiled
    if self._time_major:
      memory = tf.transpose(memory, [1, 0, 2])
    if self._is_inferring and beam_width > 0:
      memory = tf.contrib.seq2seq.tile_batch(memory, beam_width)
      src_seq_lens = tf.contrib.seq2seq.tile_batch(src_seq_lens, beam_width)

    attention_mechanism = self._create_attention_mechanism(
        memory, src_seq_lens)
    wrapped_cell = tf.contrib.seq2seq.AttentionWrapper(
        cell,
        attention_mechanism,
        attention_layer_size=self._num_units,
        alignment_history=True,
        output_attention=self._output_attention,
        name='attention')
    return wrapped_cell

  def _get_decoder_init_states(self,
                               encoder_states,
                               cell,
                               batch_size,
                               beam_width=0):
    """Creates decoder initial states from encoder states.

    The argument `beam_width` has default value and is ignored when called by
    `predict_logits` (i.e. `self._inferring` is False).

    Args:
      encoder_states: a list of `num_encoder_layers` state_tuple instances, 
        where each state_tuple contains a cell state and a hidden state tensor,
        both with shape [batch, num_units].
      cell: an RNN Cell, decoder cell wrapped by `AttentionWrapper`.
      batch_size: int scalar tensor, batch size.
      beam_width: int scalar, width for beam seach.

    Returns:
      attention_wrapper_state: an instance of AttentionWrapperState.
    """
    if self._is_inferring and beam_width > 0:
      encoder_states = tf.contrib.seq2seq.tile_batch(
          encoder_states, beam_width)
      batch_size *= beam_width
    attention_wrapper_state = cell.zero_state(batch_size, tf.float32
        ).clone(cell_state=encoder_states)
    return attention_wrapper_state

  def _create_attention_mechanism(self,
                                  memory,
                                  src_seq_lens):
    """Creates attention mechanism instance to be passed to attention
    cell wrapper `AttentionWrapper`.

    Args:
      memory: float tensor with shape [batch, max_time_src, num_units], the
        `encoder_outputs` in batch major format.
      src_seq_lens: int tensor with shape [batch], the lengths of unpadded 
        source sequences in `src_input_ids`.

    Returns:
      an AttentionMechanism instance.
    """
    if self._attention_type == 'luong':
      attention_mechanism = tf.contrib.seq2seq.LuongAttention(
          self._num_units, memory, memory_sequence_length=src_seq_lens)
    elif self._attention_type == 'scaled_luong':
      attention_mechanism = tf.contrib.seq2seq.LuongAttention(
          self._num_units,
          memory,
          memory_sequence_length=src_seq_lens,
          scale=True)
    elif self._attention_type == 'bahdanau':
      attention_mechanism = tf.conrib.seq2seq.BahdanauAttention(
          self._num_units, memory, memory_sequence_length=src_seq_lens)
    elif self._attention_type == 'normed_bahdanau':
      attention_mechanism = tf.conrib.seq2seq.BahdanauAttention(
          self._num_units,
          memory,
          memory_sequence_length=src_seq_lens,
          normalize=True)
    else:
      raise ValueError('Unknown attention type {}'.format(self._attention_type))
    return attention_mechanism
 
