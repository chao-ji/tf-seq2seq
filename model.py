from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import data
import model_helper


class Seq2SeqModel(object):

  def __init__(self, hparams, dataset, mode, dtype=tf.float32, scope=None):

    assert isinstance(dataset, data.Seq2SeqDataset)
    self._mode = mode
    self._dtype = dtype
    self._time_major = hparams.time_major

    self._num_encoder_layers = hparams.num_encoder_layers
    self._num_decoder_layers = hparams.num_decoder_layers
    self._num_encoder_res_layers = hparams.num_encoder_residual_layers
    self._num_decoder_res_layers = hparams.num_decoder_residual_layers
    self._batch_size = tf.size(dataset.src_seq_lens)

    initializer = model_helper.get_initializer(
        hparams.init_op, hparams.random_seed, hparams.init_weight)
    tf.get_variable_scope().set_initializer(initializer)

    self._embed_encoder, self._embed_decoder = self._init_embeddings(
                                                  hparams, scope)

    with tf.variable_scope(scope or "build_netword"):
      with tf.variable_scope("decoder/output_projection"):
        self._output_layer = tf.layers.Dense(
            hparams.tgt_vocab_size, use_bias=False, name="output_projection")

    self.logits, self.loss, self.states, self.sample_id = self._build_graph(
        hparams, dataset, scope)

  @property
  def mode(self):
    return self._mode

  @property
  def dtype(self):
    return self._dtype

  @property
  def time_major(self):
    return self._time_major

  @property
  def batch_size(self):
    return self._batch_size

  @property
  def num_encoder_layers(self):
    return self._num_encoder_layers

  @property
  def num_decoder_layers(self):
    return self._num_decoder_layers

  @property
  def num_encoder_res_layers(self):
    return self._num_encoder_res_layers

  @property
  def num_decoder_res_layers(self):
    return self._num_decoder_res_layers

  def _init_embeddings(self, hparams, scope):
    return model_helper.create_embed_encoder_and_embed_decoder(
        share_vocab=hparams.share_vocab,
        src_vocab_size=hparams.src_vocab_size,
        tgt_vocab_size=hparams.tgt_vocab_size,
        src_embed_size=hparams.num_units,
        tgt_embed_size=hparams.num_units,
        scope=scope)

  def _build_graph(self, hparams, dataset, scope=None):
    with tf.variable_scope(scope or "dynamic_seq2seq", dtype=self.dtype):
      # "uni": outputs_encoder = [N, T, D] or [T, N, D]
      # "bi": outputs_encoder = [N, T, 2D] or [T, N, 2D]
      # states_encoder = [state_tuple(c=[N, D], h=[N, D])]*NUM_LAYERS
      outputs_encoder, states_encoder = self._build_encoder(hparams, dataset)

      # TRAIN or EVAL mode
      # logits = [N, T, V] or [T, N, V] 
      # sample_id = [N, T] or [T, N]
      # EVAL mode
      # if hparams.beam_width > 0
      # logits = tf.no_op() 
      # sample_id = [N, T, B] or [T, N, B] 
      # else
      # logits = [N, T, V] or [T, N, V] 
      # sample_id = [N, T] or [T, N]
      logits, sample_id, states = self._build_decoder(
          hparams, dataset, outputs_encoder, states_encoder)

      if self.mode != tf.contrib.learn.ModeKeys.INFER:
        loss = self._compute_loss(dataset, logits)
      else:
        loss = tf.no_op()

    return logits, loss, states, sample_id

  def _build_encoder(self, hparams, dataset):
    num_layers = self.num_encoder_layers
    num_res_layers = self.num_encoder_res_layers
    # embed = [V, D]
    embed = self._embed_encoder
    # src_input_ids = [N, T]
    src_input_ids = dataset.src_input_ids
    # src_input_ids = [T, N]
    if self.time_major:
      src_input_ids = tf.transpose(src_input_ids)
 
    with tf.variable_scope("encoder") as scope:
      # inputs = [N, T, D] or [T, N, D]
      inputs = tf.nn.embedding_lookup(embed, src_input_ids)

      if hparams.encoder_type == "uni":
        cell = self._build_encoder_cell(
            hparams, num_layers, num_res_layers)
        # outputs = [N, T, D] or [T, N, D]
        # states = [state_tuple(c=[N, D], h=[N, D])]*NUM_LAYERS
        outputs, states = tf.nn.dynamic_rnn(
            cell,
            inputs,
            dtype=self.dtype,
            # dataset.src_seq_lens = [N]
            sequence_length=dataset.src_seq_lens,
            time_major=self.time_major,
            swap_memory=True)
      elif hparams.encoder_type == "bi":
        num_bi_layers = num_layers // 2
        num_bi_res_layers = num_res_layers // 2
        # bi_outputs = ([N, T, D], [N, T, D]) or ([T, N, D], [T, N, D])
        # bi_states = ([state_tuple(c=[N, D], h=[N, D])]*NUM_BI_LAYERS,
        #              [state_tuple(c=[N, D], h=[N, D])]*NUM_BI_LAYERS)
        bi_outputs, bi_states = self._build_bidirectional_rnn(
            hparams,
            inputs,
            # dataset.src_seq_lens = [N]
            dataset.src_seq_lens,
            self.dtype,
            num_bi_layers,
            num_bi_res_layers)
        # outputs = [N, T, 2D] or [T, N, 2D]
        outputs = tf.concat(bi_outputs, -1)

        if num_bi_layers == 1:
          states = bi_states
        else:
          states = []
          for l in range(num_bi_layers):
            states.append(bi_states[0][l])
            states.append(bi_states[1][l])
          states = tuple(states)
        # states = [state_tuple(c=[N, D], h=[N, D])]*NUM_LAYERS
      else:
        raise ValueError("Unknown encoder_type: %s" % hparams.encoder_type)
    return outputs, states

  def _build_encoder_cell(self, hparams, num_layers, num_res_layers):
    return model_helper.create_rnn_cell(
        hparams.unit_type,
        hparams.num_units,
        num_layers,
        num_res_layers,
        hparams.forget_bias,
        hparams.dropout,
        self.mode)

  def _build_decoder(self, hparams, dataset, outputs_encoder, states_encoder):
    num_layers = self.num_decoder_layers
    num_res_layers = self.num_decoder_res_layers
    # embed = [V, D]
    embed = self._embed_decoder
 
    with tf.variable_scope("decoder") as scope:
      cell = self._build_decoder_cell(
          hparams,
          num_layers,
          num_res_layers,
          # "uni": outputs_encoder = [N, T, D] or [T, N, D]
          # "bi": outputs_encoder = [N, T, 2D] or [T, N, 2D]
          outputs_encoder,
          # dataset.src_seq_lens = [N]
          dataset.src_seq_lens)

      # NOTE: `decoder_init_state` is batch-tiled in INFER mode with beam search
      # [state_tuple(c=[N(*B), D], h=[N(*B), D])]*NUM_LAYERS
      decoder_init_state = self._get_decoder_init_states(
          hparams,
          # states_encoder = [state_tuple(c=[N, D], h=[N, D])]*NUM_LAYERS
          states_encoder,
          cell)

      ####################
      # TRAIN or EVAL mode
      ####################
      if self.mode != tf.contrib.learn.ModeKeys.INFER:
        # tgt_input_ids: [N, T]
        tgt_input_ids = dataset.tgt_input_ids
        # tgt_input_ids: [T, N]
        if self.time_major:
          tgt_input_ids = tf.transpose(tgt_input_ids)
        # inputs = [N, T, D] or [T, N, D]
        inputs = tf.nn.embedding_lookup(embed, tgt_input_ids)

        # Helper
        helper = tf.contrib.seq2seq.TrainingHelper(
            inputs, dataset.tgt_seq_lens, time_major=self.time_major)

        # Decoder
        decoder = tf.contrib.seq2seq.BasicDecoder(
            cell,
            helper,
            decoder_init_state)

        # outputs: tf.contrib.seq2seq.BasicDecoderOutput
        # outputs.rnn_output = [N, T, D] or [T, N, D]
        # outputs.sample_id = [N, T] or [T, N] --- argmax(rnn_output, axis=2)
        #
        # if hparams.attention == ""
        # states = [state_tuple(c=[N, D], h=[N, D])]*NUM_LAYERS
        # if hparams.attention != ""
        # states: tf.contrib.seq2seq.AttentionWrapperState
        #
        # states.cell_state = [state_tuple(c=[N, D], h=[N, D])]*NUM_LAYERS
        # states.attention = [N, D]
        # states.time = []
        # states.alignments = [N, Tsrc]
        # states.alignment_history = [Ttgt, N, Tsrc]
        outputs, states, decode_seq_lens = tf.contrib.seq2seq.dynamic_decode(
            decoder,
            output_time_major=self.time_major,
            swap_memory=True,
            scope=scope)
        # sample_id = [N, T] or [T, N]
        sample_id = outputs.sample_id
        # logits = [N, T, V] or [T, N, V]
        logits = self._output_layer(outputs.rnn_output)

      ############
      # INFER mode
      ############
      else:
        beam_width = hparams.beam_width
        length_penalty_weight = hparams.length_penalty_weight
        start_tokens = tf.fill([self.batch_size], dataset.tgt_sos_id)
        end_token = dataset.tgt_eos_id

        if beam_width > 0:
          # BEAM SEARCH MODE
          # NOTE: In INFER mode with beam search, 
          # `memory`, 
          # `src_seq_lens`
          # `states_encoder` must be batch-tiled
          # Link: https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/BeamSearchDecoder
          decoder = tf.contrib.seq2seq.BeamSearchDecoder(
              cell=cell,
              embedding=embed,
              start_tokens=start_tokens,
              end_token=end_token,
              initial_state=decoder_init_state,
              beam_width=beam_width,
              output_layer=self._output_layer,
              length_penalty_weight=length_penalty_weight)
        else:
          sampling_temperature = hparams.sampling_temperature
          if sampling_temperature > 0.0:
            # SAMPLING MODE
            helper = tf.contrib.seq2seq.SampleEmbeddingHelper(
                embed, start_tokens, end_token,
                softmax_temperature=sampling_temperature,
                seed=hparams.random_seed)
          else:
            #  GREEDY MODE 
            helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                embed, start_tokens, end_token)

          decoder = tf.contrib.seq2seq.BasicDecoder(
              cell,
              helper,
              decoder_init_state,
              output_layer=self._output_layer)

        maximum_iterations = hparams.tgt_max_len_infer if \
            hparams.tgt_max_len_infer else 2 * dataset.get_max_time_src()

        # if beam_width > 0
        # outputs: seq2seq.FinalBeamSearchDecoderOutput
        # outputs.predicted_ids = [N, T, B] or [T, N, B]
        # outputs.beam_search_decoder_output
        # else
        # outputs: seq2seq.BasicDecoderOutput
        # outputs.rnn_output = [N, T, V] or [T, N, V] (output_layer applied)
        # outputs.sample_id = [N, T] or [T, N] --- argmax(rnn_output, axis=2)
        outputs, states, decode_seq_lens = tf.contrib.seq2seq.dynamic_decode(
            decoder,
            maximum_iterations=maximum_iterations,
            output_time_major=self.time_major,
            swap_memory=True,
            scope=scope)

        if beam_width > 0:
          logits = tf.no_op()
          # sample_id = [N, T, B] or [T, N, B]
          sample_id = outputs.predicted_ids
        else:
          # logits = [N, T, V] or [T, N, V]
          logits = outputs.rnn_output
          # sample_id = [N, T] or [T, N]
          sample_id = outputs.sample_id

    return logits, sample_id, states

  def _build_decoder_cell(self,
                          hparams,
                          num_layers,
                          num_res_layers,
                          outputs_encoder,
                          src_seq_lens):
    # `outputs_encoder`
    # `src_seq_lens`
    # not used in non-attentional model architecture
    del outputs_encoder, src_seq_lens
    return model_helper.create_rnn_cell(
        hparams.unit_type,
        hparams.num_units,
        num_layers,
        num_res_layers,
        hparams.forget_bias,
        hparams.dropout,
        self.mode)

  def _use_tile_batch(self, hparams):
    return (self.mode == tf.contrib.learn.ModeKeys.INFER and 
        hparams.beam_width > 0)

  def _get_decoder_init_states(self, hparams, states_encoder, cell):
    # `cell` not used in non-attentional model architecture
    del cell
    if self._use_tile_batch(hparams):
      return tf.contrib.seq2seq.tile_batch(states_encoder, hparams.beam_width)
    else:
      return states_encoder 

  def _build_bidirectional_rnn(self,
                               hparams,
                               inputs,
                               sequence_length,
                               dtype,
                               num_bi_layers,
                               num_bi_res_layers):
    cell_fw = self._build_encoder_cell(hparams,
                                       num_bi_layers,
                                       num_bi_res_layers)
    cell_bw = self._build_encoder_cell(hparams,
                                       num_bi_layers,
                                       num_bi_res_layers)

    bi_outputs, bi_states = tf.nn.bidirectional_dynamic_rnn(
        cell_fw,
        cell_bw,
        inputs,
        sequence_length=sequence_length,
        dtype=dtype,
        time_major=self.time_major,
        swap_memory=True)

    return bi_outputs, bi_states

  def _compute_loss(self, dataset, logits):
    # tgt_output_ids = [N, T]
    tgt_output_ids = dataset.tgt_output_ids
    # tgt_output_ids = [T, N]
    if self.time_major:
      tgt_output_ids = tf.transpose(tgt_output_ids)
    # xentropy = [N, T] or [T, N]     
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=tgt_output_ids, logits=logits)

    max_time = dataset.get_max_time_tgt()
    # xentropy_weights = [N, T]
    xentropy_weights = tf.sequence_mask(
        dataset.tgt_seq_lens, max_time, dtype=self.dtype)
    # xentropy_weights = [T, N]
    if self.time_major:
      xentropy_weights = tf.transpose(xentropy_weights)

    # per sentence pair loss
    loss = tf.reduce_sum(
        xentropy * xentropy_weights) / tf.to_float(self.batch_size)
    
    return loss
