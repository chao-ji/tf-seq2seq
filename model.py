from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
#from . import data
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

    self._embed_encoder, self._embed_decoder = self._init_embed(hparams, scope)

    with tf.variable_scope(scope or "build_netword"):
      with tf.variable_scope("decoder/output_projection"):
        self._output_layer = tf.layers.Dense(
            hparams.tgt_vocab_size, use_bias=False, name="output_projection")

    self.logits, self.loss, self.states, self.sample_id = self._build_graph(hparams, dataset)

  @property
  def mode(self):
    return self._mode

  @property
  def dtype(self):
    return self._dtype

  @property
  def batch_size(self):
    return self._batch_size

  @property
  def time_major(self):
    return self._time_major

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

  def _init_embed(self, hparams, scope):
    return model_helper.create_encoder_decoder_embeddings(
        share_vocab=hparams.share_vocab,
        src_vocab_size=hparams.src_vocab_size,
        tgt_vocab_size=hparams.tgt_vocab_size,
        src_embed_size=hparams.num_units,
        tgt_embed_size=hparams.num_units,
        scope=scope)

  def _build_graph(self, hparams, dataset, scope=None):
    with tf.variable_scope(scope or "dynamic_seq2seq", dtype=self.dtype):
      outputs_encoder, states_encoder = self._build_encoder(hparams, dataset)

      logits, sample_id, states = self._build_decoder(hparams, dataset, outputs_encoder, states_encoder)
      
      if self.mode != tf.contrib.learn.ModeKeys.INFER:
        loss = self._compute_loss(dataset, logits)
      else:
        loss = None

    return logits, loss, states, sample_id

  def _build_encoder(self, hparams, dataset):
    num_layers = self.num_encoder_layers
    num_res_layers = self.num_encoder_res_layers
    embed = self._embed_encoder
    src_inputs_ids = dataset.src_input_ids

    if self.time_major:
      src_inputs_ids = tf.transpose(src_inputs_ids)
 
    with tf.variable_scope("encoder") as scope:
      inputs = tf.nn.embedding_lookup(embed, src_inputs_ids)

      if hparams.encoder_type == "uni":
        cell = self._build_encoder_cell(
            hparams, num_layers, num_res_layers) 

        outputs, states = tf.nn.dynamic_rnn(
            cell,
            inputs,
            dtype=self.dtype,
            sequence_length=dataset.src_seq_lens,
            time_major=self.time_major)
      elif hparams.encoder_type == "bi":
        num_bi_layers = num_layers // 2
        num_bi_res_layers = num_res_layers // 2
        bi_outputs, bi_states = self._build_bidirectional_rnn(
            hparams,
            inputs,
            dataset.src_seq_lens,
            self.dtype,
            num_bi_layers,
            num_bi_res_layers)

        outputs = tf.concat(bi_outputs, -1)

        if num_bi_layers:
          states = bi_states
        else:
          states = []
          for l in range(num_bi_layers):
            states.append(bi_states[0][l])
            states.append(bi_states[1][l])
          states = tuple(states)
      else:
        pass
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
    embed = self._embed_decoder
    src_seq_lens = dataset.src_seq_lens   
 
    with tf.variable_scope("decoder") as scope:
      cell = self._build_decoder_cell(
          hparams, num_layers, num_res_layers, outputs_encoder, src_seq_lens)
      decoder_init_state = self._get_decoder_init_states(
          hparams, states_encoder, cell)

      if self.mode != tf.contrib.learn.ModeKeys.INFER:
        tgt_input_ids = dataset.tgt_input_ids
        if self.time_major:
          tgt_input_ids = tf.transpose(tgt_input_ids)
        inputs = tf.nn.embedding_lookup(embed, tgt_input_ids)

        # Helper
        helper = tf.contrib.seq2seq.TrainingHelper(
            inputs, dataset.tgt_seq_lens, time_major=self.time_major)

        # Decoder
        basic_decoder = tf.contrib.seq2seq.BasicDecoder(
            cell,
            helper,
            decoder_init_state)

        outputs, states, decode_seq_lens = tf.contrib.seq2seq.dynamic_decode(
            basic_decoder,
            output_time_major=self.time_major,
            scope=scope)

        sample_id = outputs.sample_id
        logits = self._output_layer(outputs.rnn_output)
      else:
        tgt_sos_id = dataset.tgt_sos_id
        tgt_eos_id = dataset.tgt_eos_id

        logits = tf.no_op() 
        sample_id = tf.no_op()
        states = tf.no_op()

    return logits, sample_id, states

  def _build_decoder_cell(self,
                          hparams,
                          num_layers,
                          num_res_layers,
                          outputs_encoder,
                          src_seq_lens):
    # `outputs_encoder` and `src_seq_lens` not used in standard architecture
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
    # `cell` not used
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
        time_major=self.time_major) 

    return bi_outputs, bi_states

  def _compute_loss(self, dataset, logits):
    tgt_output_ids = dataset.tgt_output_ids
    if self.time_major:
      tgt_output_ids = tf.transpose(tgt_output_ids)
    
    crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=tgt_output_ids, logits=logits)

    max_time = data.get_max_time(dataset) 
    weights = tf.sequence_mask(
        dataset.tgt_seq_lens, max_time, dtype=self.dtype)
    if self.time_major:
      weights = tf.transpose(weights)

    loss = tf.reduce_sum(
        crossent * weights) / tf.to_float(self.batch_size)
    return loss
