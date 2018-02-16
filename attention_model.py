from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import model
import model_helper

class AttentionModel(model.Seq2SeqModel):

  def _build_decoder_cell(self,
                          hparams,
                          num_layers,
                          num_res_layers,
                          outputs_encoder,
                          src_seq_lens):
    attention_option = hparams.attention
    attention_architecture = hparams.attention_architecture

    num_units = hparams.num_units
    beam_width = hparams.beam_width

    memory = outputs_encoder
    if self.time_major:
      memory = tf.transpose(memory, [1, 0, 2]) 

    if self._use_tile_batch(hparams):
      memory = tf.contrib.seq2seq.tile_batch(memory, beam_width)
      src_seq_lens = tf.contrib.seq2seq.tile_batch(src_seq_lens, beam_width)
     
    attention_mechanism = _create_attention_mechanism(
        attention_option, num_units, memory, src_seq_lens)

    cell = model_helper.create_rnn_cell(
        hparams.unit_type,
        num_units,
        num_layers,
        num_res_layers,
        hparams.forget_bias,
        hparams.dropout,
        self.mode)

    alignment_history = (self.mode == tf.contrib.learn.ModeKeys.INFER and
                         beam_width == 0)

    return tf.contrib.seq2seq.AttentionWrapper(
        cell,
        attention_mechanism,
        attention_layer_size=num_units,
        alignment_history=alignment_history,
        output_attention=hparams.output_attention,
        name="attention")
  
  def _get_decoder_init_states(self, hparams, states_encoder, cell):
    beam_width = hparams.beam_width
    if self._use_tile_batch(hparams):
      states_encoder = tf.contrib.seq2seq.tile_batch(
          states_encoder, beam_width)
      batch_size = self.batch_size * beam_width
    else:
      batch_size = self.batch_size

    if hparams.pass_hidden_state:
      return cell.zero_state(batch_size,
          self.dtype).clone(cell_state=states_encoder)
    else:
      return cell.zero_state(batch_size, self.dtype)


def _create_attention_mechanism(attention_option,
                                num_units,
                                memory,
                                src_seq_lens):

  if attention_option == "luong":
    attention_mechanism = tf.contrib.seq2seq.LuongAttention(
        num_units, memory, memory_sequence_length=source_sequence_length)
  elif attention_option == "scaled_luong":
    attention_mechanism = tf.contrib.seq2seq.LuongAttention(
        num_units,
        memory,
        memory_sequence_length=src_seq_lens,
        scale=True)
  elif attention_option == "bahdanau":
    attention_mechanism = tf.conrib.seq2seq.BahdanauAttention(
        num_units, memory, memory_sequence_length=src_seq_lens)
  elif attention_option == "normed_bahdanau":
    attention_mechanism = tf.conrib.seq2seq.BahdanauAttention(
        num_units,
        memory,
        memory_sequence_length=src_seq_lens,
        normalize=True)
  else:
    raise ValueError("Unknown attention option %s" % attention_option) 

  return attention_mechanism 
