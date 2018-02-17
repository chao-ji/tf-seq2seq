from __future__ import print_function

import codecs

import numpy as np
import tensorflow as tf

def load_src_sents(inference_input_file):
  """inference.load_data"""
  with codecs.getreader("utf-8")(
      tf.gfile.GFile(inference_input_file, mode="rb")) as f:
    inference_data = f.read().splitlines()

  return inference_data


def translate_from_file(hparams, model_runner, sess, src_file, tgt_file=None):
  """train.run_external_eval
  train._external_eval
  nmt_utils.decode_and_evaluate"""
  src_sents = load_src_sents(src_file)
  tgt_sents = translate_from_data(hparams, model_runner, sess, src_sents, tgt_file=None)

  return tgt_sents


def get_translation(token_arr, hparams):
  tgt_eos = hparams.eos.encode("utf-8")
  token_list = list(token_arr) 
  if tgt_eos and tgt_eos in token_list:
    token_list = token_list[:token_list.index(tgt_eos)]

  return b" ".join(token_list)


def translate_from_data(hparams, model_runner, sess, src_sents, tgt_file=None):
  assert type(model_runner).mode == tf.contrib.learn.ModeKeys.INFER

  feed_dict = {model_runner.src_placeholder: src_sents,
      model_runner.batch_size_placeholder: hparams.infer_batch_size}

  model_runner.dataset.init_iterator(sess, feed_dict)

  tgt_sents = []

  while True:
    try:
      sample_words = model_runner.decode(sess)
      if sample_words.ndim == 2:
        sample_words = np.expand_dims(sample_words, 0)
      batch_size = sample_words.shape[1]

#      for sent_id in range(batch_size):
#        tgt_sents.append(get_translation(sample_words[0][sent_id], hparams))

      tgt_sents.extend(map(lambda token_arr:
          get_translation(token_arr, hparams), sample_words[0]))

    except tf.errors.OutOfRangeError:
      print("done, translated %d sentences" % len(tgt_sents))
      break

  if tgt_file:
    with codecs.getwriter("utf-8")(
        tf.gfile.GFile(tgt_file, mode="wb")) as trans_f:
      trans_f.write("")

      for sent in tgt_sents:
        trans_f.write((sent + b"\n").decode("utf-8"))

  return tgt_sents

  
