r"""Executable for performing internal evaluation using Seq2Seq model.

You need to compile the protobuf module first `seq2seq_pb2.py` by running

  protoc --python_out=/OUTPUT/PATH seq2seq.proto

  and provide a config file (protos/seq2seq.config) containing 
  parameter settings.

To run internal evaluation, run

  python run_internal_evaluation.py \ 
    --ckpt_path=/PATH/TO/CKPT \
    --config_file=/PATH/TO/CONFIG_FILE \
    --src_file=/PATH/TO/SRC_FILE \
    --tgt_file=/PATH/TO/TGT_FILE \
    --src_vocab_file=/PATH/TO/SRC_VOCAB_FILE \
    --tgt_vocab_file=/PATH/TO/TGT_VOCAB_FILE \
"""

import tensorflow as tf
import numpy as np

from google.protobuf import text_format
from protos import seq2seq_pb2
import model_runners 
import eval_utils
import model_runners_utils

flags = tf.app.flags

flags.DEFINE_string('src_file', None, 'path to source sequence file.')
flags.DEFINE_string('tgt_file', None, 'path to target sequence file.')
flags.DEFINE_string('src_vocab_file', None, 'path to the source vocab file.')
flags.DEFINE_string('tgt_vocab_file', None, 'path to the target vocab file.')
flags.DEFINE_string('config_file', None, 'path to the protobuf config file.')
flags.DEFINE_string('ckpt_path', None, 'path to checkpoint file holding trained'
    ' variables.')

FLAGS = flags.FLAGS


def main(_):
  config = seq2seq_pb2.Seq2SeqModel()
  text_format.Merge(open(FLAGS.config_file).read(), config)
  prediction_model = model_runners_utils.build_prediction_model(config, 'eval')
  dataset = model_runners_utils.build_dataset(config, 'eval')
  model_evaluator = model_runners.Seq2SeqModelEvaluator(prediction_model)

  tensor_dict = model_evaluator.evaluate(
      FLAGS.src_file,
      FLAGS.tgt_file,
      FLAGS.src_vocab_file,
      FLAGS.tgt_vocab_file,
      dataset)

  restore_saver = model_runners_utils.create_restore_saver()

  tables_initializer = tf.tables_initializer()
  iterator_initializer = dataset.iterator_initializer

  sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True), 
      graph=tf.get_default_graph())

  sess.run(tables_initializer)
  sess.run(iterator_initializer)
  restore_saver.restore(sess, FLAGS.ckpt_path)

  loss_list = []
  predict_count_list = []
  batch_size_list = []
  while True:
    try:
      result_dict = sess.run(tensor_dict)
    except tf.errors.OutOfRangeError:
      break
    loss_list.append(result_dict['loss'])
    predict_count_list.append(result_dict['predict_count'])
    batch_size_list.append(result_dict['batch_size'])


  ppl = eval_utils.compute_perplexity(
      loss_list, batch_size_list, predict_count_list)
  print('Per-word perplexity:', ppl)

if __name__ == '__main__':
  tf.flags.mark_flag_as_required('src_file')
  tf.flags.mark_flag_as_required('tgt_file')
  tf.flags.mark_flag_as_required('src_vocab_file')
  tf.flags.mark_flag_as_required('tgt_vocab_file')
  tf.flags.mark_flag_as_required('config_file')
  tf.flags.mark_flag_as_required('ckpt_path')

  tf.app.run()

