r"""Executable for training Seq2Seq model.

You need to compile the protobuf module first `seq2seq_pb2.py` by running

  protoc --python_out=/OUTPUT/PATH seq2seq.proto

  and provide a config file (protos/seq2seq.config) containing 
  parameter settings.

To perform training, run
  python run_training.py \
    --src_file_list=/PATH/TO/SRC_FILE_LIST
    --tgt_file_list=/PATH/TO/TGT_FILE_LIST
    --src_vocab_file=/PATH/TO/SRC_VOCAB_FILE
    --tgt_vocab_file=/PATH/TO/TGT_VOCAB_FILE
    --config_file=/PATH/TO/CONFIG_FILE
    --out_dir=/PATH/TO/OUT_DIR
"""
import os

import tensorflow as tf
import numpy as np

from google.protobuf import text_format
from protos import seq2seq_pb2
import model_runners 
import eval_utils
import model_runners_utils

flags = tf.app.flags

flags.DEFINE_list('src_file_list', None, 'list of source sequence files.')
flags.DEFINE_list('tgt_file_list', None, 'list of target sequence files.')
flags.DEFINE_string('src_vocab_file', None, 'path to the source vocab file.')
flags.DEFINE_string('tgt_vocab_file', None, 'path to the target vocab file.')
flags.DEFINE_string('config_file', None, 'path to the protobuf config file.')
flags.DEFINE_string('out_dir', '/tmp/seq2seq/train', 'path to output directory ' 
    'where checkpoints and tensorboard log file are located.')

FLAGS = flags.FLAGS


def main(_):
  config = seq2seq_pb2.Seq2SeqModel()
  text_format.Merge(open(FLAGS.config_file).read(), config)
  prediction_model = model_runners_utils.build_prediction_model(config, 'train')
  dataset = model_runners_utils.build_dataset(config, 'train')
  optimizer, learning_rate = model_runners_utils.build_optimizer(
      config.optimization)

  model_trainer = model_runners.Seq2SeqModelTrainer(
      prediction_model, config.optimization.max_grad_norm)

  tensor_dict = model_trainer.train(
      FLAGS.src_file_list, 
      FLAGS.tgt_file_list,
      FLAGS.src_vocab_file,
      FLAGS.tgt_vocab_file,
      dataset,
      optimizer)

  persist_saver = model_runners_utils.create_persist_saver(None)

  tables_initializer = tf.tables_initializer()
  iterator_initializer = dataset.iterator_initializer
  weights_initializer = tf.global_variables_initializer()

  sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True),
                    graph=tf.get_default_graph())

  sess.run(tables_initializer)
  sess.run(iterator_initializer)
  sess.run(weights_initializer)

  writer = tf.summary.FileWriter(FLAGS.out_dir)

  loss_list = []
  predict_count_list = []
  batch_size_list = []
  grad_norm_list = []

  for _ in range(config.optimization.num_train_steps):
    result_dict = sess.run(tensor_dict)
    writer.add_summary(result_dict['summary'], result_dict['global_step'])

    if (result_dict['global_step'] != 0 and 
        result_dict['global_step'] % config.steps_per_stats == 0):
      print('ppl: %.3f, grad norm: %.3f, global_step: %d' % (
          eval_utils.compute_perplexity(
              loss_list, batch_size_list, predict_count_list), 
          np.sum(grad_norm_list) / config.steps_per_stats,
          result_dict['global_step'])) 
      loss_list = []
      predict_count_list = []
      batch_size_list = []
      grad_norm_list = []

    if (result_dict['global_step'] != 0 and result_dict['global_step'] % 
        config.steps_ckpt == 0):
      persist_saver.save(sess, os.path.join(FLAGS.out_dir, 'seq2seq.ckpt'), 
          global_step=result_dict['global_step'])
      
    loss_list.append(result_dict['loss'])
    predict_count_list.append(result_dict['predict_count'])
    batch_size_list.append(result_dict['batch_size'])
    grad_norm_list.append(result_dict['grad_norm'])

  persist_saver.save(sess, os.path.join(FLAGS.out_dir, 'seq2seq.ckpt'), 
      global_step=result_dict['global_step'])
  writer.close()

if __name__ == '__main__':
  tf.flags.mark_flag_as_required('src_file_list')
  tf.flags.mark_flag_as_required('tgt_file_list')
  tf.flags.mark_flag_as_required('src_vocab_file')
  tf.flags.mark_flag_as_required('tgt_vocab_file')
  tf.flags.mark_flag_as_required('config_file')

  tf.app.run()
  
