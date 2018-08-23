"""
Trainer
Evaluator
Inferencer
"""
import tensorflow as tf
import numpy as np

from protos import seq2seq_pb2
from google.protobuf import text_format

import model_runners 
import eval_utils
import model_runners_utils


flags = tf.app.flags

flags.DEFINE_list('src_file_list', [], 'list of source sequence files.')
flags.DEFINE_list('tgt_file_list', [], 'list of target sequence files.')
flags.DEFINE_string('src_vocab_file', '', 'path to the source vocabulary file.')
flags.DEFINE_string('tgt_vocab_file', '', 'path to the target vocabulary file.')
flags.DEFINE_string('config_file', '', 'path to the protobuf config file.')

FLAGS = flags.FLAGS

def main(_):
  assert FLAGS.src_file_list, '`src_file_list` is missing.'
  assert FLAGS.tgt_file_list, '`tgt_file_list` is missing.'
  assert FLAGS.src_vocab_file, '`src_vocab_file` is missing.'
  assert FLAGS.tgt_vocab_file, '`tgt_vocab_file` is missing.'
  assert FLAGS.config_file, '`config_file` is missing.'

  config = seq2seq_pb2.Seq2SeqModel()
  text_format.Merge(open(FLAGS.config_file).read(), config)
  prediction_model = model_runners_utils.build_prediction_model(config, 'train')
  dataset = model_runners_utils.build_dataset(config, 'train')
  optimizer, learning_rate = model_runners_utils.build_optimizer(
      config.optimization)

  model_trainer = model_runners.Seq2SeqModelTrainer(
      prediction_model, config.optimization.max_grad_norm)

  to_be_run_dict = model_trainer.train(
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

  writer = tf.summary.FileWriter('.')

  loss_list = []
  predict_count_list = []
  batch_size_list = []
  grad_norm_list = []

  for _ in range(config.optimization.num_train_steps):
    result_dict = sess.run(to_be_run_dict)
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

    if result_dict['global_step'] != 0 and result_dict['global_step'] % 1000 == 0:
      persist_saver.save(sess, './seq2seq.ckpt', global_step=result_dict['global_step'])
      
    loss_list.append(result_dict['loss'])
    predict_count_list.append(result_dict['predict_count'])
    batch_size_list.append(result_dict['batch_size'])
    grad_norm_list.append(result_dict['grad_norm'])

  persist_saver.save(sess, './seq2seq.ckpt', global_step=result_dict['global_step'])

  writer.close()

if __name__ == '__main__':
  tf.app.run()

