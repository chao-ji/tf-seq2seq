"""
Trainer
Evaluator
Inferencer
"""

import tensorflow as tf
import model_runners_utils
from protos import seq2seq_pb2
from google.protobuf import text_format
import model_runners 
import numpy as np
import eval_utils


config = seq2seq_pb2.Seq2SeqModel()

text_format.Merge(open('protos/seq2seq.config').read(), config)

initializer = model_runners_utils.build_initializer(config.initializer, config.random_seed)
prediction_model = model_runners_utils.build_prediction_model(config.prediction_model, config.src_vocab_size,
  config.tgt_vocab_size, 'eval', initializer)
dataset = model_runners_utils.build_dataset(config.dataset, config.src_vocab_size, config.tgt_vocab_size, 'eval', config.random_seed)

model_evaluator = model_runners.Seq2SeqModelEvaluator(prediction_model)

to_be_run_dict = model_evaluator.evaluate(
    ['/home/chaoji/Desktop/nmt/nmt_data/tst2012.vi'],
    ['/home/chaoji/Desktop/nmt/nmt_data/tst2012.en'],
    config.dataset.src_vocab_file,
    config.dataset.tgt_vocab_file,
    dataset)

restore_saver = model_runners_utils.create_restore_saver()

tables_initializer = tf.tables_initializer()
iterator_initializer = dataset.iterator_initializer

sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True), graph=tf.get_default_graph())

sess.run(tables_initializer)
sess.run(iterator_initializer)
restore_saver.restore(sess, '/home/chaoji/Desktop/nmt/vi_en/nmt_model/translation.ckpt-12516')


L = []
P = []
B = []
while True:
  try:
    result_dict = sess.run(to_be_run_dict)
  except tf.errors.OutOfRangeError:
    break
  L.append(result_dict['loss'])
  P.append(result_dict['predict_count'])
  B.append(result_dict['batch_size'])



