import tensorflow as tf
import model_runners_utils
from protos import seq2seq_pb2
from google.protobuf import text_format
import model_runners 
import numpy as np
import eval_utils 


config = seq2seq_pb2.Seq2SeqModel()

text_format.Merge(open('protos/seq2seq.config').read(), config)

prediction_model = model_runners_utils.build_prediction_model(config, 'infer')
dataset = model_runners_utils.build_dataset(config, 'infer')

model_inferencer = model_runners.Seq2SeqModelInferencer(prediction_model, 
    config.decoding.beam_width,
    config.decoding.length_penalty_weight,
    config.decoding.sampling_temperature,
    maximum_iterations=None,
    random_seed=config.random_seed)

to_be_run_dict = model_inferencer.infer(
    ['/home/chaoji/Desktop/nmt/nmt_data/tst2013.vi'],
    '/home/chaoji/Desktop/nmt/nmt_data/vocab.vi',
    '/home/chaoji/Desktop/nmt/nmt_data/vocab.en',
    dataset)


restore_saver = model_runners_utils.create_restore_saver()


tables_initializer = tf.tables_initializer()
iterator_initializer = dataset.iterator_initializer

sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True), graph=tf.get_default_graph())


sess.run(tables_initializer)
sess.run(iterator_initializer)
restore_saver.restore(sess, '/home/chaoji/Desktop/nmt/vi_en/nmt_model/translation.ckpt-11473')

while True:
  try:
    result_dict = sess.run(to_be_run_dict['decoded_symbols'])
  except tf.errors.OutOfRangeError:
    break

  tgt_seqs = eval_utils.decoded_symbols_to_strings(result_dict['decoded_symbols'])
  eval_utils.write_predicted_target_sequences(tgt_seqs, 'local.txt')



