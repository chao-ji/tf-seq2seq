r"""Executable for making inferences or external evaluation using Seq2Seq model.

You need to compile the protobuf module first `seq2seq_pb2.py` by running

  protoc --python_out=/OUTPUT/PATH seq2seq.proto

  and provide a config file (protos/seq2seq.config) containing 
  parameter settings.

To make inference, run
  python run_inference.py \ 
    --ckpt_path=/PATH/TO/CKPT \
    --config_file=/PATH/TO/CONFIG_FILE \
    --src_file=/PATH/TO/SRC_FILE \
    --src_vocab_file=/PATH/TO/SRC_VOCAB_FILE \
    --tgt_vocab_file=/PATH/TO/TGT_VOCAB_FILE \
To run external evaluation, add set additional flag
    --tgt_file=/PATH/TO/TGT_FILE
"""
import os

import tensorflow as tf
import matplotlib.pyplot as plt

from google.protobuf import text_format
from protos import seq2seq_pb2
import model_runners 
import eval_utils 
import model_runners_utils

OUTPUT_FILE = 'output.txt'

flags = tf.app.flags

flags.DEFINE_string('src_file', None, 'path to source sequence file.')
flags.DEFINE_string('tgt_file', None, 'path to target sequence file.')
flags.DEFINE_string('src_vocab_file', None, 'path to the source vocab file.')
flags.DEFINE_string('tgt_vocab_file', None, 'path to the target vocab file.')
flags.DEFINE_string('config_file', None, 'path to the protobuf config file.')
flags.DEFINE_integer('num_alignments', 20, 'number of alignments to visualize.')
flags.DEFINE_string('ckpt_path', None, 'path to checkpoint file holding trained'
    ' variables.')
flags.DEFINE_string('out_dir', '/tmp/seq2seq/infer', 'path to output directory '
    ' containing output file and log file.')

FLAGS = flags.FLAGS


def main(_):
  config = seq2seq_pb2.Seq2SeqModel()
  text_format.Merge(open(FLAGS.config_file).read(), config)
  prediction_model = model_runners_utils.build_prediction_model(config, 'infer')
  dataset = model_runners_utils.build_dataset(config, 'infer')
  maximum_iterations = (config.decoding.maximum_iterations 
      if config.decoding.HasField('maximum_iterations') else None)

  model_inferencer = model_runners.Seq2SeqModelInferencer(prediction_model, 
      config.decoding.beam_width,
      config.decoding.length_penalty_weight,
      config.decoding.sampling_temperature,
      maximum_iterations=maximum_iterations,
      random_seed=config.random_seed)

  tensor_dict = model_inferencer.infer(
      FLAGS.src_file,
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

  writer = tf.summary.FileWriter(FLAGS.out_dir)

  i = 0
  while True:
    try:
      result_dict = sess.run(tensor_dict)

      decoded_symbols = result_dict['decoded_symbols']
      tgt_seqs = eval_utils.decoded_symbols_to_strings(decoded_symbols)
      eval_utils.write_predicted_target_sequences(
          tgt_seqs, os.path.join(FLAGS.out_dir, OUTPUT_FILE))

      if result_dict['alignment'] is not None:
        if i < FLAGS.num_alignments:
          image = eval_utils.visualize_alignment(result_dict)
          alignment_summary = tf.summary.image('src_tgt_' + str(i + 1), image)
          writer.add_summary(sess.run(alignment_summary))

    except tf.errors.OutOfRangeError:
      break
    i += 1
  writer.close()
  if FLAGS.tgt_file is not None:
    bleu_score = eval_utils.compute_bleu(FLAGS.tgt_file, 
        os.path.join(FLAGS.out_dir, OUTPUT_FILE))    
    print('BLEU score:', bleu_score)
  print('Output file:', os.path.join(FLAGS.out_dir, OUTPUT_FILE))

if __name__ == '__main__':
  tf.flags.mark_flag_as_required('src_file')
  tf.flags.mark_flag_as_required('src_vocab_file')
  tf.flags.mark_flag_as_required('tgt_vocab_file')
  tf.flags.mark_flag_as_required('config_file')
  tf.flags.mark_flag_as_required('ckpt_path')

  tf.app.run()

