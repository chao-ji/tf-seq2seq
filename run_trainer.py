import glob
import os

import tensorflow as tf
from absl import app
from absl import flags

from commons import utils
from commons import dataset
from commons import tokenization
from model import Seq2SeqModel
from model_runners import SequenceTransducerTrainer


SUFFIX = '*.tfrecord'
flags.DEFINE_string(
    'data_dir', None, 'Path to the directory storing all TFRecord files (with '
        'pattern *train*) for training.')
flags.DEFINE_string(
    'vocab_path', None, 'Path to the vocabulary file.')
flags.DEFINE_string(
    'model_dir', None, 'Path to the directory that checkpoint files will be '
        'written to.')

flags.DEFINE_integer(
    'hidden_size', 512, 'The dimensionality of the embedding vector.')
flags.DEFINE_float(
    'dropout_rate', 0.1, 'Dropout rate for the Dropout layers.')
flags.DEFINE_string(
    'attention_model', 'luong', 'Type of attention model '
        '("luong" or "bahdanau").')

flags.DEFINE_integer(
    'batch_size', 128, 'Static batch size.')
flags.DEFINE_integer(
    'num_buckets', 8, 'Number of sequence length buckets.')
flags.DEFINE_integer(
    'bucket_width', 10, 'Size of each sequence length bucket.') 
flags.DEFINE_integer(
    'max_length', 64, 'Source or target seqs longer than this will be filtered'
        ' out.')
flags.DEFINE_integer(
    'num_parallel_calls', 8, 'Num of TFRecord files to be processed '
        'concurrently.')

flags.DEFINE_float(
    'learning_rate', 2.0, 'Base learning rate.')
flags.DEFINE_float(
    'learning_rate_warmup_steps', 16000, 'Number of warm-ups steps.')
flags.DEFINE_float(
    'optimizer_adam_beta1', 0.9, '`beta1` of Adam optimizer.')
flags.DEFINE_float(
    'optimizer_adam_beta2', 0.997, '`beta2` of Adam optimizer.')
flags.DEFINE_float(
    'optimizer_adam_epsilon', 1e-9, '`epsilon` of Adam optimizer.')

flags.DEFINE_float(
    'label_smoothing', 0.1, 'Amount of probability mass withheld for negative '
        'classes.')
flags.DEFINE_integer(
    'num_steps', 100000, 'Num of training iterations (minibatches).')
flags.DEFINE_integer(
    'save_ckpt_per_steps', 5000, 'Every this num of steps to save checkpoint.')


FLAGS = flags.FLAGS

def main(_):
  data_dir = FLAGS.data_dir
  vocab_path = FLAGS.vocab_path
  model_dir = FLAGS.model_dir

  hidden_size = FLAGS.hidden_size
  dropout_rate = FLAGS.dropout_rate
  attention_model = FLAGS.attention_model

  batch_size = FLAGS.batch_size
  num_buckets = FLAGS.num_buckets
  bucket_width = FLAGS.bucket_width
  max_length = FLAGS.max_length
  num_parallel_calls = FLAGS.num_parallel_calls

  learning_rate = FLAGS.learning_rate
  learning_rate_warmup_steps = FLAGS.learning_rate_warmup_steps
  optimizer_adam_beta1 = FLAGS.optimizer_adam_beta1
  optimizer_adam_beta2 = FLAGS.optimizer_adam_beta2
  optimizer_adam_epsilon = FLAGS.optimizer_adam_epsilon

  label_smoothing = FLAGS.label_smoothing
  num_steps = FLAGS.num_steps
  save_ckpt_per_steps = FLAGS.save_ckpt_per_steps

  # seq2seq model
  subtokenizer = tokenization.restore_subtokenizer_from_vocab_files(vocab_path)
  vocab_size = subtokenizer.vocab_size
  model = Seq2SeqModel(vocab_size=vocab_size, 
                       hidden_size=hidden_size,
                       dropout_rate=dropout_rate,
                       attention_model=attention_model)

  # training dataset
  builder = dataset.StaticBatchDatasetBuilder(
      batch_size=batch_size,
      shuffle=True,
      max_length=max_length,
      num_parallel_calls=num_parallel_calls,
      num_buckets=num_buckets,
      bucket_width=bucket_width)
  filenames = sorted(glob.glob(os.path.join(data_dir, SUFFIX)))
  train_ds = builder.build_dataset(filenames)

  # learning rate and optimizer
  optimizer = tf.keras.optimizers.Adam(
      utils.LearningRateSchedule(learning_rate,
                                 hidden_size,
                                 learning_rate_warmup_steps),
      optimizer_adam_beta1,
      optimizer_adam_beta2,
      epsilon=optimizer_adam_epsilon)

  # checkpoint
  ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)

  # build trainer and start training
  trainer = SequenceTransducerTrainer(model, label_smoothing)
  trainer.train(
      train_ds, optimizer, ckpt, model_dir, num_steps, save_ckpt_per_steps)


if __name__  == '__main__':
  flags.mark_flag_as_required('data_dir')
  flags.mark_flag_as_required('model_dir')
  flags.mark_flag_as_required('vocab_path')
  app.run(main)
