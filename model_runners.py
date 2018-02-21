from __future__ import print_function
from __future__ import division

import tensorflow as tf

import data
import model_helper


class _BaseModelRunner(object):
  mode = None
  def __init__(self, builder, hparams):
#    print(type(self).mode)
    tf.contrib.learn.ModeKeys.validate(type(self).mode)
    self._graph = tf.Graph()
    with self._graph.as_default():
      if type(self).mode == tf.contrib.learn.ModeKeys.INFER:
        self._src_placeholder = tf.placeholder(
            shape=[None], dtype=tf.string, name="src_placeholder")
        self._batch_size_placeholder = tf.placeholder(
            shape=[], dtype=tf.int64, name="batch_size_placeholder")
        self._dataset = data.Seq2SeqDataset(
            hparams,
            type(self).mode,
            self._src_placeholder,
            self._batch_size_placeholder)
      else:
        self._dataset = data.Seq2SeqDataset(hparams, type(self).mode)
      self._tables_initializer = tf.tables_initializer()
      self._tables_initialized = False

      self._model = builder(hparams, self.dataset, type(self).mode)
      if type(self).mode == tf.contrib.learn.ModeKeys.TRAIN:
        self._global_step = tf.Variable(0, trainable=False, name="global_step")
      self._global_variables_initializer = tf.global_variables_initializer()
      
      self._params = tf.trainable_variables()
      self._saver = tf.train.Saver(
          self._params, max_to_keep=hparams.num_keep_ckpts)

  @property
  def graph(self):
    return self._graph

  @property
  def dataset(self):
    return self._dataset

  @property
  def model(self):
    return self._model

  def restore_params_from(self, sess, ckpt_dir):
    if not self._tables_initialized:
      sess.run(self._tables_initializer)
      self._tables_initialized = True

    latest_ckpt = tf.train.latest_checkpoint(ckpt_dir)
    if latest_ckpt:
      print("%s model is loading params from %s..." % (type(self).mode, latest_ckpt))
      self._saver.restore(sess, latest_ckpt)
    else:
      print("%s model is creating fresh params..." % type(self).mode)
      sess.run(self._global_variables_initializer)

  def persist_params_to(self, sess, ckpt_dir):
    print("%s model is saving params to %s" % (type(self).mode, ckpt_dir))
    self._saver.save(sess, ckpt_dir)


class Seq2SeqModelTrainer(_BaseModelRunner):
  mode = tf.contrib.learn.ModeKeys.TRAIN
  def __init__(self, builder, hparams):
    super(Seq2SeqModelTrainer, self).__init__(
        builder=builder,
        hparams=hparams)
    
    with self.graph.as_default():
      self.word_count = self.dataset.get_word_count()
      self.predict_count = self.dataset.get_predict_count()
      self.learning_rate = self._get_learning_rate(hparams)
      self.update_op, self.grad_norm, grad_norm_summary = \
          self._create_update_op(hparams)
      self.train_summary = tf.summary.merge([
          tf.summary.scalar("lr", self.learning_rate),
          tf.summary.scalar("train_loss", self.model.loss)
      ] + grad_norm_summary)
      
  @property
  def global_step(self):
    return self._global_step

  def _get_learning_rate(self, hparams):
    learning_rate = tf.constant(hparams.learning_rate)
    learning_rate = self._get_learning_rate_warmup(hparams, learning_rate)
    learning_rate = self._get_learning_rate_decay(hparams, learning_rate)
    return learning_rate

  def _get_learning_rate_warmup(self, hparams, learning_rate):
    warmup_steps = hparams.warmup_steps
    warmup_scheme = hparams.warmup_scheme

    if warmup_scheme == "t2t":
      warmup_factor = tf.exp(tf.log(0.01) / warmup_steps)
      inv_decay = warmup_factor**(
          tf.to_float(warmup_steps - self.global_step))
    else:
      pass

    return tf.cond(
        self.global_step < hparams.warmup_steps,
        lambda: inv_decay * learning_rate,
        lambda: learning_rate,
        name="learning_rate_warmup_cond")

  def _get_learning_rate_decay(self, hparams, learning_rate):
    if hparams.decay_scheme in ["luong5", "luong10", "luong234"]:
      decay_factor = 0.5
      if hparams.decay_scheme == "luong5":
        start_decay_step = int(hparams.num_train_steps / 2)
        decay_times = 5
      elif hparams.decay_scheme == "luong10":
        start_decay_step = int(hparams.num_train_steps / 2)
        decay_times = 10
      elif hparams.decay_scheme == "luong234":
        start_decay_step = int(hparams.num_train_steps * 2 / 3)
        decay_times = 4
      remain_steps = hparams.num_train_steps - start_decay_step
      decay_steps = int(remain_steps / decay_times)
    elif not hparams.decay_scheme:
      start_decay_step = hparams.num_train_steps
      decay_steps = 0
      decay_factor = 1.0
    elif hparams.decay_scheme:
      pass

    return tf.cond(
        self.global_step < start_decay_step,
        lambda: learning_rate,
        lambda: tf.train.exponential_decay(
            learning_rate,
            (self.global_step - start_decay_step),
            decay_steps, decay_factor, staircase=True),
        name="learning_rate_decay_cond")

  def _create_update_op(self, hparams):
    if hparams.optimizer == "sgd":
      opt = tf.train.GradientDescentOptimizer(self.learning_rate)
#      tf.summary.scalar("lr", self.learning_rate)
    elif hparams.optimizer == "adam":
      opt = tf.train.AdamOptimizer(self.learning_rate)
    else:
      pass

    params = self._params
    gradients = tf.gradients(
        self.model.loss,
        params,
        colocate_gradients_with_ops=hparams.colocate_gradients_with_ops)

    clipped_grads, grad_norm_summary, grad_norm = model_helper.gradient_clip(
        gradients, hparams.max_gradient_norm)

    update_op = opt.apply_gradients(
        zip(clipped_grads, params), global_step=self.global_step)

    return update_op, grad_norm, grad_norm_summary  

  def train(self, sess):
    return sess.run([self.update_op,
                     self.model.loss,
                     self.predict_count,
                     self.train_summary,
                     self.global_step,
                     self.word_count,
                     self.model.batch_size,
                     self.grad_norm,
                     self.learning_rate])

  def eval(self, sess):
    return sess.run([self.model.loss,
                     self.predict_count,
                     self.model.batch_size])


class Seq2SeqModelEvaluator(_BaseModelRunner):
  mode = tf.contrib.learn.ModeKeys.EVAL
  def __init__(self, builder, hparams):

    super(Seq2SeqModelEvaluator, self).__init__(
        builder=builder,
        hparams=hparams)

    with self.graph.as_default():
      self.predict_count = self.dataset.get_predict_count()

  def eval(self, sess):
    return sess.run([self.model.loss,
                     self.predict_count,
                     self.model.batch_size])


class Seq2SeqModelInferencer(_BaseModelRunner):
  mode = tf.contrib.learn.ModeKeys.INFER
  def __init__(self, builder, hparams):

    super(Seq2SeqModelInferencer, self).__init__(
      builder=builder,
      hparams=hparams)

    with self.graph.as_default():
      self.sample_words = self.dataset.reverse_tgt_vocab_table.lookup(
          tf.to_int64(self.model.sample_id))

  @property
  def src_placeholder(self):
    return self._src_placeholder

  @property
  def batch_size_placeholder(self):
    return self._batch_size_placeholder

  def infer(self, sess):
    return sess.run([self.model.logits, self.model.sample_id, self.sample_words])

  def decode(self, sess):
    # without beam search:
    # sample_words = [N, T] or [T, N]
    # with beam search:
    # sample_words = [N, T, B] or [T, N, B]
    _, _, sample_words = self.infer(sess)

    if self.model.time_major:
      sample_words = sample_words.transpose()
    elif sample_words.ndim == 3:
      sample_words = sample_words.transpose([2, 0, 1])

    # sample_words = [N, T] or [B, N, T]
    return sample_words

  def _get_infer_summary(self):
    pass
