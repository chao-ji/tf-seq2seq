import tensorflow as tf

import model_runners_utils as utils


class Seq2SeqModelTrainer(object):
  """Performs training using a seq2seq prediction model."""
  def __init__(self, prediction_model, max_grad_norm):
    """Constructor.

    Args:
      prediction_model: a `Seq2SeqPredictionModel` instance.
      max_grad_norm: float scalar, the maximum gradient norm that all gradients
        are clipped to, in order to prevent exploding gradients.
    """
    self._prediction_model = prediction_model
    self._max_grad_norm = max_grad_norm

  def train(self,
            src_file_list,
            tgt_file_list,
            src_vocab_file,
            tgt_vocab_file,
            dataset,
            optimizer):
    """Adds training related ops to the graph.

    Args:
      src_file_list: a list of string scalars, the paths to the source sequence
        text files. 
      tgt_file_list: a list of string scalars, the paths to the corresponding
        target sequence text files.
      src_vocab_file: string scalar, the path to the source vocabulary file,
        where each line contains a single symbol. 
      tgt_vocab_file: string scalar, the path to the target vocabulary file, 
        where each line contains a single symbol. 
      dataset: a `Seq2SeqDataset` instance.
      optimizer: an optimizer instance.

    Returns:
      to_be_run_dict: a dict mapping from tensor/operation names to 
        tensor/operation, the set of tensor/operations that need to be run
        in a `tf.Session`.
    """
    tensor_dict = dataset.get_tensor_dict(
        src_file_list, tgt_file_list, src_vocab_file, tgt_vocab_file)
  
    logits, batch_size = self._prediction_model.predict_logits(
        tensor_dict['src_input_ids'],
        tensor_dict['src_seq_lens'],
        tensor_dict['tgt_input_ids'],
        tensor_dict['tgt_seq_lens'])

    loss = utils.create_loss(tensor_dict,
                             logits,
                             self._prediction_model.time_major)

    global_step = tf.train.get_or_create_global_step()
    grads_and_vars = optimizer.compute_gradients(
        loss, colocate_gradients_with_ops=True)
    grads_and_vars = list(zip(*grads_and_vars))
    grads, vars_ = grads_and_vars[0], grads_and_vars[1]
    clipped_grads, grad_norm_summary, grad_norm = utils.apply_gradient_clip(
        grads, self._max_grad_norm)
    grad_update_op = optimizer.apply_gradients(zip(clipped_grads, vars_),
                                               global_step=global_step)
    predict_count = utils.get_predict_count(tensor_dict['tgt_seq_lens'])

    summary = tf.summary.merge([
          tf.summary.scalar("train_loss", loss)
      ] + grad_norm_summary)

    to_be_run_dict = {'grad_update_op': grad_update_op,
                      'loss': loss,
                      'predict_count': predict_count,
                      'batch_size': batch_size,
                      'grad_norm': grad_norm,
                      'summary': summary,
                      'global_step': global_step}

    return to_be_run_dict 


class Seq2SeqModelEvaluator(object):
  """Performs internal evaluation using a seq2seq prediction model.
  
  Internal evaluation only reports the loss and the resulting perplexity.
  """
  def __init__(self, prediction_model):
    """Constructor.

    Args:
      prediction_model: a `Seq2SeqPredictionModel` instance.
    """
    self._prediction_model = prediction_model

  def evaluate(self,           
               src_file_list,
               tgt_file_list,
               src_vocab_file,
               tgt_vocab_file,
               dataset):
    """Adds evaluation related ops to the graph.

    Args:
      src_file_list: a list of string scalars, the paths to the source sequence
        text files. 
      tgt_file_list: a list of string scalars, the paths to the corresponding
        target sequence text files.
      src_vocab_file: string scalar, the path to the source vocabulary file,
        where each line contains a single symbol. 
      tgt_vocab_file: string scalar, the path to the target vocabulary file, 
        where each line contains a single symbol. 
      dataset: a `Seq2SeqDataset` instance.

    Returns:
      to_be_run_dict: a dict mapping from tensor/operation names to 
        tensor/operation, the set of tensor/operations that need to be run
        by a `tf.Session`.       
    """
    tensor_dict = dataset.get_tensor_dict(
        src_file_list, tgt_file_list, src_vocab_file, tgt_vocab_file)

    logits, batch_size = self._prediction_model.predict_logits(
        tensor_dict['src_input_ids'],
        tensor_dict['src_seq_lens'],
        tensor_dict['tgt_input_ids'],
        tensor_dict['tgt_seq_lens'])

    loss = utils.create_loss(tensor_dict, 
                             logits, 
                             self._prediction_model.time_major)
    predict_count = utils.get_predict_count(tensor_dict['tgt_seq_lens'])

    to_be_run_dict = {'loss': loss,
                      'predict_count': predict_count,
                      'batch_size': batch_size}
    return to_be_run_dict


class Seq2SeqModelInferencer(object):
  """Performs external evaluation and inference using a seq2seq prediction 
  model.

  External evaluation reports BLEU score by comparing a set of predicted target 
  sequences and the corresponding groundtruth target sequences.
  """
  def __init__(self, 
               prediction_model,
               beam_width,
               length_penalty_weight,
               sampling_temperature,
               maximum_iterations,
               random_seed):
    """Constructor

    Args:
      prediction_model: a `Seq2SeqPredictionModel` instance.
      beam_width: int scalar, width for beam seach.
      length_penalty_weight: float scalar, length penalty weight for beam 
        search. Disabled with 0.0
      sampling_temperature: float scalar > 0.0, value to divide the logits by
        before computing the softmax. Larger values (above 1.0) result in more
        random samples, while smaller values push the sampling distribution 
        towards the argmax. 
      maximum_iterations: int scalar or None, max num of iterations for dynamic
        decoding.
      random_seed: int scalar, random seed for sampling decoder.
    """
    self._prediction_model = prediction_model
    self._beam_width = beam_width
    self._length_penalty_weight = length_penalty_weight
    self._sampling_temperature = sampling_temperature
    self._maximum_iterations = maximum_iterations
    self._random_seed = random_seed

  def infer(self,
            src_file_list,
            src_vocab_file,
            tgt_vocab_file,
            dataset):
    """Adds inference related ops to the graph.

    Args:
      src_file_list: a list of string scalars, the paths to the source sequence
        text files. 
      src_vocab_file: string scalar, the path to the source vocabulary file,
        where each line contains a single symbol. 
      tgt_vocab_file: string scalar, the path to the target vocabulary file, 
        where each line contains a single symbol. 
      dataset: a `Seq2SeqDataset` instance.

    Returns:
      to_be_run_dict: a dict mapping from tensor/operation names to 
        tensor/operation, the set of tensor/operations that need to be run
        by a `tf.Session`.   
    """
    tensor_dict = dataset.get_tensor_dict(src_file_list,
                                          src_vocab_file,
                                          tgt_vocab_file)

    indices, states = self._prediction_model.predict_indices(
        tensor_dict['src_input_ids'],
        tensor_dict['src_seq_lens'],
        tensor_dict['tgt_sos_id'],
        tensor_dict['tgt_eos_id'],
        self._beam_width,
        self._length_penalty_weight,
        self._sampling_temperature,
        self._maximum_iterations,
        self._random_seed)

    if self._prediction_model.time_major:
      indices = tf.transpose(indices)
    elif indices.shape.ndims == 3:
      indices = tf.transpose(indices, [2, 0, 1])

    tgt_vocab_table = dataset.get_rev_tgt_vocab_table(tgt_vocab_file)

    decoded_symbols = tgt_vocab_table.lookup(tf.to_int64(indices))

    to_be_run_dict = {'decoded_symbols': decoded_symbols}
    return to_be_run_dict

