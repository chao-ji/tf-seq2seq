import codecs

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import logging

import bleu

HEIGHT = 12
WIDTH = 12
DPI = 256

def compute_perplexity(loss_list, batch_size_list, predict_count_list):
  """Computes per-word perplexity of prediction over a set of sequences.

  Args:
    loss_list: a list of floats, the scalar loss for each batch.
    batch_size_list: a list of ints, the num of sequences of each batch.
    predict_count_list: a list of ints, total num of predicted symbols in
      target sequences of each batch. 

  Returns:
    ppl: float scalar, perplexity.
  """
  if not isinstance(loss_list, np.ndarray):
    loss_list = np.array(loss_list)
  if not isinstance(batch_size_list, np.ndarray):
    batch_size_list = np.array(batch_size_list)
  if not isinstance(predict_count_list, np.ndarray):
    predict_count_list = np.array(predict_count_list)
  ppl = np.exp(np.sum(loss_list * batch_size_list) / np.sum(predict_count_list))
  return ppl

      
def decoded_symbols_to_strings(decoded_symbols, tgt_eos=b'</s>'):
  """For each decoded sequence in a batch, represented as a list of strings,
  discard the entries after the first end-of-sentence marker, and joins the 
  remaining strings with ' '.  

  Args:
    decode_symbols: string np array of shape [K, batch, max_time], where K = 1
      for greedy and sampling decoder, and K = beam_width for beam search 
      decoder. 
    tgt_eos: string scalar, target end-of-sentence marker.

  Returns:
    tgt_seqs: a list of strings, containing predicted sequences (e.g. sentences)
      where individual symbols (e.g. words) are delimited by a space.
  """
  tgt_seqs = []
  for symbols_list in decoded_symbols[0]:
    symbols_list = list(symbols_list)

    if tgt_eos and (tgt_eos in symbols_list):
      symbols_list = symbols_list[:symbols_list.index(tgt_eos)]
    else:
      logging('Decoding ends prematurely (EOS not found in decoded sequence). '
          'Consider increasing `maximum_iterations`.')

    tgt_seqs.append(b' '.join(symbols_list))
  return tgt_seqs


def write_predicted_target_sequences(tgt_seqs, tgt_file):
  """Writes the predicted target sequences to file.

  Args:
    tgt_seqs: a list of strings, predicted target sequences.
    tgt_file: string scalar, the path to the file to write `tgt_seqs` to.
  """
  with codecs.getwriter('utf-8')(
      tf.gfile.GFile(tgt_file, mode='ab')) as f:
    f.write('')
    for seq in tgt_seqs:
      f.write((seq + b'\n').decode('utf-8'))


def visualize_alignment(result_dict, tgt_eos='</s>'):
  """Visualize the alignment between source and decoded target sequence.

  Args:
    result_dict: dict mapping from name to numpy array with following entries:
      --decode_symbols: float np array fo shape [K, batch, max_time_tgt], where 
        K = 1 for greedy and sampling decoder, and K = beam_width for beam 
        search decoder.
      --alignment: float np array of shape [max_time_tgt, K, max_time_src], 
        where max_time_tgt = the maximum length of decoded sequences over a 
        batch, K = batch_size (not in beam-search mode) or 
        batch_size * beam_width (with batch_size varying slower, in 
        beam-search mode), max_time_src = the maximum length of source sequences
        over a batch, holding the alignment scores of each target symbol 
        w.r.t each input source symbol.
      --input_symbols: string np array of shape [batch, max_time_src], the
        input sequences of symbols.

  Returns:
    image: uint8 4-D numpy containing the image to be visualized.
  """
  input_seq = list(map(lambda s: s.decode('utf-8'), 
      result_dict['input_symbols'][0]))
  output_seq = list(map(lambda s: s.decode('utf-8'), 
      result_dict['decoded_symbols'][0, 0]))

  true_decode_len = output_seq.index(tgt_eos)
  output_seq = output_seq[:true_decode_len]
  alignment = result_dict['alignment'][:, 0, :]
  alignment = alignment[:true_decode_len, :]
  plt.imshow(alignment, cmap='gray')
  plt.xticks(range(len(input_seq)))
  plt.yticks(range(len(output_seq)))

  ax = plt.gca()
  ax.xaxis.tick_top()
  ax.set_xticklabels(input_seq, rotation=90)
  ax.set_yticklabels(output_seq)

  fig = plt.gcf()
  fig.set_size_inches(HEIGHT, WIDTH)
  fig.set_dpi(DPI)  
  fig.canvas.draw()
  data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
  data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
  return np.expand_dims(data, axis=0)


def compute_bleu(ref_file, trans_file):
  """Compute BLEU scores and handling BPE."""
  max_order = 4
  smooth = False

  ref_files = [ref_file]
  reference_text = []
  for reference_filename in ref_files:
    with codecs.getreader("utf-8")(
        tf.gfile.GFile(reference_filename, "rb")) as fh:
      reference_text.append(fh.readlines())

  per_segment_references = []
  for references in zip(*reference_text):
    reference_list = []
    for reference in references:
      reference = reference.strip()
      reference_list.append(reference.split(" "))
    per_segment_references.append(reference_list)

  translations = []
  with codecs.getreader("utf-8")(tf.gfile.GFile(trans_file, "rb")) as fh:
    for line in fh:
      line = line.strip()
      translations.append(line.split(" "))

  # bleu_score, precisions, bp, ratio, translation_length, reference_length
  bleu_score, _, _, _, _, _ = bleu.compute_bleu(
      per_segment_references, translations, max_order, smooth)
  return 100 * bleu_score

