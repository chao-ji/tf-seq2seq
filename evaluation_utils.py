import bleu
import codecs
import tensorflow as tf

def compute_bleu(ref_file, trans_file, subword_option=None):
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
      reference = _clean(reference, subword_option)
      reference_list.append(reference.split(" "))
    per_segment_references.append(reference_list)

  translations = []
  with codecs.getreader("utf-8")(tf.gfile.GFile(trans_file, "rb")) as fh:
    for line in fh:
      line = _clean(line, subword_option=None)
      translations.append(line.split(" "))

  # bleu_score, precisions, bp, ratio, translation_length, reference_length
  bleu_score, _, _, _, _, _ = bleu.compute_bleu(
      per_segment_references, translations, max_order, smooth)
  return 100 * bleu_score


def _clean(sentence, subword_option):
  """Clean and handle BPE or SPM outputs."""
  sentence = sentence.strip()

  # BPE
  if subword_option == "bpe":
    sentence = re.sub("@@ ", "", sentence)

  # SPM
  elif subword_option == "spm":
    sentence = u"".join(sentence.split()).replace(u"\u2581", u" ").lstrip()

  return sentence
