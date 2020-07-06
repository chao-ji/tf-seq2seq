# TensorFlow seq2seq model

<p align="center">
  <img src="g3doc/files/seq2seq.png" width="900">
</p>


This is a TensorFlow 2.x implementation of the seq2seq model augmented with attention mechanism for neural machine translation. Follow this [guide](https://github.com/chao-ji/tf-seq2seq/blob/master/g3doc/Build_seq2seq_model.md) for a conceptual understanding about how seq2seq model works. 


## Data Preparation, Training, Evaluation, Attention Weights Visualization 
The implementation of **seq2seq** model is designed to have the same command line inference face as the [Transformer](https://github.com/chao-ji/tf-transformer) implementation. Follow that link for detailed instructions on data preparation, training, evaluation and attention weights visualization.

### Visualize Attention Weights 
Unlike [Transformer](https://github.com/chao-ji/tf-transformer), the seq2seq model augmented with attention mechanism involves only *target-to-source* attention. Shown below is the attention weights w.r.t each source token (English) when translating the target token (German) one at a time.

<p align="center">
  <img src="g3doc/files/alignment.png" width="900">
  English-to-German Translation 
</p>


