# Build a seq2seq Model
This guide overviews the key steps and concepts to build a seq2seq model implemented as an encoder-decoder architecture with attention mechanism. While I did not intend to cover all details, I tried to make it easy-to-follow to give you a clear picture of how the seq2seq model works under the hood. It is recommended that you are familiar with how recurrent neural network works (you can check this [post](https://karpathy.github.io/2015/05/21/rnn-effectiveness/) if you don't).

### Encoder-Decoder Architecture

At a high level, the seq2seq model is composed of an encoder RNN and a decoder RNN, where the encoder operates on a **source** sequence, and the decoder operates on a **target** sequence. 

For example, in the case of NMT the source sequence is a list of four tokens in Japenese "広", "い", "部", "屋", and the target sequence is a list of two tokens in English "large", "room". The end goal is to output the phrase "large room" given the input "広い部屋". To achieve this, the encoder RNN reads the source sequence tokens one at a time, and meanwhile updates its hidden layer (i.e. real-valued multi-dimensional vector) accordingly, before handing over to the decoder.

Like the encoder, the decoder updates its hidden layer upon reading each target sequence token. But unlike the encoder that just reads in the source sequence without generating any output, the decoder is supposed to generate a prediction sequence. Note that the final state of encoder's hidden layer contains all the knowledge the encoder has about the source sequence, so it can be passed to the decoder to initialize its hidden layer. 


<p align="center">
  <img src="files/tokens.png" width="350">
  <br>
  Encoder-decoder architecture is trained on source input sequence, target input and target output Sequence.
</p>

The decoder is designed to generate prediction in two modes. 

At the time of *training*, a **target input** sequence is used to send signals to the decoder to emit predicted tokens one at a time, and a **target output** sequence serves as groundtruth. Specifically, we respectively prepend and append the target sequence "large room" with special marker `sos` (start of sequence) and `eos` (end of sequence) to obtain the target input and target output sequence. Upon reading `sos`, the decoder is given the signal that prediction has started, and is expected to emit the first token "large" in the target sequence, and on "large" emit the second token "room". When reading the last token "room", the decoder's hidden layer should be in a state that signifies it has come to the end of the prediction, and is expected to emit `eos`. 

At the time of *inference*, however, the decoder is on its own to generate the target sequence. It works by feeding the token generated at last time step to the decoder to generate the token at the current step. The detail of the inference mode is beyond the scope of this guide.

### Prepare the Data
<p align="center">
  <img src="files/lookup.png" width="450">
  <br>
  Convert raw tokens to real-valued embedding vectors.
</p>

As noted above, we need to prepare a target input and a target output sequence, together with the source sequence, to train an encoder-decoder architecture. In addition they need to be numerically represented to be directly processed by the RNNs .

The idea is to convert the raw tokens into IDs, which are just the token indices into the source or target vocabulary. For example, "広" has index 0 in Japanese vocabulary, and "room" has index 3 in English vocabulary. Note that the target vocabulary has been augmented with `sos` and `eos`. 

These integer IDs need to be further mapped to real-valued vectors. For a source vocabulary of size `Vs` and a target vocabulary of size `Vt`, we initialize a real-valued matrix of shape `[Vs, D]` and another of shape `[Vt, D]` where `D` is the desired size of embedding. The matrices can be used as lookup tables to retrieve the embedding vectors for each ID. For example, "広" is represented as the 0th row of the source matrix and "room" is represented as the 3rd row of the target matrix.   

In this way the source sequence ends up being represented as a list of 4 D-dimentional vectors and target input and output sequences as two lists of 3 D-dimensional vectors.

### Build the Encoder
Here the encoder is implemented as a two-layer bi-directional RNN. It operates like a regular RNN except that it reads the source sequence in two directions. The forward RNN (bottom) reads input sequence from left to right, while the backward RNN (top) reads the same sequence in reverse direction.

Suppose we have a source sequence of length three, represented as three matrices `x1`, `x2`, `x3` of shape `[N, D]` (`N` is the batch size, `D` is embedding size). The sequence is presented to the forward RNN in the order of `x1`, `x2`, `x3`, and to the backward RNN in the order of `x3`, `x2`, `x1`. The forward RNN will generate a sequence of hidden states `f1`, `f2`, `f3` in the order of the presented matrices, and likewise `b1`, `b2`, `b3` for backward RNN. These hidden states are paired if they are resulted from the same input token: `[f1, b3]`,  `[f2, b2]`, `[f3, b1]`. The two hidden states are concatenated along the depth dimension, so finally we are left with 3 hidden state matrices of shape `[N, 2D]`. These matrices (`encoder_outputs`) form a *memory lookup table* to be utilized by the attention mechanism which will be covered later.

<p align="center">
  <img src="files/encoder.png" width="500">
  <br>
  Bidirectional Encoder Unfolded for a Sequence of Length Three.
</p>

### Build the Decoder

<p align="center">
  <img src="files/decoder.png" width="500">
  <br>
  Decoder with Attention Mechanism Unfolded for a Sequence of Length Three.
</p>

The decoder RNN is a two-layer RNN with LSTM cells. The first layer (bottom) takes as input an *augmented* version of the embedding vectors of target tokens: they are concatenated with **attention vectors** from the previous time step. The attention vectors are used to inform the decoder about the attention decisions (i.e. which source tokens to pay attention to) made in previous step.

The second layer RNN (top) takes as input the hidden state from the first layer, and its own hidden state is fed to the **memory layer** above and the **attention layer** to generate the **attention vector** for the next time step (how attention vectors are generated is covered in the following part).

### Attention Mechanism
Unlike the vanilla seq2seq model where the information about the entire source sequence (can be *arbitrarily long*) is squashed into a *fixed length* vector (i.e. encoder's final hidden state), the attention mechanism allows the decoder to access the encoder hidden states at each time step, so that the decoder can adaptively learn which source sequence token to pay attention to when generating the next target token.

##### Memory Layer
<p align="center">
  <img src="files/memory.png" width="400">
  <br>
  Memory layer using Luong's multiplicative scoring function.
</p>

The attention mechanism is implemented by adding a **memory layer** and an **attention layer** on top of the decoder RNN. Generally, the memory layer compares a `query` vector (i.e. hidden state of the topmost decoder RNN) with the raw memory `encoder_outputs` holding the encoder's hidden states at every time step to see which step has the best match with `query`.

The comparison relies on a scoring function to quantitatively measure the matching quality. Shown in the above figure is the *Luong*'s multiplicative scoring function (another being *Bahdanau*'s additive scoring function), where a transformed memory `keys` is computed by performing projection on `encoding_outputs` and then a dot-product is computed between `query` and each time slice of `keys` (i.e. `keys[i, :]`). The outcome of dot-product then goes through a softmax function so we are left with `scores` -- the output of scoring function. Its time dimension has a size of `Tsrc` (length of source sequence), so the larger the value, the better `query` fits with that source token. Finally the `scores` is used to compute `context` -- the weighted average (over the time dimension) of the raw memory `encoder_outputs`.


##### Attention Layer
<p align="center">
  <img src="files/attention.png" width="300">
  <br>
  Attention layer generates attention vector.
</p>

The attention layer generates an **attention vector** given the `context` from memory layer and the `query` from the hidden state of decoder RNN. It simply concatenates `context` and `query` along the depth dimension and projects it into `D`-dimensional space.

The attention vector goes out in two branches: 
* As noted earlier it is combined with the embedding vector of target token from the following time step, and is fed to the decoder RNN in the first layer.
* It goes through an output projection layer to produce the final prediction logits.

<p align="center">
  <img src="files/projection.png" width="400">
  <br>
  Projection Layer. 
</p>

A softmax loss is computed between the logits (`[Ttgt, N, Vtgt]`) and target output sequence (`[Ttgt, N]`), and gradients are backpropagated to all weighted layers.

### Trainable Variables
Below is a list of all trainable variables of the seq2seq model described above.

* Encoder embedding matrix `[Vsrc, D]`
* Encoder forward RNN kernel `W: [2D, 4D]`
* Encoder forward RNN bias `b: [4D]`
* Encoder backward RNN kernel `W: [2D, 4D]`
* Encoder backward RNN bias `b: [4D]`
* Decoder embedding matrix `[Vtgt, D]`
* Decoder RNN kernel at layer 1 `W: [3D, 4D]`
* Decoder RNN bias at layer 1 `b: [4D]`
* Decoder RNN kernel at layer 2 `W: [2D, 4D]`
* Decoder RNN bias at layer 2 `b: [4D]`
* Memory layer kernel `W: [2D, D]`
* Attention layer kernel `W: [3D, D]`
* Attention layer multiplier (a scalar value)
* Project layer kernel `W: [D, Vtgt]`

### References
  1. Official TensorFlow seq2seq model tutorial, https://github.com/tensorflow/nmt 
  2. Effective Approaches to Attention-based Neural Machine Translation, Luong et al. 2015
  3. Neural Machine Translation by Jointly Learning to Align and Translate, Bahdanau et al. 2015
  4. Sequence to Sequence Learning with Neural Networks, Sutskever et al. 2014

