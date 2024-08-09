# Sentiment Analysis with Deep Learning using BERT
## __What is BERT?__

__Bidirectional Encoder Representations from Transformers (BERT)__ is a Transformer-based machine learning technique for natural language processing (NLP) pre-training developed by Google.BERT is a deeply bidirectional, unsupervised language representation, pre-trained using only a plain text corpus. Context-free models such as word2vec or GloVe generate a single word embedding representation for each word in the vocabulary, where BERT takes into account the context for each occurrence of a given word.

## What are some variants of BERT?

> BERT has inspired many variants: __RoBERTa, XLNet, MT-DNN, SpanBERT, VisualBERT, K-BERT, HUBERT__ and more. Some variants attempt to compress the model: __TinyBERT, ALERT, DistilBERT__ and more. We describe a few of the variants that outperform BERT in many tasks

> RoBERTa: Showed that the original BERT was undertrained. RoBERTa is trained longer, on more data; with bigger batches and longer sequences; without NSP; and dynamically changes the masking pattern.

> ALBERT: Uses parameter reduction techniques to yield a smaller model. To utilize inter-sentence coherence, ALBERT uses Sentence-Order Prediction (SOP) instead of NSP.
XLNet: Doesn't do masking but uses permutation to capture bidirectional context. It combines the best of denoising autoencoding of BERT and autoregressive language modelling of Transformer-XL.

> MT-DNN: Uses BERT with additional multi-task training on NLU tasks. Cross-task data leads to regularization and more general representations.

## __Dataset__
> We will use the [__SMILE Twitter DATASET__](https://doi.org/10.6084/m9.figshare.3187909.v2)

 ## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) 

```bash
pip install torch torchvision
pip install tqdm
pip install transformers

```

> An introduction to some basic theory behind BERT, and the problem we will be using it to solve

> Explore dataset distribution and some basic preprocessing

> Split dataset into training and validation using stratified approach

> Loading pretrained tokenizer to encode our text data into numerical values (tensors)

> Load in pretrained BERT with custom final layer

> Create dataloaders to facilitate batch processing

> Choose and optimizer and scheduler to control training of model

> Design performance metrics for our problem

> Create a training loop to control PyTorch finetuning of BERT using CPU or GPU acceleration

> Loading finetuned BERT model and evaluate its performance
