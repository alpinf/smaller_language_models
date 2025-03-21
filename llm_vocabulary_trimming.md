# Shrinking Large Language Models for Specific Languages

With impressive performance, large language models (LLMs) have become the de facto approach for most NLP tasks. Among the LLMs that are trained and shared daily, many are multilingual and some are reportedly proficient in more than 100 languages. Yet, performance varies between the supported languages, and unsurprisingly, performance on lower-resource languages is inferior. Probably the most common approach to adapt a language model to a certain language is fine-tuning. Indeed, fine-tuning can improve performance on the target language, but the size of the LLM isn't reduced.

One hypothesis is that multilingual LLMs contain superfluous information with respect to any specific language, such as tokens in a different script.

The goal of this work was to investigate methods for shrinking open-source multilingual models, and in particular methods that are specifically oriented towards maintaining the capabilities of the LLMs in the target language, thus achieving the same level of performance in that language, with a smaller model.

At the moment, this repository contains the code for vocabulary trimming of different models architectures. Our notebooks about pruning, knowledge distillation and quantization may be published in the future.

## Vocabulary Trimming

Vocabulary trimming is a technique that reduces the size of the model's vocabulary, which in turn reduces the size of the embedding layers and other related parameters. The idea is to keep only a subset of the vocabulary that better represents the target language or corpus.
We show below trimming the vocabulary of multilingual BERT ([mBERT](https://arxiv.org/abs/1810.04805) for French. mBERT is an encoder-only BERT model pre-trained on Wikipedia articles from over 100 languages. We have also applied the same technique to mT5, an encoder-decoder model, and mGPT, a decoder-only model, the notebooks for which are [linked below](#vocabulary-trimming-examples).

### 1. Loading the Model and Tokenizer

We start by loading the model and its tokenizer [from HuggingFace](https://huggingface.co/bert-base-multilingual-cased).

```python
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
bert_model = BertModel.from_pretrained("bert-base-multilingual-cased")
```

### 2. Selecting the Vocabulary

To create a smaller, monolingual model, we need to identify the vocabulary tokens that are relevant for the target language, in this case, French. For that, we use the [Leipzig Corpora Collection](https://wortschatz.uni-leipzig.de/en/download/French), the French _News_ section of 2023 with 1 million sentences.

We tokenize the entire corpus with the original mBERT tokenizer and keep track of the most frequent tokens to select them for the reduced vocabulary.

```python
# Load French dataset
df_fr = pd.read_csv(path_to_dataset, sep='\t', header=None, quoting=csv.QUOTE_NONE)
df_fr.columns = ["idx", "text"]
cnt = Counter()

# Count token frequencies
for text in df_fr.text:
    tokens = bert_tokenizer.encode(text)
    cnt.update(tokens)
```

```yaml
Total tokens in dataset: 33M
Unique tokens in dataset: 34K
Coverage of mBERT vocabulary: 28.2%
```

The French corpus contains about 33 million tokens, and approximately 34,000 unique ones. These unique tokens cover only 28.2% of the mBERT vocabulary, indicating that a significant portion of the vocabulary is not relevant to French (to the extent it is represented in the corpus).

Next, we define a new vocabulary with the most frequent tokens from the French corpus. In particular, we aim to retain enough tokens to cover 99.9% of the corpus occurrences, which requires keeping 23,590 tokens (19.7% of mBERT's ~120K tokens). This significant reduction in vocabulary size while maintaining high corpus coverage suggests that many of the original vocabulary tokens appear very rarely or not at all in typical French text.

```python
percentage_to_keep = 0.999
cum_sum = 0

for i, (k, v) in enumerate(cnt.most_common()):
    cum_sum += v
    if cum_sum / sum(cnt.values()) > percentage_to_keep:
        break
    num_tokens = i + 1 # we save the number of tokens to keep

print(f"\nWe will keep {num_tokens} tokens to cover {percentage_to_keep * 100}% of the corpus")
```

**Output:**

```yaml
We will keep 23590 tokens to cover 99.9% of the dataset
```

### 3. Keeping Special Tokens

When trimming the vocabulary, it's important to retain certain tokens that are essential for the model's functionality. In the case of BERT, the first 103 tokens are special tokens that need to be preserved for several reasons:

- **[PAD]**: this token is used for padding sequences to ensure uniform input size

- **[UNK]**: represents any unknown token that is not in the tokenizer's vocabulary

- **[CLS]**: a special token added at the beginning of every sequence and is used as a pooled representation for classification tasks

- **[SEP]**: separates different text segments within a single input sequence

In addition to these 4 tokens, BERT's tokenizer includes 99 unused tokens (`[unused1]`, `[unused2]`, ..., `[unused99]`). These tokens are placeholders that can be repurposed for custom vocabulary or specific tasks during fine-tuning, making them valuable for maintaining flexibility in the model.

```python
special_tokens_ids = set(range(103))
```

### 4. Reducing the Model's Embedding Layer

Once we have chosen the new vocabulary, we need to modify the model's embedding layer to reflect this reduced vocabulary. The embedding layer of a BERT model consists of a matrix where each row corresponds to a token in the vocabulary and each column represents a dimension in the embedding space. The size of this matrix is therefore directly proportional to the number of tokens in the vocabulary.

To reduce the model's embedding layer:

1. **Create a New Embedding Matrix**: We initialize a new embedding matrix with dimensions corresponding to the number of tokens in the reduced vocabulary and the embedding size of the original model, this new matrix is 80% smaller than the original embedding matrix.

2. **Map Old Embeddings to New Embeddings**: We iterate over the tokens in the new vocabulary and copy their corresponding embeddings from the original embedding matrix to the new embedding matrix. This process ensures that the new matrix retains the same embeddings for the tokens that were preserved in the reduced vocabulary.

3. **Re-tie Weights**: In BERT models, the input embeddings and the output layer often share weights, a technique known as _weight tying_. After modifying the embedding layer, we need to re-tie these weights to ensure consistency throughout the model.

Here is a simplified version of the code used to adjust the embedding layer:

```python
def select_embeddings(model, old_vocab, new_vocab):
    # 1 - Create a new embedding matrix
    old_embeddings = model.get_input_embeddings()
    old_num_tokens, old_embedding_dim = old_embeddings.weight.size()

    ...

    new_num_tokens = len(new_vocab)
    new_embeddings = nn.Embedding(new_num_tokens, old_embedding_dim)

    # 2 - Map old embeddings to new embeddings
    for new_id, old_id in enumerate(new_vocab):
        new_embeddings.weight.data[new_id] = old_embeddings.weight.data[old_id]
    model.set_input_embeddings(new_embeddings)
    model.config.vocab_size = new_num_tokens

    # 3 - Re-tie weights
    model.tie_weights()

    return new_embeddings

new_embs = select_embeddings(bert_model, bert_vocab, kept_ids)
```

### 5. Saving the Trimmed Model

Lastly, we save the modified model and tokenizer, which includes saving the model's weights, configuration, and the new tokenizer.

```python
bert_tokenizer.save_pretrained('final_bert-base-fr-cased_normal')
bert_model.save_pretrained('final_bert-base-fr-cased_normal')
```

### Vocabulary Trimming Examples

For a complete implementation of the process, refer to the notebooks linked below:

---

1. **mBERT (Encoder Only)**  
   Vocabulary trimming can be applied to encoder-only models like mBERT, which is useful for tasks such as text classification and named entity recognition.

   - [Notebook Demonstrating Vocabulary Trimming on mBERT](https://colab.research.google.com/github/alpinf/smaller_llms/blob/main/notebooks/vocab_trim_mBERT.ipynb)

2. **mT5 (Encoder + Decoder)**  
   For models like mT5 that have both an encoder and a decoder, vocabulary trimming can reduce the size of both components, which is beneficial for tasks that require text generation in addition to understanding.

   - [Notebook Demonstrating Vocabulary Trimming on mT5](https://colab.research.google.com/github/alpinf/smaller_llms/blob/main/notebooks/vocab_trim_mT5.ipynb)

3. **mGPT (Decoder Only)**  
   Decoder-only models like mGPT can also benefit from vocabulary trimming, particularly for language generation tasks where only certain tokens are needed.
   - [Notebook Demonstrating Vocabulary Trimming on mGPT](https://colab.research.google.com/github/alpinf/smaller_llms/blob/main/notebooks/vocab_trim_mGPT.ipynb)
