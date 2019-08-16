{
  "dataset_reader": {
    "type": "sds-extractive",
    "max_num_sentences": 50,
    "max_sentence_length": 15,
    "tokenizer": {
      "type": "word",
      "word_splitter": {
        "type": "just_spaces"
      }
    },
    "token_indexers": {
      "tokens": {
        "type": "single_id",
        "lowercase_tokens": true
      }
    }
  },
  "train_data_path": "summarize/tests/fixtures/data/sds.jsonl",
  "validation_data_path": "summarize/tests/fixtures/data/sds.jsonl",
  "model": {
    "type": "sds-extractive-baseline",
    "token_embedder": {
      "tokens": {
        "type": "embedding",
        "embedding_dim": 20
      }
    },
    "sentence_encoder": {
      "type": "lstm",
      "input_size": 20,
      "hidden_size": 20,
      "bidirectional": true
    },
    "sentence_extractor": {
      "type": "rnn",
      "rnn": {
        "type": "lstm",
        "input_size": 40,
        "hidden_size": 20,
        "bidirectional": true
      },
      "feed_forward": {
        "input_dim": 40,
        "hidden_dims": 1,
        "num_layers": 1,
        "activations": "linear"
      }
    },
    "max_words": 20,
    "metrics": [
      {
        "type": "python-rouge",
        "ngram_orders": [2]
      }
    ]
  },
  "iterator": {
    "type": "basic",
    "batch_size": 4,
    "instances_per_epoch": 2
  },
  "trainer": {
    "optimizer": "adam",
    "num_epochs": 5,
    "cuda_device": -1
  }
}
