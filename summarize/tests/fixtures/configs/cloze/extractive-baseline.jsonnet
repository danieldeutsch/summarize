{
  "dataset_reader": {
    "type": "cloze-extractive",
    "max_num_sentences": 50,
    "max_sentence_length": 15,
    "max_context_length": 20,
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
  "train_data_path": "summarize/tests/fixtures/data/cloze.jsonl",
  "validation_data_path": "summarize/tests/fixtures/data/cloze.jsonl",
  "model": {
    "type": "cloze-extractive-baseline",
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
    "topic_encoder": {
      "type": "lstm",
      "input_size": 20,
      "hidden_size": 20,
      "bidirectional": true
    },
    "topic_layer": {
      "input_dim": 40,
      "hidden_dims": 40,
      "num_layers": 1,
      "activations": "linear"
    },
    "context_encoder": {
      "type": "lstm",
      "input_size": 20,
      "hidden_size": 20,
      "bidirectional": true
    },
    "attention": {
      "type": "mlp",
      "encoder_size": 40,
      "decoder_size": 40,
      "attention_size": 40
    },
    "attention_layer": {
      "input_dim": 40 + 40,
      "hidden_dims": 40,
      "num_layers": 1,
      "activations": "linear"
    },
    "use_topics": true,
    "use_context": true,
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
