local embed_size = 128;
local hidden_size = 256;
local encoder_hidden_size = hidden_size * 2;
local decoder_hidden_size = hidden_size;

{
  "dataset_reader": {
    "type": "sds-abstractive",
    "lazy": false,
    "document_tokenizer": {
      "type": "word",
      "word_splitter": {
        "type": "just_spaces"
      }
    },
    "summary_tokenizer": {
      "type": "word",
      "word_splitter": {
        "type": "just_spaces"
      },
      "start_tokens": ["@start@", "@sent_start@"],
      "end_tokens": ["@sent_end@", "@end@"],
      "in_between_tokens": ["@sent_end@", "@sent_start@"]
    },
    "document_token_indexers": {
      "tokens": {
        "type": "single_id",
        "lowercase_tokens": true
      }
    },
    "max_document_length": 400,
    "max_summary_length": 100
  },
  "vocabulary": {
    "max_vocab_size": 50000,
    "tokens_to_add": {
      "tokens": ["@start@", "@end@", "@sent_start@", "@sent_end@"]
    }
  },
  "train_data_path": "https://s3.amazonaws.com/danieldeutsch/summarize/data/cnn-dailymail/cnn-dailymail/train.tokenized.v1.0.jsonl.gz",
  "validation_data_path": "https://s3.amazonaws.com/danieldeutsch/summarize/data/cnn-dailymail/cnn-dailymail/valid.tokenized.v1.0.jsonl.gz",
  "model": {
    "type": "sds-seq2seq",
    "document_token_embedder": {
      "tokens": {
        "type": "embedding",
        "embedding_dim": embed_size
      }
    },
    "encoder": {
      "type": "lstm",
      "input_size": embed_size,
      "hidden_size": encoder_hidden_size / 2,
      "bidirectional": true
    },
    "hidden_projection_layer": {
      "input_dim": encoder_hidden_size,
      "hidden_dims": decoder_hidden_size,
      "num_layers": 1,
      "activations": "relu"
    },
    "memory_projection_layer": {
      "input_dim": encoder_hidden_size,
      "hidden_dims": decoder_hidden_size,
      "num_layers": 1,
      "activations": "relu"
    },
    "attention": {
      "type": "mlp",
      "encoder_size": encoder_hidden_size,
      "decoder_size": decoder_hidden_size,
      "attention_size": encoder_hidden_size
    },
    "attention_layer": {
      "input_dim": encoder_hidden_size + decoder_hidden_size,
      "hidden_dims": decoder_hidden_size,
      "num_layers": 1,
      "activations": "linear"
    },
    "decoder": {
      "type": "lstm",
      "input_size": embed_size + decoder_hidden_size,
      "hidden_size": decoder_hidden_size
    },
    "use_input_feeding": true,
    "beam_size": 4,
    "min_output_length": 35,
    "max_output_length": 100,
    "metrics": [
      {
        "type": "python-rouge",
        "ngram_orders": [1, 2]
      }
    ]
  },
  "iterator": {
    "type": "bucket",
    "batch_size": 16,
    "sorting_keys": [["document", "num_tokens"]]
  },
  "validation_iterator": {
    "type": "bucket",
    "batch_size": 16,
    "sorting_keys": [["document", "num_tokens"]]
  },
  "trainer": {
    "optimizer": {
      "type": "adagrad",
      "lr": 0.15,
      "initial_accumulator_value": 0.1
    },
    "grad_norm": 2,
    "num_epochs": 15,
    "validation_metric": "+R2-F1",
    "cuda_device": 0
  }
}
