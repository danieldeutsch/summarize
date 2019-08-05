local embed_size = 128;
local hidden_size = 512;

{
  "dataset_reader": {
    "type": "sds-pointer-generator",
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
      "tokens": ["@start@", "@end@", "@sent_start@", "@sent_end@", "@copy@"]
    }
  },
  "train_data_path": "https://s3.amazonaws.com/danieldeutsch/summarize/data/cnn-dailymail/cnn-dailymail/train.tokenized.v1.0.jsonl.gz",
  "validation_data_path": "https://s3.amazonaws.com/danieldeutsch/summarize/data/cnn-dailymail/cnn-dailymail/valid.tokenized.v1.0.jsonl.gz",
  "datasets_for_vocab_creation": ["train"],
  "model": {
    "type": "sds-pointer-generator",
    "document_token_embedder": {
      "tokens": {
        "type": "embedding",
        "embedding_dim": embed_size
      }
    },
    "summary_token_embedder": {
      "type": "embedding",
      "embedding_dim": embed_size
    },
    "encoder": {
      "type": "lstm",
      "input_size": embed_size,
      "hidden_size": hidden_size / 2,
      "bidirectional": true
    },
    "bridge": {
      "share_bidirectional_parameters": true,
      "layers": [
        {
          "input_dim": hidden_size / 2,
          "hidden_dims": hidden_size / 2,
          "num_layers": 1,
          "activations": "relu"
        },
        {
          "input_dim": hidden_size / 2,
          "hidden_dims": hidden_size / 2,
          "num_layers": 1,
          "activations": "relu"
        }
      ]
    },
    "attention": {
      "type": "matrix-attention",
      "matrix_attention": {
        "type": "mlp",
        "encoder_size": hidden_size,
        "decoder_size": hidden_size,
        "attention_size": hidden_size
      }
    },
    "attention_layer": {
      "input_dim": hidden_size + hidden_size,
      "hidden_dims": hidden_size,
      "num_layers": 1,
      "activations": "linear"
    },
    "generate_probability_function": {
      "type": "onmt",
      "decoder_dim": hidden_size
    },
    "decoder": {
      "type": "lstm",
      "input_size": embed_size + hidden_size,
      "hidden_size": hidden_size
    },
    "use_input_feeding": true,
    "loss_normalization": "summary_length",
    "coverage_loss_weight": 0.0,
    "beam_search": {
      "beam_size": 10,
      "min_steps": 35,
      "max_steps": 100,
      "disallow_repeated_ngrams": 3,
      "repeated_ngrams_exceptions": [[".", "@sent_end@", "@sent_start@"]],
      "length_penalizer": {
        "type": "wu",
        "alpha": 0.9
      },
      "coverage_penalizer": {
        "type": "onmt",
        "beta": 5
      }
    },
    "metrics": [
      {
        "type": "python-rouge",
        "ngram_orders": [1, 2]
      }
    ],
    "initializer": [
      [".*", {"type": "uniform", "a": -0.1, "b": 0.1}]
    ]
  },
  "iterator": {
    "type": "bucket",
    "batch_size": 16,
    "sorting_keys": [["document", "num_tokens"]],
    "instances_per_epoch": 160000
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
    "learning_rate_scheduler": {
      "type": "multi_step",
      "gamma": 0.5,
      "milestones": std.range(5, 20)
    },
    "grad_norm": 2,
    "num_epochs": 20,
    "cuda_device": 0,
    "shuffle": true
  }
}