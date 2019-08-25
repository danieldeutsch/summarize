local parse_boolean(x) =
  local lower = std.asciiLower(x);
  if lower == "true" then
    true
  else
    false;

local data_dir = std.extVar("DATA_DIR");
local use_context = parse_boolean(std.extVar("USE_CONTEXT"));
local embed_size = 200;
local hidden_size = 512;

{
  "dataset_reader": {
    "type": "cloze-pointer-generator",
    "lazy": false,
    "document_tokenizer": {
      "type": "word",
      "word_splitter": {
        "type": "just_spaces"
      }
    },
    "topic_tokenizer": {
      "type": "word",
      "word_splitter": {
        "type": "just_spaces"
      }
    },
    "context_tokenizer": {
      "type": "word",
      "word_splitter": {
        "type": "just_spaces"
      }
    },
    "cloze_tokenizer": {
      "type": "word",
      "word_splitter": {
        "type": "just_spaces"
      },
      "start_tokens": ["@start@"],
      "end_tokens": ["@end@"],
    },
    "document_token_indexers": {
      "tokens": {
        "type": "single_id",
        "lowercase_tokens": true
      }
    },
    "max_document_length": 200,
    "max_context_length": 100,
    "max_cloze_length": 50
  },
  "vocabulary": {
    "max_vocab_size": 50000,
    "pretrained_files": {
      "tokens": "(http://nlp.stanford.edu/data/glove.6B.zip)#glove.6B.200d.txt"
    },
    "tokens_to_add": {
      "tokens": ["@start@", "@end@", "@copy@"]
    }
  },
  "train_data_path": data_dir + "/train.jsonl.gz",
  "validation_data_path": data_dir + "/valid.jsonl.gz",
  "test_data_path": data_dir + "/test.jsonl.gz",
  // We can look at the test data because we only use pretrained words in the vocabulary
  "datasets_for_vocab_creation": ["train", "validation", "test"],
  "model": {
    "type": "cloze-pointer-generator",
    "document_token_embedder": {
      "tokens": {
        "type": "embedding",
        "embedding_dim": embed_size,
        "trainable": false,
        "pretrained_file": "(http://nlp.stanford.edu/data/glove.6B.zip)#glove.6B.200d.txt"
      }
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
      "type": "mlp",
      "encoder_size": hidden_size,
      "decoder_size": hidden_size,
      "attention_size": hidden_size
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
      "input_size": embed_size,
      "hidden_size": hidden_size
    },
    "use_input_feeding": false,
    "loss_normalization": "summary_length",
    "coverage_loss_weight": 0.0,
    "beam_search": {
      "beam_size": 10,
      "max_steps": 100
    },
    "use_context": use_context,
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
    "sorting_keys": [["document", "num_tokens"]],
    "instances_per_epoch": 2000
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
    "validation_metric": "+R2-F1",
    "num_epochs": 20,
    "cuda_device": 0,
    "shuffle": true
  }
}