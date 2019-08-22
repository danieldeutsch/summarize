local document_namespace = "document_tokens";
local cloze_namespace = "cloze_tokens";

{
  "dataset_reader": {
    "type": "cloze-abstractive",
    "max_document_length": 40,
    "max_context_length": 20,
    "max_cloze_length": 15,
    "cloze_tokenizer": {
      "type": "word",
      "word_splitter": {
        "type": "just_spaces"
      },
      "start_tokens": ["@start@"],
      "end_tokens": ["@end@"]
    },
    "document_token_indexers": {
      "tokens": {
        "type": "single_id",
        "namespace": document_namespace,
        "lowercase_tokens": true
      }
    },
    "cloze_token_indexers": {
      "tokens": {
        "type": "single_id",
        "namespace": cloze_namespace,
        "lowercase_tokens": true
      }
    }
  },
  "train_data_path": "summarize/tests/fixtures/data/cloze.jsonl",
  "validation_data_path": "summarize/tests/fixtures/data/cloze.jsonl",
  "vocabulary": {
    "tokens_to_add": {
      // Using the cloze_namespace variable as the key does not work
      "cloze_tokens": ["@start@", "@end@"]
    }
  },
  "model": {
    "type": "cloze-seq2seq",
    "cloze_namespace": cloze_namespace,
    "document_token_embedder": {
      "tokens": {
        "type": "embedding",
        "embedding_dim": 20,
        "vocab_namespace": document_namespace
      }
    },
    "cloze_token_embedder": {
      "type": "embedding",
      "embedding_dim": 10,
      "vocab_namespace": cloze_namespace
    },
    "encoder": {
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
    "bridge": {
      "share_bidirectional_parameters": false,
      "layers": [
        {
          "input_dim": 40,
          "hidden_dims": 40,
          "num_layers": 1,
          "activations": "relu"
        },
        {
          "input_dim": 40,
          "hidden_dims": 40,
          "num_layers": 1,
          "activations": "relu"
        }
      ]
    },
    "decoder": {
      "type": "lstm",
      "input_size": 10,
      "hidden_size": 40
    },
    "use_input_feeding": false,
    "beam_search": {
      "namespace": cloze_namespace,
      "beam_size": 5,
      "min_steps": 5,
      "max_steps": 10,
      "disallow_repeated_ngrams": 1
    },
    "use_context": true,
    "metrics": [
      {
        "type": "python-rouge",
        "ngram_orders": [2],
        "namespace": cloze_namespace
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
