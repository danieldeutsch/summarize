local document_namespace = "document_tokens";
local summary_namespace = "summary_tokens";

{
  "dataset_reader": {
    "type": "sds-abstractive",
    "max_document_length": 400,
    "max_summary_length": 50,
    "summary_tokenizer": {
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
    "summary_token_indexers": {
      "tokens": {
        "type": "single_id",
        "namespace": summary_namespace,
        "lowercase_tokens": true
      }
    }
  },
  "train_data_path": "summarize/tests/fixtures/data/sds.jsonl",
  "validation_data_path": "summarize/tests/fixtures/data/sds.jsonl",
  "model": {
    "type": "sds-seq2seq",
    "summary_namespace": summary_namespace,
    "document_token_embedder": {
      "tokens": {
        "type": "embedding",
        "embedding_dim": 20,
        "vocab_namespace": document_namespace
      }
    },
    "summary_token_embedder": {
      "type": "embedding",
      "embedding_dim": 10,
      "vocab_namespace": summary_namespace
    },
    "encoder": {
      "type": "lstm",
      "input_size": 20,
      "hidden_size": 20,
      "bidirectional": true
    },
    "attention": {
      "type": "cosine"
    },
    "attention_layer": {
      "input_dim": 40 + 40,
      "hidden_dims": 40,
      "num_layers": 1,
      "activations": "tanh"
    },
    "bridge": {
      "share_parameters": false,
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
    "beam_size": 5,
    "min_output_length": 5,
    "max_output_length": 10,
    "metrics": [
      {
        "type": "python-rouge",
        "ngram_orders": [2],
        "namespace": summary_namespace
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
