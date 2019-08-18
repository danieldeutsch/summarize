local encoder = std.extVar("ENCODER");

// The size of the decoder's input changes based on the encoder choice
local decoder_input_size =
  if encoder == "avg" then 200
  else if encoder == "rnn" then 400;

{
  "dataset_reader": {
    "type": "sds-extractive",
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
    },
    "max_num_sentences": 50
  },
  "vocabulary": {
    "pretrained_files": {
      "tokens": "(http://nlp.stanford.edu/data/glove.6B.zip)#glove.6B.200d.txt"
    },
    "only_include_pretrained_words": true
  },
  "train_data_path": "https://s3.amazonaws.com/danieldeutsch/summarize/data/kedzie2018/cnn-dailymail/train.v1.0.jsonl.gz",
  "validation_data_path": "https://s3.amazonaws.com/danieldeutsch/summarize/data/kedzie2018/cnn-dailymail/valid.v1.0.jsonl.gz",
  "model": {
    "type": "sds-extractive-baseline",
    "token_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "embedding_dim": 200,
          "trainable": false,
          "pretrained_file": "(http://nlp.stanford.edu/data/glove.6B.zip)#glove.6B.200d.txt",
        }
      }
    },
    "sentence_encoder":
      if encoder == "avg" then {
        "type": "boe",
        "embedding_dim": 200,
        "averaged": true
      }
      else if encoder == "rnn" then {
        "type": "gru",
        "input_size": 200,
        "hidden_size": 200,
        "bidirectional": true
      }
    ,
    "sentence_extractor": {
      "type": "rnn",
      "rnn": {
        "type": "gru",
        "input_size": decoder_input_size,
        "hidden_size": 300,
        "bidirectional": true,
      },
      "feed_forward": {
        "input_dim": 600,
        "num_layers": 2,
        "hidden_dims": [100, 1],
        "activations": ["relu", "linear"],
        "dropout": [0.25, 0.0]
      },
      "dropout": 0.25
    },
    "max_words": 100,
    "dropout": 0.25,
    "metrics": [
      {
        "type": "python-rouge",
        "ngram_orders": [2],
        "max_words": 100,
        "remove_stopwords": true
      }
    ]
  },
  "iterator": {
    "type": "bucket",
    "batch_size": 32,
    "sorting_keys": [["document", "num_fields"]]
  },
  "validation_iterator": {
    "type": "bucket",
    "batch_size": 32,
    "sorting_keys": [["document", "num_fields"]]
  },
  "trainer": {
    "optimizer": {
      "type": "adam",
      "lr": 0.0001
    },
    "grad_norm": 5,
    "num_epochs": 20,
    "validation_metric": "+R2-R",
    "cuda_device": 0
  }
}
