local parse_boolean(x) =
  local lower = std.asciiLower(x);
  if lower == "true" then
    true
  else
    false;

local use_topics = parse_boolean(std.extVar("USE_TOPICS"));
local use_context = parse_boolean(std.extVar("USE_CONTEXT"));

local embed_size = 200;
local sentence_encoder_size = embed_size;
local topic_encoder_size = embed_size;
local context_encoder_size = sentence_encoder_size;
local decoder_hidden_size = 600;

{
  "dataset_reader": {
    "type": "cloze-extractive",
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
    "max_num_sentences": 80,
    "max_sentence_length": 50,
    "max_context_length": 100
  },
  "vocabulary": {
    "pretrained_files": {
      "tokens": "(http://nlp.stanford.edu/data/glove.6B.zip)#glove.6B.200d.txt"
    },
    "only_include_pretrained_words": true
  },
  "train_data_path": "https://danieldeutsch.s3.amazonaws.com/summarize/data/deutsch2019/train.v1.1.jsonl.gz",
  "validation_data_path": "https://danieldeutsch.s3.amazonaws.com/summarize/data/deutsch2019/valid.v1.1.jsonl.gz",
  "test_data_path": "https://danieldeutsch.s3.amazonaws.com/summarize/data/deutsch2019/test.v1.1.jsonl.gz",
  // We can look at the test data because we only use pretrained words in the vocabulary
  "datasets_for_vocab_creation": ["train", "validation", "test"],
  "model": {
    "type": "cloze-extractive-baseline",
    "use_topics": use_topics,
    "use_context": use_context,
    "token_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "embedding_dim": embed_size,
          "trainable": false,
          "pretrained_file": "(http://nlp.stanford.edu/data/glove.6B.zip)#glove.6B.200d.txt"
        }
      }
    },
    "sentence_encoder": {
      "type": "bag_of_embeddings",
      "embedding_dim": embed_size,
      "averaged": true
    },
    "topic_encoder": {
      "type": "bag_of_embeddings",
      "embedding_dim": embed_size,
      "averaged": true
    },
    "topic_layer": {
      "input_dim": embed_size,
      "hidden_dims": context_encoder_size,
      "num_layers": 1,
      "activations": "linear"
    },
    "context_encoder": {
      "type": "gru",
      "input_size": embed_size,
      "hidden_size": context_encoder_size / 2,
      "bidirectional": true
    },
    "attention": {
      "type": "mlp",
      "encoder_size": context_encoder_size,
      "decoder_size": sentence_encoder_size,
      "attention_size": sentence_encoder_size
    },
    "attention_layer": {
      "input_dim": context_encoder_size + sentence_encoder_size,
      "hidden_dims": sentence_encoder_size,
      "num_layers": 1,
      "activations": "linear"
    },
    "sentence_extractor": {
      "type": "rnn",
      "rnn": {
        "type": "gru",
        "input_size": sentence_encoder_size,
        "hidden_size": decoder_hidden_size / 2,
        "bidirectional": true,
      },
      "feed_forward": {
        "input_dim": decoder_hidden_size,
        "num_layers": 2,
        "hidden_dims": [100, 1],
        "activations": ["relu", "linear"],
        "dropout": [0.25, 0.0]
      },
      "dropout": 0.25
    },
    "max_words": 200,
    "dropout": 0.25,
    "metrics": [
      {
        "type": "python-rouge",
        "ngram_orders": [1, 2],
        "max_words": 200
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
    "cuda_device": 0,
    "num_serialized_models_to_keep": 1
  }
}
