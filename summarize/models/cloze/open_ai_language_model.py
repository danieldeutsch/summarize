import argparse
import json
import numpy as np
import os
import spacy
import sys
from tqdm import tqdm

from summarize.data.io import JsonlReader, JsonlWriter


class OpenAILanguageModel(object):
    def __init__(self,
                 model_dir: str,
                 length: int,
                 temperature: float,
                 top_k: int,
                 seed: int = 4) -> None:
        # Hide the tensorflow dependency
        import tensorflow as tf

        # This class requires loading classes from the gpt-2 repository which
        # will be added to the path. AllenNLP will try to import every module
        # within the package, so we edit the path and add the imports here
        # so that code doesn't run every time the overall package is used
        sys.path.append('external/gpt-2/src')
        import model
        import sample
        from encoder import Encoder

        # Copied from encoder.py
        with open(os.path.join(model_dir, 'encoder.json'), 'r') as f:
            encoder = json.load(f)
        with open(os.path.join(model_dir, 'vocab.bpe'), 'r', encoding="utf-8") as f:
            bpe_data = f.read()
        bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]]
        self.enc = Encoder(encoder=encoder, bpe_merges=bpe_merges)

        hparams = model.default_hparams()
        with open(os.path.join(model_dir, 'hparams.json')) as f:
            hparams.override_from_dict(json.load(f))

        if length is None:
            length = hparams.n_ctx // 2
        elif length > hparams.n_ctx:
            raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

        self.sess = tf.Session(graph=tf.Graph())
        self.sess.__enter__()
        # The random seeds need to be set here for determinism
        np.random.seed(seed)
        tf.set_random_seed(seed)
        batch_size = 1
        self.context = tf.placeholder(tf.int32, [batch_size, None])
        self.output = sample.sample_sequence(
            hparams=hparams,
            length=length,
            context=self.context,
            batch_size=batch_size,
            temperature=temperature,
            top_k=top_k
        )

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(model_dir)
        saver.restore(self.sess, ckpt)

        self.nlp = spacy.load('en_core_web_sm', parser=False, tagger=False, entity=False)

    def _get_first_sentence(self, text: str) -> str:
        # Sometimes a special '<|endoftext|>' token is written. This sometimes
        # messes up the sentence tokenization, so we remove it if it exists
        end_of_text_token = '<|endoftext|>'
        text = text.replace(end_of_text_token, ' ').strip()
        for sentence in self.nlp(text).sents:
            return sentence.text.strip()

    def sample_next_sentence(self, condition_text: str) -> str:
        context_tokens = self.enc.encode(condition_text)
        out = self.sess.run(self.output, feed_dict={
            self.context: [context_tokens]
        })[:, len(context_tokens):]
        output_text = self.enc.decode(out[0])
        first_sentence = self._get_first_sentence(output_text)
        return first_sentence


def main(args):
    model_dir = args.model_dir
    length = args.length
    temperature = args.temperature
    top_k = args.top_k
    seed = args.seed

    lm = OpenAILanguageModel(model_dir, length, temperature, top_k, seed=seed)
    with JsonlWriter(args.output_jsonl) as out:
        with JsonlReader(args.input_jsonl) as f:
            for instance in tqdm(f):
                context = instance['context']
                context = ' '.join(context)

                first_sentence = lm.sample_next_sentence(context)
                output_data = {
                    'cloze': first_sentence
                }
                out.write(output_data)


if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument('model_dir', help='The directory of the pretrained Open AI language model')
    argp.add_argument('input_jsonl', help='The input summary cloze dataset')
    argp.add_argument('output_jsonl', help='The output file')
    argp.add_argument('temperature', type=float, help='Controls the randomness in the Boltzmann distribution')
    argp.add_argument('top_k', type=int, help='Controls the diversity')
    argp.add_argument('--length', type=int, default=100, help='The length of sequence to generate')
    argp.add_argument('--seed', type=int, default=4, help='Sets the random seed')
    args = argp.parse_args()
    main(args)
