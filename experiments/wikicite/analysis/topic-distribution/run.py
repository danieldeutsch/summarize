# flake8: noqa
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
from collections import Counter
from tqdm import tqdm

sys.path.append('../summarize')

from summarize.data.io import JsonlReader


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    topic_counts = Counter()
    with JsonlReader(args.input_jsonl) as f:
        for instance in tqdm(f):
            headings = instance['headings']
            for topic in headings:
                topic_counts[topic.lower()] += 1

    total_topk = 0
    topk_topics = []
    topk_counts = []
    for topic, count in topic_counts.most_common(15):
        topk_topics.append(topic)
        topk_counts.append(count)
        total_topk += count

    other_count = sum(topic_counts.values()) - total_topk
    topk_topics.append('other')
    topk_counts.append(other_count)

    for i in range(len(topk_counts)):
        topk_counts[i] /= 1000

    plt.figure()
    fig, ax = plt.subplots()
    x = list(reversed(range(len(topk_counts))))
    ax.barh(x, topk_counts)
    ax.set_yticks(x)
    ax.set_yticklabels(topk_topics)
    ax.set_xlabel('Thousands of Occurrences')
    ax.set_title('Topic Frequencies')
    plt.tight_layout()
    plt.savefig(f'{args.output_dir}/topic-distribution.png', dpi=1000)

    count_histogram = [0] * 10
    for count in topic_counts.values():
        if count >= 10:
            count_histogram[-1] += 1
        else:
            count_histogram[count - 1] += 1

    plt.figure()
    fig, ax = plt.subplots()
    x = list(range(len(count_histogram)))
    labels = list(range(1, len(count_histogram))) + ['10+']
    ax.bar(x, count_histogram)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel('Number of Occurrences')
    ax.set_ylabel('Number of Topics')
    ax.set_title('Topic Frequency Histogram')
    plt.tight_layout()
    plt.savefig(f'{args.output_dir}/frequency-histogram.png', dpi=1000)

    print('Total unique topics: ', len(topic_counts))

    print('Sample unique topics')
    print('--------------------')
    for topic, _ in topic_counts.most_common()[-50:]:
        print(topic)


if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument('input_jsonl', help='The WikiCite dataset to analyze')
    argp.add_argument('output_dir', help='The directory where the plot should be written')
    args = argp.parse_args()
    main(args)
