import collections
import csv
import random

import tensorflow as tf


def _read_words(filename):
    with tf.gfile.GFile(filename, "r") as f:
        return filter(None, f.read().decode("utf-8").replace(".", " <eos>").replace("\n", " <eop> ").split())


def _read_sentences(filename):
    with tf.gfile.GFile(filename, "r") as f:
        s = f.read().decode("utf-8").replace(".", " <eos>").replace("\n", " <eop><EOP_TAG>")
        return filter(None, s.split("<EOP_TAG>"))


def build_vocab(filename):
    data = _read_words(filename)

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))

    return word_to_id


def get_raw_data(xs, ys, train_frac=0.7, val_frac=0.2, test_frac=0.1):
    vocab = build_vocab(ys)
    sentences = _read_sentences(ys)
    ret_data = []
    xs = list(csv.reader(open(xs, "r"), delimiter=' '))
    for n in range(len(sentences)):
        words = filter(None, sentences[n].split())
        cur_ys = []
        for word in words:
            cur_ys.append(vocab[word])
        x = map(float, xs[n])
        ret_data.append((x, cur_ys))

    # Randomly shuffle data
    random.shuffle(ret_data)

    # Compute split points
    tr_end = int(train_frac * len(ret_data))
    val_end = tr_end + int(val_frac * len(ret_data))

    return ret_data[:tr_end], ret_data[tr_end:val_end], ret_data[val_end:]


if __name__ == "__main__":
    train, val, test = get_raw_data("/home/sauce/git/upgraded-system/data/xs1000.txt",
                                    "/home/sauce/git/upgraded-system/data/targets1000.txt")
    c = _read_sentences("/home/sauce/git/upgraded-system/data/targets1000.txt")
    print(build_vocab("/home/sauce/git/upgraded-system/data/targets1000.txt"))
