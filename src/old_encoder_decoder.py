import collections
import csv
import random

import numpy as np
import tensorflow as tf


def _read_words(filename):
    with tf.gfile.GFile(filename, "r") as f:
        return filter(None, f.read().decode("utf-8").replace(".", " <eos>").
                      replace(", ", " <comma> ").replace("\n", " <eop> ").split())


def _read_sentences(filename):
    with tf.gfile.GFile(filename, "r") as f:
        s = f.read().decode("utf-8").replace(".", " <eos>").replace(", ", " <comma> ").replace("\n", " <eop><EOP_TAG>")
        return filter(None, s.split("<EOP_TAG>"))


def build_vocab(filename):
    data = _read_words(filename)

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(1, len(words) + 1)))

    return word_to_id


def _split(data):
    xs = []
    ys = []
    for (x, y) in data:
        xs.append(x)
        ys.append(y)

    return xs, ys


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

    return _split(ret_data[:tr_end]), _split(ret_data[tr_end:val_end]), _split(ret_data[val_end:]), vocab


# Input - seqs: num_samples*3, labels: num_samples*[list]
# Return X:maxlen*num_samples*3, X_mask: max_len*num_samples, labels: maxlen*num_samples
def prepare_data(seqs, labels, maxlen=None, xdim=3):
    """Create the matrices from the datasets.

    This pad each sequence to the same length: the length of the
    longest sequence or maxlen.

    if maxlen is set, we will cut all sequence to this maximum
    length.

    This swap the axis!
    """
    # Trim all output seqs to have only maxlen steps
    if maxlen is not None:
        Iseqs = []
        Oseqs = []
        for i_seq, o_seq in zip(seqs, labels):
            if len(o_seq) < maxlen:
                Iseqs.append(i_seq)
                Oseqs.append(o_seq)
        seqs = Iseqs
        labels = Oseqs
    else:
        maxlen = 40

    # Pad and compute masks
    ret_X = np.zeros((maxlen, len(seqs), xdim))
    mask_X = np.zeros((maxlen, len(seqs)))
    labels_X = np.zeros((maxlen, len(seqs)))
    for k in range(len(seqs)):
        mask_X[:len(labels[k]), k] = 1
        ret_X[:len(labels[k]), k] = np.asarray(seqs[k])
        labels_X[:len(labels[k]), k] = labels[k]

    return ret_X, mask_X, labels_X


if __name__ == "__main__":
    train, val, test, vocab = get_raw_data("../data/xs1000.txt",
                                    "../data/targets1000.txt")
    c = _read_sentences("../data/targets1000.txt")
    print np.array(train[0]);
    # print val;
    # print test;
    print( len(vocab) )
    print(build_vocab("../data/targets1000.txt"))
