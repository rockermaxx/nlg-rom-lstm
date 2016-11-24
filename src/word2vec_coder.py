import collections
import csv
import random

import gensim.models.word2vec as word2vec
import numpy as np
import tensorflow as tf

model = None


# Input : np.array : model.vector_size+3
# Output : [(word, match_score)]*num_words
def get_words(vec, num_words=2):
    return model.similar_by_vector(vec[:-3], topn=num_words)


# Returns model.vector_size dim feature vector
def get_model_feature(word):
    return model[word]


def _read_words(filename):
    with tf.gfile.GFile(filename, "r") as f:
        return filter(None, f.read().decode("utf-8").encode("ascii").replace(".", " <eos> ").
                      replace(", ", " <comma> ").replace("\n", " <eop> ").split())


def _read_sentences(filename):
    with tf.gfile.GFile(filename, "r") as f:
        s = f.read().decode("utf-8").encode("ascii").replace(".", " <eos> ").replace(", ", " <comma> "). \
            replace("\n", " <eop> <EOP_TAG>")
        return filter(None, s.split("<EOP_TAG>"))


def _generate_vector_map(words):
    vec_map = {}
    for word in words:
        vec = np.zeros(model.vector_size + 3)
        if word == "<eop>":
            vec[-1] = 1.
            vec_map["<eop>"] = vec
        elif word == "<eos>":
            vec[-2] = 1.
            vec_map["<eos>"] = vec
        elif word == "<comma>":
            vec[-3] = 1.
            vec_map["<comma>"] = vec
        else:
            vec[:model.vector_size] = model[word]
            vec_map[word] = vec

    return vec_map


def _init_word2vec(load=False):
    global model
    # model = word2vec.Word2Vec.load_word2vec_format('../models/GoogleNews-vectors-negative300.bin', binary=True)
    # model = word2vec.Word2Vec.load_word2vec_format('/dev/shm/GoogleNews-vectors-negative300.bin', binary=True)
    model = word2vec.Word2Vec.load('../models/norm.bin')
    # model.init_sims(replace=True)


def build_vocab(filename):
    data = _read_words(filename)

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    print words

    return _generate_vector_map(words)


def _split(data):
    xs = []
    ys = []
    for (x, y) in data:
        xs.append(x)
        ys.append(y)

    return xs, ys


# Returns (Nx5, Nx[D]) train, validation and test data
def get_raw_data(xs, ys, train_frac=0.7, val_frac=0.2, test_frac=0.1):
    # Init word2vec
    _init_word2vec()

    # word -> w2v vector  mapping
    vocab = build_vocab(ys)

    # [sentence]
    sentences = _read_sentences(ys)
    ret_data = []

    # [[x_ij]]
    xs = list(csv.reader(open(xs, "r"), delimiter=' '))
    for n in range(len(sentences)):
        words = filter(None, sentences[n].split())
        cur_ys = []
        for word in words:
            cur_ys.append(vocab[word])
        x = np.asarray(map(float, xs[n]))
        ret_data.append((x, cur_ys))

    # Randomly shuffle data
    random.shuffle(ret_data)

    # Compute split points
    tr_end = int(train_frac * len(ret_data))
    val_end = tr_end + int(val_frac * len(ret_data))

    return _split(ret_data[:tr_end]), _split(ret_data[tr_end:val_end]), _split(ret_data[val_end:]), vocab


# Seqs : Nx5, labels : Nxlist(D) (list of len <= maxlen)
# Output : X - TxNx3 float16, mask - TxN bool, Y - TxNxD
def prepare_data(seqs, labels, maxlen=40, x_dim=3):
    """Create the matrices from the datasets.

    This pad each sequence to the same length: the length of the
    longest sequence or maxlen.

    if maxlen is set, we will cut all sequence to this maximum
    length.

    This swap the axis!
    """
    # Find out maxlen if maxlen=None
    if maxlen is None:
        maxlen = 0
        for o_seq in labels:
            maxlen = np.max([len(o_seq), maxlen])

    # Trim all output seqs to have only maxlen steps
    Iseqs = []
    Oseqs = []
    for i_seq, o_seq in zip(seqs, labels):
        if len(o_seq) <= maxlen:
            Iseqs.append(i_seq)
            Oseqs.append(o_seq)
    seqs = Iseqs
    labels = Oseqs

    # Pad and compute masks
    ret_X = np.zeros((maxlen, len(seqs), x_dim))
    mask_X = np.zeros((maxlen, len(seqs)))
    # model.vector_size + 3 -> 3 extra for (<COMMA>, <EOS>, <EOP>) at idx : <300:302>
    labels_X = np.zeros((maxlen, len(seqs), model.vector_size + 3))
    for k in range(len(seqs)):
        t_dim = len(labels[k])
        mask_X[:t_dim, k] = 1
        ret_X[:t_dim, k] = np.asarray(seqs[k])
        labels_X[:t_dim, k] = np.asarray(labels[k])

    return ret_X, mask_X, labels_X


if __name__ == "__main__":
    train, val, test = get_raw_data("../data/xs1000.txt",
                                    "../data/targets1000.txt")
    c = _read_sentences("../data/targets1000.txt")
    print np.array(train[0]);
    # print val;
    # print test;

    print(build_vocab("../data/targets1000.txt"))
