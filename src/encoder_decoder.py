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
    data += ["NIL"];

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(2, len(words) + 2)))

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
        #x = map(float, xs[n])
        ret_data.append((filter(None,xs[n]), cur_ys))

    # Randomly shuffle data
    random.shuffle(ret_data)

    # Compute split points
    tr_end = int(train_frac * len(ret_data))
    val_end = tr_end + int(val_frac * len(ret_data))

    return _split(ret_data[:tr_end]), _split(ret_data[tr_end:val_end]), _split(ret_data[val_end:]), vocab

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
# Input - seqs: num_samples*3, labels: num_samples*[list]
# Return X:maxlen*num_samples*3, X_mask: max_len*num_samples, labels: maxlen*num_samples
def prepare_data(seqs, labels, maxlen=None, x_dim = 3, mapping=None, max_mapping=None):
    """Create the matrices from the datasets.

    This pad each sequence to the same length: the length of the
    longest sequence or maxlen.

    if maxlen is set, we will cut all sequence to this maximum
    length.

    This swap the axis!
    """
    assert mapping is not None;
    assert max_mapping is not None;
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

    new_seqs = [];
    memory = [];
    for seq in seqs:
        #print "In seq:";
        new_seq = [];
        mem_seq = [];
        for item_num in range( 0, len(seq) ):
            item = seq[item_num];
            # Create a memory item corresponding to every input item.
            memitem = [];
            if not is_number(item):
                #print "not a number: " + item;
                for k in range(0, max_mapping):
                    if mapping.has_key(item) and k == mapping[item]:
                        new_seq.append( 1 );
                        memitem.append( 1 );
                    else:
                        new_seq.append( 0 );
                        memitem.append( 0 );
                # Add to read-only memory matrix.
                mem_seq.append( memitem );
            elif item_num < 18:
                for k in range(0, max_mapping):
                    if mapping.has_key(item) and k == mapping[item]:
                        memitem.append( 1 );
                    else:
                        memitem.append( 0 );
                new_seq.append( item );
                # Add to read-only memory matrix.
                mem_seq.append( memitem );
            else:
                new_seq.append( item );

        new_seqs.append( new_seq );
        memory.append( mem_seq );

    # Pad and compute masks
    ret_X = np.zeros((maxlen, len(seqs), x_dim))
    mask_X = np.zeros((maxlen, len(seqs)))
    # start out with ones. Ones are the null characters.
    labels_X = np.ones((maxlen, len(seqs)))
    # NxMxW
    memory_X = np.array( memory );
    #print memory_X;
    for k in range(len(seqs)):
        mask_X[:len(labels[k]), k] = 1
        ret_X[:len(labels[k]), k] = np.asarray(new_seqs[k])
        labels_X[:len(labels[k]), k] = labels[k]

    return ret_X, mask_X, labels_X, memory_X


if __name__ == "__main__":
    #train, val, test, vocab = get_raw_data("../data/corpus/inputs_x.txt", "../data/corpus/targets_x.txt")
    #c = _read_sentences("../data/corpus/targets_x.txt")
    train, val, test, vocab = get_raw_data("../data/xs1000.txt", "../data/targets1000.txt")
    c = _read_sentences("../data/targets1000.txt")
    print np.array(train[0]);
    # print val;
    # print test;

    print(vocab)
    print( len( vocab ) )
