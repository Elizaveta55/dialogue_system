import nltk
import itertools
import pickle
import numpy as np

EN_WHITELIST = '0123456789abcdefghijklmnopqrstuvwxyzёйцукенгшщзхъфывапролджэячсмитьбю ' # space is included in whitelist
EN_BLACKLIST = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\''

limit = {
        'maxq' : 20,
        'minq' : 0,
        'maxa' : 20,
        'mina' : 3
        }

UNK = 'unk'
VOCAB_SIZE = 6000

def ddefault():
    return 1

def read_lines(dataset):
    return dataset[:-1]


def filter_line(line, whitelist):
    return ''.join([ ch for ch in line if ch in whitelist ])


def index_(tokenized_sentences, vocab_size):
    freq_dist = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    vocab = freq_dist.most_common(vocab_size)
    index2word = ['_'] + [UNK] + [ x[0] for x in vocab ]
    word2index = dict([(w,i) for i,w in enumerate(index2word)] )
    return index2word, word2index, freq_dist


def filter_data(sequences):
    filtered_q, filtered_a = [], []
    raw_data_len = len(sequences)//2

    for i in range(0, len(sequences), 2):
        qlen, alen = len(sequences[i].split(' ')), len(sequences[i+1].split(' '))
        if qlen >= limit['minq'] and qlen <= limit['maxq']:
            if alen >= limit['mina'] and alen <= limit['maxa']:
                filtered_q.append(sequences[i])
                filtered_a.append(sequences[i+1])

    filt_data_len = len(filtered_q)
    filtered = int((raw_data_len - filt_data_len)*100/raw_data_len)

    return filtered_q, filtered_a


def zero_pad(qtokenized, atokenized, w2idx):
    data_len = len(qtokenized)

    idx_q = np.zeros([data_len, limit['maxq']], dtype=np.int32)
    idx_a = np.zeros([data_len, limit['maxa']], dtype=np.int32)

    for i in range(data_len):
        q_indices = pad_seq(qtokenized[i], w2idx, limit['maxq'])
        a_indices = pad_seq(atokenized[i], w2idx, limit['maxa'])
        idx_q[i] = np.array(q_indices)
        idx_a[i] = np.array(a_indices)

    return idx_q, idx_a


def pad_seq(seq, lookup, maxlen):
    indices = []
    for word in seq:
        if word in lookup:
            indices.append(lookup[word])
        else:
            indices.append(lookup[UNK])
    return indices + [0]*(maxlen - len(seq))


def process_data():

    lines = read_lines(dataset)
    lines = [ line.lower() for line in lines ]
    lines = [ filter_line(line, EN_WHITELIST) for line in lines ]
    qlines, alines = filter_data(lines)

    qtokenized = [ wordlist.split(' ') for wordlist in qlines ]
    atokenized = [ wordlist.split(' ') for wordlist in alines ]

    idx2w, w2idx, freq_dist = index_( qtokenized + atokenized, vocab_size=VOCAB_SIZE)

    idx_q, idx_a = zero_pad(qtokenized, atokenized, w2idx)

    np.save('idx_q3.npy', idx_q)
    np.save('idx_a3.npy', idx_a)

    metadata = {
            'w2idx' : w2idx,
            'idx2w' : idx2w,
            'limit' : limit,
            'freq_dist' : freq_dist
                }
    with open('metadata3.pkl', 'wb') as f:
        pickle.dump(metadata, f)


with open('ru_5.txt', encoding='utf8') as f:
    dataset = f.readlines()
process_data()