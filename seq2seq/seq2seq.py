import tensorflow as tf
import tensorlayer as tl
import numpy as np
from tensorlayer.cost import cross_entropy_seq, cross_entropy_seq_with_mask
from tqdm import tqdm
from sklearn.utils import shuffle
from tensorlayer.models.seq2seq import Seq2seq
from tensorlayer.models.seq2seq_with_attention import Seq2seqLuongAttention
import os
import pickle
from sklearn.model_selection import train_test_split


def initial_setup():
    idx_q, idx_a = np.load('data/idx_q.npy'), np.load('data/idx_a.npy')
    with open('data/metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    x, testX, y, testY = train_test_split(idx_q, idx_a, test_size=0.2, train_size=0.8)
    trainX, validX, trainY, validY = train_test_split(x, y, test_size=0.2, train_size=0.8)
    trainX = tl.prepro.remove_pad_sequences(trainX.tolist())
    trainY = tl.prepro.remove_pad_sequences(trainY.tolist())
    testX = tl.prepro.remove_pad_sequences(testX.tolist())
    testY = tl.prepro.remove_pad_sequences(testY.tolist())
    validX = tl.prepro.remove_pad_sequences(validX.tolist())
    validY = tl.prepro.remove_pad_sequences(validY.tolist())
    return metadata, trainX, trainY, testX, testY, validX, validY


metadata, trainX, trainY, testX, testY, validX, validY = initial_setup()

src_len = len(trainX)
tgt_len = len(trainY)

assert src_len == tgt_len

batch_size = 32
n_step = src_len // batch_size
src_vocab_size = len(metadata['idx2w'])
emb_dim = 1024

word2idx = metadata['w2idx']
idx2word = metadata['idx2w']

unk_id = word2idx['unk']
pad_id = word2idx['_']

start_id = src_vocab_size
end_id = src_vocab_size + 1

word2idx.update({'start_id': start_id})
word2idx.update({'end_id': end_id})
idx2word = idx2word + ['start_id', 'end_id']

src_vocab_size = tgt_vocab_size = src_vocab_size + 2

num_epochs = 50
vocabulary_size = src_vocab_size


def inference(seed, top_n):
    model_.eval()
    seed_id = [word2idx.get(w, unk_id) for w in seed.split(" ")]
    sentence_id = model_(inputs=[[seed_id]], seq_length=20, start_token=start_id, top_n=top_n)
    sentence = []
    for w_id in sentence_id[0]:
        w = idx2word[w_id]
        if w == 'end_id':
            break
        sentence = sentence + [w]
    return sentence


decoder_seq_length = 20
model_ = Seq2seq(
    decoder_seq_length=decoder_seq_length,
    cell_enc=tf.keras.layers.GRUCell,
    cell_dec=tf.keras.layers.GRUCell,
    n_layer=3,
    n_units=256,
    embedding_layer=tl.layers.Embedding(vocabulary_size=vocabulary_size, embedding_size=emb_dim),
)


# load_weights = tl.files.load_npz(name='model.npz')
# tl.files.assign_weights(load_weights, model_)

optimizer = tf.optimizers.Adam(learning_rate=0.001)
model_.train()

sent1 = "Я пошел гулять."
sent2 = "Я чувствую обиду и злость за свой поступок."
sent3 = "Снег автомату рознь, а собака скользкая."
sent4 = "Скажи мне что-либо приятное."
seeds = [sent1,sent2,sent3,sent4]

for epoch in range(num_epochs):
    model_.train()
    trainX, trainY = shuffle(trainX, trainY, random_state=0)
    total_loss, n_iter = 0, 0
    for X, Y in tqdm(tl.iterate.minibatches(inputs=trainX, targets=trainY, batch_size=batch_size, shuffle=False),
                     total=n_step, desc='Epoch[{}/{}]'.format(epoch + 1, num_epochs), leave=False):
        X = tl.prepro.pad_sequences(X)
        _target_seqs = tl.prepro.sequences_add_end_id(Y, end_id=end_id)
        _target_seqs = tl.prepro.pad_sequences(_target_seqs, maxlen=decoder_seq_length)
        _decode_seqs = tl.prepro.sequences_add_start_id(Y, start_id=start_id, remove_last=False)
        _decode_seqs = tl.prepro.pad_sequences(_decode_seqs, maxlen=decoder_seq_length)
        _target_mask = tl.prepro.sequences_get_mask(_target_seqs)

        with tf.GradientTape() as tape:
            output = model_(inputs=[X, _decode_seqs])

            output = tf.reshape(output, [-1, vocabulary_size])
            loss = cross_entropy_seq_with_mask(logits=output, target_seqs=_target_seqs, input_mask=_target_mask)

            grad = tape.gradient(loss, model_.all_weights)
            optimizer.apply_gradients(zip(grad, model_.all_weights))

        total_loss += loss
        n_iter += 1

    print('Epoch [{}/{}]: loss {:.4f}'.format(epoch + 1, num_epochs, total_loss / n_iter))

    for seed in seeds:
        print("Query >", seed)
        top_n = 3
        for i in range(top_n):
            sentence = inference(seed, top_n)
            print(" >", ' '.join(sentence))

    tl.files.save_npz(model_.all_weights, name='model.npz')
