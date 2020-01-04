import tensorflow as tf
import numpy as np
import time
import cv2
import os
import random

from model import *

# Training maximum rounds
num_epochs = 300

# Initialize learning rate
INITIAL_LEARNING_RATE = 1e-3
DECAY_STEPS = 2000
LEARNING_RATE_DECAY_FACTOR = 0.9  # The learning rate decay factor
MOMENTUM = 0.9

# Step interval for outputting string results
REPORT_STEPS = 5000

# Number of training sets
BATCH_SIZE = 50
TRAIN_SIZE = 7368
BATCHES = TRAIN_SIZE//BATCH_SIZE
test_num = 3

ti = 'train_in'         # Training set position
vi = 'valid_in'         # Validation set location
img_size = [94, 24]
tl = None
vl = None
num_channels = 3
label_len = 7

CHARS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
         'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z'
         ]

CHARS_DICT = {char:i for i, char in enumerate(CHARS)}

NUM_CHARS = len(CHARS)


def encode_label(s):
    label = np.zeros([len(s)])
    for i, c in enumerate(s):
        label[i] = CHARS_DICT[c]
    return label


# Read image and label to generate batch
class TextImageGenerator:
    def __init__(self, img_dir, label_file, batch_size, img_size, num_channels=3, label_len=7):
        self._img_dir = img_dir
        self._label_file = label_file
        self._batch_size = batch_size
        self._num_channels = num_channels
        self._label_len = label_len
        self._img_w, self._img_h = img_size

        self._num_examples = 0
        self._next_index = 0
        self._num_epoches = 0
        self.filenames = []
        self.labels = []

        self.labels = []
        fs = os.listdir(self._img_dir)
        for filename in fs:
            self.filenames.append(filename)
        for filename in self.filenames:
            print('Processing file:', filename)
            # import pdb; pdb.set_trace()
            label = encode_label(filename.split('.')[0].strip().replace(' ', ''))
            self.labels.append(np.float32(label))
            self._num_examples += 1
        # import pdb; pdb.set_trace()
        self.labels = np.array(self.labels)

    def next_batch(self):
        # Shuffle the data
        if self._next_index == 0:
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._filenames = [self.filenames[i] for i in perm]
            self._labels = self.labels[perm]

        batch_size = self._batch_size
        start = self._next_index
        end = self._next_index + batch_size
        if end > self._num_examples:
            self._next_index = 0
            start = self._next_index
            end = self._next_index + batch_size
            self._num_epoches += 1
        else:
            self._next_index = end
        images = np.zeros([batch_size, self._img_h, self._img_w, self._num_channels])
        # labels = np.zeros([batch_size, self._label_len])

        for j, i in enumerate(range(start, end)):
            fname = self._filenames[i]
            img = cv2.imread(os.path.join(self._img_dir, fname))
            if img is not None:
                img = cv2.resize(img, (self._img_w, self._img_h), interpolation=cv2.INTER_CUBIC)
                images[j, ...] = img
        images = np.transpose(images, axes=[0, 2, 1, 3])
        labels = self._labels[start:end, ...]
        targets = [np.asarray(i) for i in labels]
        sparse_labels = sparse_tuple_from(targets)
        # input_length = np.zeros([batch_size, 1])

        seq_len = np.ones(self._batch_size) * 24
        return images, sparse_labels, seq_len


def sparse_tuple_from(sequences, dtype=np.int32):
    """
    Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

    return indices, values, shape


def decode_sparse_tensor(sparse_tensor):
    decoded_indexes = list()
    current_i = 0
    current_seq = []
    for offset, i_and_index in enumerate(sparse_tensor[0]):
        i = i_and_index[0]
        if i != current_i:
            decoded_indexes.append(current_seq)
            current_i = i
            current_seq = list()
        current_seq.append(offset)
    decoded_indexes.append(current_seq)
    result = []
    for index in decoded_indexes:
        result.append(decode_a_seq(index, sparse_tensor))
    return result


def decode_a_seq(indexes, spars_tensor):
    decoded = []
    for m in indexes:
        str = CHARS[spars_tensor[1][m]]
        decoded.append(str)
    return decoded


def train():

    train_gen = TextImageGenerator(img_dir=ti,
                                   label_file=tl,
                                   batch_size=BATCH_SIZE,
                                   img_size=img_size,
                                   num_channels=num_channels,
                                   label_len=label_len)

    val_gen = TextImageGenerator(img_dir=vi,
                                 label_file=vl,
                                 batch_size=BATCH_SIZE,
                                 img_size=img_size,
                                 num_channels=num_channels,
                                 label_len=label_len)
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.compat.v1.train.exponential_decay(INITIAL_LEARNING_RATE,
                                               global_step,
                                               DECAY_STEPS,
                                               LEARNING_RATE_DECAY_FACTOR,
                                               staircase=True)

    logits, inputs, targets, seq_len = get_train_model(num_channels, label_len,BATCH_SIZE, img_size)
    logits = tf.transpose(logits, (1, 0, 2))
    # targets is a sparse matrix
    loss = tf.nn.ctc_loss(labels=targets, inputs=logits, sequence_length=seq_len)
    cost = tf.reduce_mean(loss)

    # optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=MOMENTUM).minimize(cost, global_step=global_step)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)

    # After dividing the blocks mentioned above, find the generic probability distribution of each block. The ctc_beam_search_decoder method is to find the largest K probability distributions at a time.
    # Another greedy strategy is to find only the one with the highest probability, which is the case of K = 1. Ctc_ greedy_decoder
    decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False)

    acc = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), targets))

    init = tf.global_variables_initializer()

    def report_accuracy(decoded_list, test_targets):
        original_list = decode_sparse_tensor(test_targets)
        detected_list = decode_sparse_tensor(decoded_list)
        true_numer = 0

        if len(original_list) != len(detected_list):
            print("len(original_list)", len(original_list), "len(detected_list)", len(detected_list),
                  " test and detect length desn't match")
            return
        print("T/F: original(length) <-------> detectcted(length)")
        for idx, number in enumerate(original_list):
            detect_number = detected_list[idx]
            hit = (number == detect_number)
            print(hit, number, "(", len(number), ") <-------> ", detect_number, "(", len(detect_number), ")")
            if hit:
                true_numer = true_numer + 1
        print("Test Accuracy:", true_numer * 1.0 / len(original_list))

    def do_report(val_gen,num):
        for i in range(num):
            test_inputs, test_targets, test_seq_len = val_gen.next_batch()
            test_feed = {inputs: test_inputs,
                        targets: test_targets,
                        seq_len: test_seq_len}
            st =time.time()
            dd= session.run(decoded[0], test_feed)
            tim = time.time() -st
            print('time:%s'%tim)
            report_accuracy(dd, test_targets)

    def test_report(testi,files):
        true_numer = 0
        num = files//BATCH_SIZE

        for i in range(num):
            test_inputs, test_targets, test_seq_len = val_gen.next_batch()
            test_feed = {inputs: test_inputs,
                        targets: test_targets,
                        seq_len: test_seq_len}
            dd = session.run([decoded[0]], test_feed)
            original_list = decode_sparse_tensor(test_targets)
            detected_list = decode_sparse_tensor(dd)
            for idx, number in enumerate(original_list):
                detect_number = detected_list[idx]
                hit = (number == detect_number)
                if hit:
                    true_numer = true_numer + 1
        print("Test Accuracy:", true_numer * 1.0 / files)


    def do_batch(train_gen,val_gen):
        train_inputs, train_targets, train_seq_len = train_gen.next_batch()

        feed = {inputs: train_inputs, targets: train_targets, seq_len: train_seq_len}

        b_loss, b_targets, b_logits, b_seq_len, b_cost, steps, _ = session.run(
            [loss, targets, logits, seq_len, cost, global_step, optimizer], feed)

        #print(b_cost, steps)
        if steps > 0 and steps % REPORT_STEPS == 0:
            do_report(val_gen,test_num)
            saver.save(session, "./model/LPRtf3.ckpt", global_step=steps)
        return b_cost, steps

    with tf.Session() as session:
        session.run(init)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
        for curr_epoch in range(num_epochs):
            print("Epoch.......", curr_epoch)
            train_cost = train_ler = 0
            for batch in range(BATCHES):
                start = time.time()
                c, steps = do_batch(train_gen,val_gen)
                train_cost += c * BATCH_SIZE
                seconds = time.time() - start
                #print("Step:", steps, ", batch seconds:", seconds)

            train_cost /= TRAIN_SIZE
            val_cs=0
            val_ls =0
            for i in range(test_num):
                train_inputs, train_targets, train_seq_len = val_gen.next_batch()
                val_feed = {inputs: train_inputs,
                            targets: train_targets,
                            seq_len: train_seq_len}

                val_cost, val_ler, lr, steps = session.run([cost, acc, learning_rate, global_step], feed_dict=val_feed)
                val_cs+=val_cost
                val_ls+=val_ler

        log = "Epoch {}/{}, steps = {}, train_cost = {:.3f}, train_ler = {:.3f}, val_cost = {:.3f}, val_ler = {:.3f}, time = {:.3f}s, learning_rate = {}"
        print(log.format(curr_epoch + 1, num_epochs, steps, train_cost, train_ler, val_cs/test_num, val_ls/test_num, time.time() - start, lr))


if __name__ == "__main__":
    train()
