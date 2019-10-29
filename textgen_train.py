import os
import time

import numpy as np
import tensorflow as tf

REVIEWS = 'reviews.txt'

def main():
    text = open(REVIEWS, 'rb').read().decode(encoding='utf-8')
    print(text[:250])
    vocab = sorted(set(text))
    print('{} unique characters'.format(len(vocab)))
    print(vocab)

    # Creating a mapping from unique characters to indices
    char2idx = {u: i for i, u in enumerate(vocab)}
    idx2char = np.array(vocab)
    text_as_int = np.array([char2idx[c] for c in text])
    print('{} ---- characters mapped to int ---- > {}'.format(repr(text[:13]), text_as_int[:13]))

    # The maximum length sentence we want for a single input in characters
    seq_length = 100
    examples_per_epoch = len(text) // (seq_length + 1)

    # Create training examples / targets
    char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

    for i in char_dataset.take(5):
        print(idx2char[i.numpy()])

    sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)

    for item in sequences.take(5):
        print(repr(''.join(idx2char[item.numpy()])))

    def split_input_target(chunk):
        input_text = chunk[:-1]
        target_text = chunk[1:]
        return input_text, target_text

    dataset = sequences.map(split_input_target)

    for input_example, target_example in dataset.take(1):
        print('Input data: ', repr(''.join(idx2char[input_example.numpy()])))
        print('Target data:', repr(''.join(idx2char[target_example.numpy()])))

    BATCH_SIZE = 64
    BUFFER_SIZE = 10000

    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

    print(dataset)


if __name__ == '__main__':
    main()
