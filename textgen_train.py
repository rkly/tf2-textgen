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

if __name__ == '__main__':
    main()
