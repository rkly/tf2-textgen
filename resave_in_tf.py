import tensorflow as tf


def main():
    def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                      batch_input_shape=[batch_size, None]),
            tf.keras.layers.GRU(rnn_units,
                                return_sequences=True,
                                stateful=True,
                                recurrent_initializer='glorot_uniform'),
            tf.keras.layers.Dense(vocab_size)
        ])
        return model

    text = open('reviews.txt', 'rb').read().decode(encoding='utf-8')
    alphabet = sorted(set(text))

    # Length of the vocabulary in chars
    vocab_size = len(alphabet)
    # The embedding dimension
    embedding_dim = 256
    # Number of RNN units
    rnn_units = 1024

    model = build_model(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        rnn_units=rnn_units,
        batch_size=1)
    model.load_weights("textgen.h5")
    model.build(tf.TensorShape([1, None]))
    model.summary()

    tf.saved_model.save(model, "saved_model")
    model.save('saved_model', save_format='tf')


if __name__ == "__main__":
    main()
