import tensorflow as tf
import numpy as np


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

    def generate_text(model, start_string):
        # Evaluation step (generating text using the learned model)
        # Number of characters to generate
        num_generate = 500

        # Converting our start string to numbers (vectorizing)
        input_eval = [char2idx[s] for s in start_string]
        input_eval = tf.expand_dims(input_eval, 0)

        # Empty string to store our results
        text_generated = []

        # Low temperatures results in more predictable text.
        # Higher temperatures results in more surprising text.
        # Experiment to find the best setting.
        temperature = 0.5

        model.reset_states()
        for i in range(num_generate):
            predictions = model(input_eval)
            # remove the batch dimension
            predictions = tf.squeeze(predictions, 0)

            # using a categorical distribution to predict the character returned by the model
            predictions = predictions / temperature
            predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

            # We pass the predicted character as the next input to the model
            # along with the previous hidden state
            input_eval = tf.expand_dims([predicted_id], 0)
            text_generated.append(idx2char[predicted_id])

        return start_string + ''.join(text_generated)

    text = open('reviews.txt', 'rb').read().decode(encoding='utf-8')
    alphabet = sorted(set(text))
    char2idx = {u: i for i, u in enumerate(alphabet)}
    idx2char = np.array(alphabet)

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

    start_string = input("Start string:")
    print(generate_text(model, start_string=start_string))


if __name__ == "__main__":
    main()
