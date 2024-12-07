#!/usr/bin/env python3

"""
Assignment 3 starter code!

Based largely on:
    https://github.com/keras-team/keras-io/blob/master/examples/nlp/text_classification_from_scratch.py
"""

import os
import sys
os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
import tensorflow as tf
import numpy as np
from keras import layers

## Loading the "20newsgroups" dataset.
def load_textfiles():
    RANDOM_SEED = 1337

    batch_size = 32
    raw_train_ds = keras.utils.text_dataset_from_directory(
        "20_newsgroups",
        batch_size=batch_size,
        validation_split=0.2,
        subset="training",
        seed=RANDOM_SEED,
    )
    raw_val_ds = keras.utils.text_dataset_from_directory(
        "20_newsgroups",
        batch_size=batch_size,
        validation_split=0.2,
        subset="validation",
        seed=RANDOM_SEED,
    )

    raw_test_ds = keras.utils.text_dataset_from_directory(
        "20_newsgroups_test",
        batch_size=batch_size,
        seed=RANDOM_SEED,
    )

    print(f"Number of batches in raw_train_ds: {raw_train_ds.cardinality()}")
    print(f"Number of batches in raw_val_ds: {raw_val_ds.cardinality()}")
    print(f"Number of batches in raw_test_ds: {raw_test_ds.cardinality()}")
    return raw_train_ds, raw_val_ds, raw_test_ds


# Model constants.
max_features = 20 * 1000
embedding_dim = 128
sequence_length = 500

vectorize_layer = keras.layers.TextVectorization(
    standardize="lower_and_strip_punctuation",
    max_tokens=max_features,
    output_mode="int",
    output_sequence_length=sequence_length,
)


def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label


def build_model():
    """
    Build an RNN-based model to embed the input text as a sequence of vectors,
    transform the sequence of embeddings into a single vector using a Bidirectional SimpleRNN,
    and apply a feed-forward layer on that vector to obtain the label.
    """
    # A integer input for vocab indices (sequence of word indices).
    inputs = keras.Input(shape=(None,), dtype="int64")

    # Step 1: Embed the input text as a sequence of vectors
    x = layers.Embedding(max_features, embedding_dim)(inputs)  # embedding_dim = 16
    x = layers.Dropout(0.1)(x)  # Dropout with rate 0.5 to prevent overfitting

    # Step 2: Transform the sequence of embeddings into a single vector using RNN
    x = layers.Bidirectional(layers.SimpleRNN(64, activation="tanh"))(x)  # hidden_dim = 64, nonlinearity = tanh

    # Step 3: Apply a feed-forward layer on that vector to obtain the label
    x = layers.Dropout(0.5)(x)  # Dropout after RNN layer
    predictions = layers.Dense(20, activation="softmax", name="predictions")(x)  # 20 output classes (Newsgroups)

    # Build the model
    model = keras.Model(inputs, predictions)

    # Compile the model with sparse categorical crossentropy and Adam optimizer
    model.compile(
        loss="sparse_categorical_crossentropy", 
        optimizer=keras.optimizers.Adam(learning_rate=0.001),  # learning_rate = 0.001
        metrics=["accuracy"]
    )

    return model

def build_LSTM():
    """
    Build an LSTM-based model to embed the input text as a sequence of vectors,
    transform the sequence of embeddings into a single vector and apply a feed-forward layer on that vector to obtain the label.
    """
    # A integer input for vocab indices (sequence of word indices).
    inputs = keras.Input(shape=(None,), dtype="int64")

    # Step 1: Embed the input text as a sequence of vectors
    x = layers.Embedding(max_features, embedding_dim)(inputs)  # embedding_dim = 16
    x = layers.Dropout(0.1)(x)  # Dropout with rate 0.5 to prevent overfitting

    # Step 2: Transform the sequence of embeddings into a single vector using LSTM
    x = layers.Bidirectional(layers.LSTM(32, return_sequences = False))(x)  # hidden_dim = 32
    x = layers.Flatten()(x)
    x = layers.Dense(32, activation = "tanh")(x)
    # Step 3: Apply a feed-forward layer on that vector to obtain the label
    x = layers.Dropout(0.5)(x)  # Dropout after FC layer
    predictions = layers.Dense(20, activation="softmax", name="predictions")(x)  # 20 output classes (Newsgroups)

    # Build the model
    model = keras.Model(inputs, predictions)

    # Compile the model with sparse categorical crossentropy and Adam optimizer
    model.compile(
        loss="sparse_categorical_crossentropy", 
        optimizer=keras.optimizers.Adam(learning_rate=0.001),  # learning_rate = 0.001
        metrics=["accuracy"]
    )

    return model

def main():
    if len(sys.argv) != 3:
        print("Usage: python assignment3.py <param1> <param2>")
        return
    modelIn = sys.argv[1]
    epochIn = sys.argv[2]
    if not epochIn.isdigit():
        print("Incorrect usage, Epochs must be a number!")
    if (modelIn != "LSTM") and (modelIn != "RNN"):
        print("Incorrect usage, only support for model 'LSTM', and 'RNN'") 
    raw_train_ds, raw_val_ds, raw_test_ds = load_textfiles()

    # set the vocabulary!
    text_ds = raw_train_ds.map(lambda x, y: x)
    vectorize_layer.adapt(text_ds)

    # Vectorize the data.
    train_ds = raw_train_ds.map(vectorize_text)
    val_ds = raw_val_ds.map(vectorize_text)
    test_ds = raw_test_ds.map(vectorize_text)

    # Do async prefetching / buffering of the data for best performance on GPU.
    train_ds = train_ds.cache().prefetch(buffer_size=10)
    val_ds = val_ds.cache().prefetch(buffer_size=10)
    test_ds = test_ds.cache().prefetch(buffer_size=10)
    if modelIn == "LSTM":
        model = build_LSTM() # modify based on model
    else:
        model = build_model()
    epochs = int(epochIn) # modify based on model
    # Actually perform training.
    print(f"Training model: {modelIn} for {epochs} epochs")
    model.fit(train_ds, validation_data=val_ds, epochs=epochs)

    """
    ## Evaluate the model on the test set or validation set.
    """
    model.evaluate(test_ds)


if __name__ == "__main__":
    main()
