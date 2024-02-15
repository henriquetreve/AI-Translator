import tensorflow as tf
from keras.layers import Input, LSTM, Embedding, Dense
from keras.models import Model
import numpy as np
import pandas as pd
from keras.regularizers import l2

# Load the dataset
df = pd.read_csv('reduced_dataset.csv')

# Extract English and Portuguese sentences
english_sentences = df['English'].tolist()
portuguese_sentences = df['Portuguese'].tolist()

# Basic tokenization and padding
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(english_sentences + portuguese_sentences)
input_sequences = tokenizer.texts_to_sequences(english_sentences)
target_sequences = tokenizer.texts_to_sequences(portuguese_sentences)

input_sequences = tf.keras.preprocessing.sequence.pad_sequences(input_sequences, padding='post')
target_sequences = tf.keras.preprocessing.sequence.pad_sequences(target_sequences, padding='post')


# Define model parameters
vocab_size = len(tokenizer.word_index) + 1
embedding_size = 256
lstm_units = 512
dropout_rate = 0.5  # Dropout rate (between 0 and 1)

# Encoder
encoder_inputs = Input(shape=(None,))
enc_emb = Embedding(vocab_size, embedding_size)(encoder_inputs)
encoder_lstm = LSTM(lstm_units, return_state=True, dropout=dropout_rate, recurrent_dropout=dropout_rate)
_, state_h, state_c = encoder_lstm(enc_emb)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = Input(shape=(None,))
dec_emb_layer = Embedding(vocab_size, embedding_size)
dec_emb = dec_emb_layer(decoder_inputs)
decoder_lstm = LSTM(lstm_units, return_sequences=True, return_state=True, dropout=dropout_rate, recurrent_dropout=dropout_rate)
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)
decoder_dense = Dense(vocab_size, activation='softmax', kernel_regularizer=l2(0.01))  # L2 regularization
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Compile the model
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Correcting decoder input data preparation
decoder_input_data = np.zeros_like(target_sequences)
decoder_input_data[:, :-1] = target_sequences[:,1:]  # Shifting target sequences

# Training the model
history = model.fit([input_sequences, decoder_input_data], target_sequences, batch_size=64, epochs=5, validation_split=0.2)

# Inference Models
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(lstm_units,))
decoder_state_input_c = Input(shape=(lstm_units,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

dec_emb2 = dec_emb_layer(decoder_inputs)
decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=decoder_states_inputs)
decoder_states2 = [state_h2, state_c2]
decoder_outputs2 = decoder_dense(decoder_outputs2)

decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs2] + decoder_states2)

# Function to generate sequences
def decode_sequence(input_seq):
    states_value = encoder_model.predict(input_seq, verbose=0)

    target_seq = np.zeros((1,1))
    target_seq[0, 0] = 1  # Starting with the first token in the tokenizer

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value, verbose=0)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        if sampled_token_index == 0:
            break
        sampled_word = tokenizer.index_word[sampled_token_index]
        decoded_sentence += ' ' + sampled_word

        # Check for repeated words and break if detected
        if len(decoded_sentence.split()) > 1 and decoded_sentence.split()[-1] == decoded_sentence.split()[-2]:
            break

        if (sampled_word == 'end' or len(decoded_sentence) > 50):
            stop_condition = True

        target_seq = np.zeros((1,1))
        target_seq[0, 0] = sampled_token_index

        states_value = [h, c]

    return decoded_sentence


# Test the model
for seq_index in range(len(input_sequences)):
    input_seq = input_sequences[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    # print('Input sentence:', english_sentences[seq_index])
    # print('Decoded sentence:', decoded_sentence)

# Function to process new input sentences
def process_new_input(sentence):
    sequence = tokenizer.texts_to_sequences([sentence])
    padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=input_sequences.shape[1], padding='post')
    return padded_sequence

# Function to translate a sentence
def translate(sentence):
    processed_input = process_new_input(sentence)
    translated_sentence = decode_sequence(processed_input)
    return translated_sentence.strip()

# User input
while True:
    user_input = input("Enter a sentence to translate (or 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    translated = translate(user_input)
    print("Translated sentence:", translated)

