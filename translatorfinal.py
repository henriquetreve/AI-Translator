import numpy as np
import pandas as pd
import string
from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras.optimizers import Adam
import tkinter as tk
from keras.regularizers import l2


# Load Dataset
data_path = 'English-Spanish-data.csv'
df = pd.read_csv(data_path)
df = df[:1000]  # Use only the first 10000 rows for simplicity, total rows of the dataset is 50k, my machine is not capable of the total. 

# Data Preprocessing
input_texts = []
target_texts = []
input_characters = set()
target_characters = set()

# Function for text normalization (lowercasing and removing punctuation)
def normalize_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

# Assuming 'english_text' and 'spanish_text' are column names
for index, row in df.iterrows():
    input_text = normalize_text(row['english_text'])
    target_text = '\t' + normalize_text(row['spanish_text']) + '\n'
    input_texts.append(input_text)
    target_texts.append(target_text)
    for char in input_text:
        input_characters.add(char)
    for char in target_text:
        target_characters.add(char)

input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
num_encoder_tokens = 6164
num_decoder_tokens = 11932
max_encoder_seq_length = 15
max_decoder_seq_length = 15

# Tokenization/Vectorization
input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])

# Encode input sentences
def encode_input_sentence(input_text, num_encoder_tokens, input_token_index, max_length):
    encoder_input_data = np.zeros((1, max_length, num_encoder_tokens), dtype='float32')
    for t, char in enumerate(input_text):
        encoder_input_data[0, t, input_token_index[char]] = 1.
    return encoder_input_data

# Prepare encoder and decoder data
encoder_input_data = np.zeros((len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype='float32')
decoder_input_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype='float32')
decoder_target_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype='float32')

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        if t < max_encoder_seq_length:
            encoder_input_data[i, t, input_token_index[char]] = 1.
    for t, char in enumerate(target_text):
        if t < max_decoder_seq_length:
            decoder_input_data[i, t, target_token_index[char]] = 1.
            if t > 0:
                # Ensure that the target index (t-1) is also within bounds
                decoder_target_data[i, t - 1, target_token_index[char]] = 1.


# Build the Seq2Seq Model
# Encoder
reg = l2(0.001)
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder_lstm = LSTM(50, return_state=True, kernel_regularizer=reg)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
encoder_states = [state_h, state_c]


# Decoder
decoder_inputs = Input(shape=(None, num_decoder_tokens))
decoder_lstm = LSTM(50, return_sequences=True, return_state=True, kernel_regularizer=reg)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)


# Define the model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Compile the model with accuracy
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
        batch_size=64,
        epochs=100,  # Increase epochs for better accuracy
        validation_split=0.2)

# Evaluation
# Calculate and print the loss and accuracy on the validation set
validation_loss, validation_accuracy = model.evaluate([encoder_input_data, decoder_input_data], decoder_target_data)
print("=====Results=====")
print(f"Validation Loss: {validation_loss:.4f}")  # Print with 4 decimal places
print(f"Validation Accuracy: {validation_accuracy:.2%}")  # Print as a percentage with 2 decimal places
print(f"Batch Size: {64}")  # Replace with the actual batch size used
print(f"Epochs: {100}")  # Replace with the actual number of epochs used
print(f"Learning Rate: {0.001}")  # Replace with the actual learning rate used
print("================")

# Inference setup
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(50,))
decoder_state_input_c = Input(shape=(50,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h_dec, state_c_dec]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

# Reverse-lookup token index to decode sequences back to something readable
reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())

# Decoding function
def decode_sequence(input_seq):
    # Encode the input as state vectors
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1 with the start character
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    target_seq[0, 0, target_token_index['\t']] = 1.

    # Sampling loop for a batch of sequences
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length or find stop character
        if sampled_char == '\n' or len(decoded_sentence) > max_decoder_seq_length:
            stop_condition = True

        # Update the target sequence (of length 1)
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence

# Tkinter UI setup
def translate():
    # Get the English text from the entry field and normalize it
    input_text = normalize_text(english_text_entry.get())

    # Truncate the input text if it's longer than the maximum encoder sequence length
    if len(input_text) > max_encoder_seq_length:
        input_text = input_text[:max_encoder_seq_length]

    # Encode the input text for prediction
    input_seq = encode_input_sentence(input_text, num_encoder_tokens, input_token_index, max_encoder_seq_length)

    # Get the decoded Spanish sentence
    decoded_spanish_sentence = decode_sequence(input_seq)
    spanish_text_entry.delete(1.0, tk.END)
    spanish_text_entry.insert(tk.END, decoded_spanish_sentence)

    # Optionally, translate back to English (requires additional implementation)

root = tk.Tk()
root.title("Language Translator")

# English text entry
tk.Label(root, text="English Text:").pack()
english_text_entry = tk.Entry(root)
english_text_entry.pack()

# Button to perform translation
translate_button = tk.Button(root, text="Translate", command=translate)
translate_button.pack()

# Spanish text display
tk.Label(root, text="Spanish Translation:").pack()
spanish_text_entry = tk.Text(root, height=4, width=50)
spanish_text_entry.pack()

# Optionally, display for back translation to English

root.mainloop()

