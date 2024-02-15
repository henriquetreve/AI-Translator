import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import string
import tensorflow as tf
from keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed
from keras.models import Model
from keras.optimizers import Adam

### 1 STEP - PRE-PROCESSING ###
# Load the dataset
df = pd.read_csv('processed_dataset_50k.csv')

# Function to remove punctuation
def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

# Text normalization (lowercasing)
df['English'] = df['English'].str.lower().apply(remove_punctuation)
df['Portuguese'] = df['Portuguese'].str.lower().apply(remove_punctuation)

# Define the maximum number of words to keep in the vocabulary
max_vocab_size = 10000  # You can adjust this number based on your dataset

# Tokenization
tokenizer = Tokenizer(num_words=max_vocab_size)
tokenizer.fit_on_texts(df['English'].tolist() + df['Portuguese'].tolist())

# Convert text to sequences of integers
eng_sequences = tokenizer.texts_to_sequences(df['English'])
por_sequences = tokenizer.texts_to_sequences(df['Portuguese'])

# Determine the maximum sequence length
max_sequence_length = max(max(len(seq) for seq in eng_sequences), max(len(seq) for seq in por_sequences))

# Padding sequences to have the same length
eng_padded = pad_sequences(eng_sequences, maxlen=max_sequence_length, padding='post')
por_padded = pad_sequences(por_sequences, maxlen=max_sequence_length, padding='post')

# Splitting the dataset (80% train, 20% test)
split_size = int(len(eng_padded) * 0.8)
train_eng, test_eng = eng_padded[:split_size], eng_padded[split_size:]
train_por, test_por = por_padded[:split_size], por_padded[split_size:]

### STEP 2 - Building the Seq2Seq Model ###
# Set parameters
embedding_dim = 256
units = 512
vocab_size = len(tokenizer.word_index) + 1

# Encoder
encoder_inputs = Input(shape=(max_sequence_length,))
enc_emb = Embedding(vocab_size, embedding_dim)(encoder_inputs)
encoder_lstm = LSTM(units, return_state=True, return_sequences=True)  # return_sequences=True is important here
encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = Input(shape=(max_sequence_length,))
dec_emb_layer = Embedding(vocab_size, embedding_dim)
dec_emb = dec_emb_layer(decoder_inputs)
decoder_lstm = LSTM(units, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)

# Attention Layer
attention_layer = tf.keras.layers.Attention()
attention_result = attention_layer([encoder_outputs, decoder_outputs])

# Concatenate attention output and decoder LSTM output
decoder_concat_input = Concatenate(axis=-1)([decoder_outputs, attention_result])

# Dense layer
decoder_dense = TimeDistributed(Dense(vocab_size, activation='softmax'))
decoder_outputs = decoder_dense(decoder_concat_input)

# Define the model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Compile the model
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Summary of the model
model.summary()

### STEP 3 - TRAINING THE MODEL ###
# Prepare decoder input data and output data
decoder_input_data = train_por[:, :-1]  # all timesteps except the last
decoder_output_data = train_por[:, 1:]  # all timesteps except the first

# Ensure decoder input and output data are padded to max_sequence_length
decoder_input_data = pad_sequences(decoder_input_data, maxlen=max_sequence_length, padding='post')
decoder_output_data = pad_sequences(decoder_output_data, maxlen=max_sequence_length, padding='post')

# Reshape decoder output data to be 3D as required for sparse categorical cross-entropy
decoder_output_data = np.expand_dims(decoder_output_data, -1)

# Train the model
history = model.fit([train_eng, decoder_input_data], decoder_output_data,
                    batch_size=64,  #256, 128, 64, 32 Batch size
                    epochs=2,  # Number of epochs
                    validation_split=0.2)

# Print training history
print("Training History:")
for epoch, logs in enumerate(history.history['loss'], start=1):
    print(f"Epoch {epoch}/{len(history.history['loss'])}, "
        f"Loss: {logs:.4f}, "
        f"Accuracy: {history.history['accuracy'][epoch-1]:.4f}, "
        f"Validation Loss: {history.history['val_loss'][epoch-1]:.4f}, "
        f"Validation Accuracy: {history.history['val_accuracy'][epoch-1]:.4f}")

### STEP 4 - EVALUATING THE MODEL ###
# Prepare decoder input data for test set
test_decoder_input_data = test_por[:, :-1]
test_decoder_output_data = test_por[:, 1:]
test_decoder_output_data = np.expand_dims(test_decoder_output_data, -1)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate([test_eng, test_decoder_input_data], test_decoder_output_data)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Translation function is not included as it requires modifications to the model architecture.
