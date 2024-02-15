# import pandas as pd
# import numpy as np
# import string
# from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences
# from sklearn.model_selection import train_test_split
# from keras.models import Sequential, load_model
# from keras.layers import LSTM, Dense, Embedding, RepeatVector, Dropout
# from keras.optimizers import Adam
# from keras.callbacks import ModelCheckpoint
# from keras.layers import Bidirectional
# from keras.regularizers import l2
# from keras.models import load_model


# ### STEP 1 - PREPROCESS THE DATASET ###
# # Load the dataset
# df = pd.read_csv('processed_dataset_50k.csv')

# # Remove punctuation and convert text to lowercase
# def clean_text(text):
#     if isinstance(text, str):
#         text = text.translate(str.maketrans('', '', string.punctuation))
#         return text.lower()
#     else:
#         return text

# df['English'] = df['English'].apply(clean_text)
# df['Portuguese'] = df['Portuguese'].apply(clean_text)
# df = df.fillna('') # Fill any empty spaces with empty strings

# # Tokenize sentences
# eng_tokenizer = Tokenizer()
# pt_tokenizer = Tokenizer()

# eng_tokenizer.fit_on_texts(df['English'])
# pt_tokenizer.fit_on_texts(df['Portuguese'])

# # Convert sentences to sequences of integers
# df['eng_seq'] = eng_tokenizer.texts_to_sequences(df['English'])
# df['pt_seq'] = pt_tokenizer.texts_to_sequences(df['Portuguese'])

# # Find the maximum length of sequences in both languages for padding
# max_length_eng = max(df['eng_seq'].apply(len))
# max_length_pt = max(df['pt_seq'].apply(len))
# # max_length_eng = 12
# # max_length_pt = 12

# # Pad the sequences
# df['eng_seq_padded'] = pad_sequences(df['eng_seq'], maxlen=max_length_eng, padding='post').tolist()
# df['pt_seq_padded'] = pad_sequences(df['pt_seq'], maxlen=max_length_pt, padding='post').tolist()

# # Split the dataset into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(df['eng_seq_padded'], df['pt_seq_padded'], test_size=0.2, random_state=42)

# # Convert the lists to numpy arrays for Keras
# X_train = np.array(X_train.tolist())
# X_test = np.array(X_test.tolist())
# y_train = np.array(y_train.tolist())
# y_test = np.array(y_test.tolist())


# # ## STEP 2 - BUILD AND DEFINE THE MODEL ###
# # embeding_dimentions of layers
# # hideden units of layers
# # 12_lambda = 0.001 or 0.0001 l2 regularization
# # dropout_rate = 0.4 high help  reduce overfitting or low not overfitting
# # custom leraning rate = 0.001 | too hill will curve quicker too low will curve slower, for example 0.001 is a common starting learning rate
# # tensorflow slicing
# # split more data for training and use diferent treath for training

# eng_vocab_size = len(eng_tokenizer.word_index) + 1
# pt_vocab_size = len(pt_tokenizer.word_index) + 1

# # Configuration
# embedding_dim = 128
# lstm_dim = 128
# l2_lambda = 0.001  # L2 regularization factor
# dropout_rate = 0.4  # Dropout rate
# custom_lr = 0.001  # Custom learning rate

# # # Define the model
# # model = Sequential()
# # model.add(Embedding(eng_vocab_size, embedding_dim, input_length=max_length_eng, mask_zero=True))
# # model.add(LSTM(lstm_dim))
# # model.add(RepeatVector(max_length_pt))
# # model.add(LSTM(lstm_dim, return_sequences=True))
# # model.add(Dense(pt_vocab_size, activation='softmax'))

# # # Define the model
# # model = Sequential()
# # model.add(Embedding(eng_vocab_size, embedding_dim, input_length=max_length_eng, mask_zero=True))
# # model.add(Bidirectional(LSTM(lstm_dim)))
# # model.add(RepeatVector(max_length_pt))
# # model.add(Bidirectional(LSTM(lstm_dim, return_sequences=True)))
# # model.add(Dense(pt_vocab_size, activation='softmax'))

# # Define the model with L2 regularization and dropout
# model = Sequential()
# model.add(Embedding(eng_vocab_size, embedding_dim, input_length=max_length_eng, mask_zero=True))
# model.add(LSTM(lstm_dim, kernel_regularizer=l2(l2_lambda)))
# model.add(Dropout(dropout_rate))
# model.add(RepeatVector(max_length_pt))
# model.add(LSTM(lstm_dim, return_sequences=True, kernel_regularizer=l2(l2_lambda)))
# model.add(Dropout(dropout_rate))
# model.add(Dense(pt_vocab_size, activation='softmax'))

# # # Compile the model
# # model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# # Compile the model with custom learning rate
# model.compile(optimizer=Adam(learning_rate=custom_lr), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# # Summary of the model
# model.summary()


# ### STEP 3 - TRAIN THE MODEL ###
# checkpoint = ModelCheckpoint('model.h5', save_best_only=True, monitor='val_loss', mode='min')

# # Fit the model
# history = model.fit(X_train, np.expand_dims(y_train, -1), validation_data=(X_test, np.expand_dims(y_test, -1)),
#                     epochs=2, batch_size=64, callbacks=[checkpoint]) # try 32, 64, 128, 256, 512, 1024

# # Print training history
# print("Training History:")
# for epoch, logs in enumerate(history.history['loss'], start=1):
#     print(f"Epoch {epoch}/{len(history.history['loss'])}, "
#         f"Loss: {logs:.4f}, "
#         f"Accuracy: {history.history['accuracy'][epoch-1]:.4f}, "
#         f"Validation Loss: {history.history['val_loss'][epoch-1]:.4f}, "
#         f"Validation Accuracy: {history.history['val_accuracy'][epoch-1]:.4f}")


# ### STEP 4 - EVALUATING THE MODEL ###
# # Evaluate the model on the test set
# test_loss, test_accuracy = model.evaluate(X_test, np.expand_dims(y_test, -1))
# print(f"Test Loss: {test_loss:.4f}")
# print(f"Test Accuracy: {test_accuracy:.4f}")


# ### SAVE MODEL ###

# model.save('model.keras')
# model = load_model('model.keras')

# # Load the pre-trained model
# model = load_model('model.h5')

# # Function to preprocess and translate the input text
# def translate(input_text):
#     # Preprocess the input text
#     input_text = clean_text(input_text)
#     input_seq = eng_tokenizer.texts_to_sequences([input_text])
#     input_seq_padded = pad_sequences(input_seq, maxlen=max_length_eng, padding='post')
    
#     # Get the model prediction (translated sequence)
#     prediction = model.predict(np.array(input_seq_padded))
    
#     # Convert the sequence of integers to words
#     translated_text = ''
#     for i in prediction[0]:
#         word = ''
#         for key, value in pt_tokenizer.word_index.items():
#             if value == np.argmax(i):
#                 word = key
#                 break
#         if word != '':
#             translated_text += word + ' '

#     return translated_text.strip()

# # User Interface Loop
# while True:
#     user_input = input("Enter English text to translate (or type 'exit' to stop): ")
#     if user_input.lower() == 'exit':
#         break
#     translation = translate(user_input)
#     print("Translated Text:", translation)











import pandas as pd
import numpy as np
import string
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding, RepeatVector, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.regularizers import l2
from keras.models import load_model

### STEP 1 - PREPROCESS THE DATASET ###
# Load the dataset
df = pd.read_csv('processed_dataset_50k.csv')

# Remove punctuation and convert text to lowercase
def clean_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.lower()

df['English'] = df['English'].apply(clean_text)
df['Portuguese'] = df['Portuguese'].apply(clean_text)
df = df.fillna('') # Fill any empty spaces with empty strings

# Tokenize sentences
eng_tokenizer = Tokenizer()
pt_tokenizer = Tokenizer()

eng_tokenizer.fit_on_texts(df['English'])
pt_tokenizer.fit_on_texts(df['Portuguese'])

# Convert sentences to sequences of integers
df['eng_seq'] = eng_tokenizer.texts_to_sequences(df['English'])
df['pt_seq'] = pt_tokenizer.texts_to_sequences(df['Portuguese'])

# Find the maximum length of sequences in both languages for padding
max_length_eng = max(df['eng_seq'].apply(len))
max_length_pt = max(df['pt_seq'].apply(len))

# Pad the sequences
df['eng_seq_padded'] = pad_sequences(df['eng_seq'], maxlen=max_length_eng, padding='post').tolist()
df['pt_seq_padded'] = pad_sequences(df['pt_seq'], maxlen=max_length_pt, padding='post').tolist()

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(df['eng_seq_padded'], df['pt_seq_padded'], test_size=0.2, random_state=42)

# Convert the lists to numpy arrays for Keras
X_train = np.array(X_train.tolist())
X_test = np.array(X_test.tolist())
y_train = np.array(y_train.tolist())
y_test = np.array(y_test.tolist())

### STEP 2 - BUILD AND DEFINE THE MODEL ###
eng_vocab_size = len(eng_tokenizer.word_index) + 1
pt_vocab_size = len(pt_tokenizer.word_index) + 1

# Configuration
embedding_dim = 64
lstm_dim = 64
l2_lambda = 0.001  # L2 regularization factor
dropout_rate = 0.4  # Dropout rate
custom_lr = 0.001  # Custom learning rate

# Define the model with L2 regularization and dropout
model = Sequential()
model.add(Embedding(eng_vocab_size, embedding_dim, input_length=max_length_eng, mask_zero=True))
model.add(LSTM(lstm_dim, kernel_regularizer=l2(l2_lambda)))
model.add(Dropout(dropout_rate))
model.add(RepeatVector(max_length_pt))
model.add(LSTM(lstm_dim, return_sequences=True, kernel_regularizer=l2(l2_lambda)))
model.add(Dropout(dropout_rate))
model.add(Dense(pt_vocab_size, activation='softmax'))

# Compile the model with custom learning rate
model.compile(optimizer=Adam(learning_rate=custom_lr), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Summary of the model
model.summary()

### STEP 3 - TRAIN THE MODEL ###
checkpoint = ModelCheckpoint('model.h5', save_best_only=True, monitor='val_loss', mode='min')
early_stopping = EarlyStopping(monitor='val_loss', patience=5)

# Fit the model
history = model.fit(X_train, np.expand_dims(y_train, -1), validation_data=(X_test, np.expand_dims(y_test, -1)),
                    epochs=30, batch_size=64, callbacks=[checkpoint, early_stopping])

# Print training history
print("Training History:")
for epoch, logs in enumerate(history.history['loss'], start=1):
    print(f"Epoch {epoch}/{len(history.history['loss'])}, "
        f"Loss: {logs:.4f}, "
        f"Accuracy: {history.history['accuracy'][epoch-1]:.4f}, "
        f"Validation Loss: {history.history['val_loss'][epoch-1]:.4f}, "
        f"Validation Accuracy: {history.history['val_accuracy'][epoch-1]:.4f}")

### STEP 4 - EVALUATING THE MODEL ###
# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, np.expand_dims(y_test, -1))
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

### SAVE MODEL ###
model.save('model.keras')
model = load_model('model.keras')

# Load the pre-trained model
model = load_model('model.keras')

# Function to preprocess and translate the input text
def translate(input_text):
    # Preprocess the input text
    input_text = clean_text(input_text)
    input_seq = eng_tokenizer.texts_to_sequences([input_text])
    input_seq_padded = pad_sequences(input_seq, maxlen=max_length_eng, padding='post')
    
    # Get the model prediction (translated sequence)
    prediction = model.predict(np.array(input_seq_padded))
    
    # Convert the sequence of integers to words
    translated_text = ''
    for i in prediction[0]:
        word = ''
        for key, value in pt_tokenizer.word_index.items():
            if value == np.argmax(i):
                word = key
                break
        if word != '':
            translated_text += word + ' '

    return translated_text.strip()

# User Interface Loop
while True:
    user_input = input("Enter English text to translate (or type 'exit' to stop): ")
    if user_input.lower() == 'exit':
        break
    translation = translate(user_input)
    print("Translated Text:", translation)
