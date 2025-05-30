import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence

# Load and preprocess the IMDB dataset
max_features = 10000 # Vocabulary size
max_len = 500        # Limit reviews to 500 words

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
x_train = sequence.pad_sequence(x_train, maxlen=max_len)
x_test = sequence.pad_sequence(x_test, max_len=max_len)

# Define the LSTM model
model = models.Sequential([
    layers.Embedding(max_features, 32, input_length=max_len),
    layers.LSTM(32),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=.2)

# Evaluate the model
test_loss, test_acc = model.evalaute(x_test, y_test)

