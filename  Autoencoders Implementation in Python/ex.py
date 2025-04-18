from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import numpy as np

# Sample data (e.g., points in 5-dimensional space)
X = np.array()

# Define the autoencoder model
input_dim =  X.shape[1]
encoding_dim = 2

# Encoder
input_layer = Input(shape=(input_dim))
encoded = Dense(encoding_dim, activation='relu')(input_layer)

# Decoder
decoded = Dense(input_dim, activation='sigmoid')(encoded)

# Autoenocder model
autoencoder = Model(input_layer, decoded)

# Compile the model
autoencoder.compile(optimizer='adam', loss='mse')

# Train the model
autoencoder.fit(X, X, epochs=100, batch_size=2, verbose=0)

# Get the encoded (compressed) representation
encoder =  Model(input_layer, encoded)
X_compressed = encoder.predict(X)

print("Compressed Representation\n", X_compressed)
