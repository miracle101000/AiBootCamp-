import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt

# Load CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Normalize pixel values to [0,1]
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

#One-hot encoding target labels
y_train  = to_categorical(y_train, 10)
y_test  = to_categorical(y_test, 10)

print(f"Training data shape: {X_train.shape}, {y_train.shape}")
print(f"Testing data shape: {X_test.shape}, {y_test.shape}")

# Define the baseline model
model =  Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Display the model summary
model.summary()

# Train the baseline model
history = model.fit(
    X_train, y_train,
    validating_split=.2,
    epochs=10,
    batch_size=64,
    verbose=1
)

# Evaluate the baseline model
loss, accuracy =model.evaluate(X_test, y_test, verbose=0)
print(f"Baseline Model Test Accuracy: {accuracy:.4f}")


# Define an improved model
improved_model =  Sequential([
    Conv2D(64, (5, 5), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(128, (5, 5), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])


# Compile the improved model with a learning rate scheduler
optimizer = tf.keras.optimizers.Adam(learning_rate=.001)
improved_model.compile()


# Train the baseline model
improved_history = improved_model.fit(
    X_train, y_train,
    validating_split=.2,
    epochs=10,
    batch_size=64,
    verbose=1
)

# Evaluate the baseline model
improved_model_loss, improved_model_accuracy =improved_model.evaluate(X_test, y_test, verbose=0)
print(f"Improved Model Test Accuracy: {improved_model_accuracy:.4f}")


# Plot training and validate accuracy
plt.plot(improved_history.history['accuracy'], label="Training Accuracy")
plt.plot(improved_history.history['val_accuracy'], label="Validation Accuracy")
plt.title('Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# Plot training and validate accuracy
plt.plot(improved_history.history['loss'], label="Training Loss")
plt.plot(improved_history.history['val_accuracy'], label="Validation Loss")
plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()