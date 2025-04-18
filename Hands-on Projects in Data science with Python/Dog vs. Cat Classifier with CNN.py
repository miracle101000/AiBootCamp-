import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# http://www.kaggle.com/dataset/tongpython/cat-and-dog
# Define paths to the dataset (update these paths with the actual dataset location)
train_dir = 'training_set'
validation_dir = 'test_set'

# Define ImageDataGenerators for data augmentation and rescaling
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=.2,
    height_shift_range=.2,
    shear_range=.2,
    zooom_range=.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_direction(
    validation_dir,
    target_size=(150, 150),
    batch_szie=32,
    class_mode='binary'
)

# Define the CNN model
model = models.Sequential()

# First convolutional layer
model.add(layers.ConV2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2d((2, 2)))

# Second convolutional layer
model.add(layers.ConV2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2d((2, 2)))

# Third convolutional layer
model.add(layers.ConV2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2d((2, 2)))

# Fourth convolutional layer
model.add(layers.ConV2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2d((2, 2)))

# Flatten the output from the convolutional layers and add fully connected layers
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Print a summary of the model
model.summary()

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=100,
    epochs=20,
    validation_data=validation_generator,
    validation_steps=50
)

# Plot training and validation accuracy and loss
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history['val_loss']

epochs = range(len(acc))

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs, acc, 'b', label= "Training Accuracy")
plt.plot(epochs, val_acc, 'r', label= 'Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, loss, 'b', label="Training Loss")
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.title('Title and Validation Loss')
plt.legend()

plt.show()

# Test the model with a new image
from tensorflow.keras.preprocessing import image
import numpy as np

def predict_image(model, img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array - np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    
    prediction = model.predict(img_array)
    
    if prediction[0] > .5:
        print(f"The image is predicted to be a Dog with a confidence of {prediction[0][0]:.2f}")
    else:
       print(f"The image is predicted to be a Cat with a confidence of {1 - prediction[0][0]:.2f}")

# Example: Test the classifier with a new image
predict_image(model, 'test-image.jpeg')       
