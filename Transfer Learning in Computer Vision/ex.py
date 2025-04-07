import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load pre-trained ResNet50 without the top (fully connected) layers
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# Freeze all layers initially
for layer in base_model.layers:
    layer.trainable = False

# Unfreeze only the last few layers (layer4 equivalent in PyTorch)
for layer in base_model.layers[-10:]:  # Adjust number of layers as needed
    layer.trainable = True  

# Add new classifier layers
x = GlobalAveragePooling2D()(base_model.output)  # Converts feature maps into a vector
x = Dense(256, activation="relu")(x)
x = Dropout(0.4)(x)
output = Dense(5, activation="softmax")(x)  # 5 classes

# Create final model
model = Model(inputs=base_model.input, outputs=output)

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# Data augmentation & preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # 80-20 train-validation split
)

train_generator = train_datagen.flow_from_directory(
    "PATH_TO_FOLDER_TRAIN",
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    subset="training"
)

val_generator = train_datagen.flow_from_directory(
    "PATH_TO_FOLDER_TRAIN",
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    subset="validation"
)

# Train the model
model.fit(train_generator, validation_data=val_generator, epochs=10)

# Evaluate the model
loss, accuracy = model.evaluate(val_generator)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")
