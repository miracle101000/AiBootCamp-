from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

# Data augmentation and preprocessing
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=.2,
    height_shift_range=.2,
    shear_range=.2,
    zoom_range=.2,
    horizontal_flip=True,
    validation_split=.2
)

train_data = datagen.flow_from_directory(
    "PATH TO DATASET",
    target_szie=(224, 224),
    batch_size=32,
    class_mode="binary",
    subset="training"
)

val_data = datagen.flow_from_directory(
    "PATH TO DATASET",
    target_szie=(224, 224),
    batch_size=32,
    class_mode="binary",
    subset="validation"
)

base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

for layer in base_model.layers:
    layer.trainable =  False

x =  GlobalAveragePooling2D()(base_model.output)
output =  Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10
)
    
val_loss, val_accuracy = model.evaluate()    
print(f"Validation Accuracy {val_accuracy}")