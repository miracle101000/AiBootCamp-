import tensorflow as tf
from tensorflow.keras.applications import ResNet50

# Load a pre-trained ResNet50 model
model =  ResNet50(weights="imagenet")

# Display the models architecture
# model.summary()

# Access specific Layers
# for i, layers in enumerate(model.layers):
#     print(f"Layer {i}: {layers.name}, Trainable {layers.trainable}")
    

for layer in model.layers[:-10]:
    layer.trainable = False
    