from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LayerNormalization, Add, MultiHeadAttention

# Define a simplified Transformer Encoder Block
def transformer_encoder(input_dim, num_heads, ff_dim):
    inputs = Input(shape=(None, input_dim))
    
    # Multi-Head Self-Attention
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=input_dim)(inputs, inputs)
    attention_output = Add()([inputs, attention_output])  # Residual Connection
    attention_output = LayerNormalization()(attention_output)  # Normalize
    
    # Feed-Forward Neural Network
    ff_output = Dense(ff_dim, activation='relu')(attention_output)
    ff_output = Dense(input_dim)(ff_output)
    
    outputs = Add()([attention_output, ff_output])  # Fix: Pass as a list
    outputs = LayerNormalization()(outputs)  # Final normalization
    
    return Model(inputs, outputs)

# Create and visualize a sample Transformer Encoder Block
encoder_block = transformer_encoder(input_dim=64, num_heads=8, ff_dim=128)
plot_model(encoder_block, show_shapes=True, to_file='transformer_encoder.png')
