import json
import tensorflow as tf            
import model.custom_layers as custom_layers

def print_parameters(params):
    print('-Model Parameters-')
    for param in params:
        print(f'{param}: {params[param]}')


def create_model():
    with open('parameters.json') as f:
        params = json.load(f)
    
    print_parameters(params)

    inputs = tf.keras.layers.Input(shape=(params['sequence_length'],), dtype=tf.int32)

    padding_token = params['vocab_size'] - 1
    padding_mask = tf.keras.layers.Masking(mask_value=padding_token)
    x = padding_mask(inputs)
    
    embedding_layer = custom_layers.TokenAndPositionEmbedding(params['sequence_length'], params['vocab_size'], params['embedding_size'])
    x = embedding_layer(inputs)
    
    transformer_blocks = [custom_layers.TransformerBlock(params['embedding_size'], params['num_heads'], params['feed_forward_dim'], params['dropout'], params['sequence_length']) for i in range(params['num_blocks'])]
    for block in transformer_blocks:
        x = block(x)

    outputs = tf.keras.layers.Dense(params['vocab_size'])(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='Midinet')
    optimizer = tf.keras.optimizers.Adam(learning_rate=params['learning_rate'])
    loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])
    return model, optimizer, loss_function
