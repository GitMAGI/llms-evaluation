from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Dense, Reshape, Flatten, BatchNormalization, Activation
from tensorflow.keras import backend as K
import numpy as np

"""
This function creates a convolutional autoencoder model.
Parameters:
    input_shape: tuple, shape of the input data (height, width, channels)
    conv_filters: list of integers, number of filters in each convolutional layer
    conv_kernels: list of integers, size of the kernel in each convolutional layer
    conv_strides: list of integers, strides in each convolutional layer
    conv_activations: list of strings, activations in each convolutional layer
    latent_space_dim: integer, dimension of the latent space
"""
def create_convolutional_autoencoder(
        input_shape,            
        conv_filters,           # dimension of the output space
        conv_kernels,           # size of the convolution window
        conv_strides,
        conv_activations,
        latent_space_dim
):
    if len(conv_filters) != len(conv_kernels) != len(conv_strides) != len(conv_activations):
        raise ValueError("The length of the following lists must be the same: conv_filters, conv_kernels, conv_strides, conv_activations")

    # Internal variables
    _num_conv_layers = len(conv_filters)
    _shape_before_bottleneck = None

    # Encoder
    # Create Input layer
    encoder_input = Input(shape=input_shape, name="encoder_input")
    # Create all convolutional blocks in encoder
    x = encoder_input
    for layer_index in range(_num_conv_layers):
        layer_number = layer_index + 1
        conv_layer = Conv2D(
            filters=conv_filters[layer_index],          # (int) dimension of the output space
            kernel_size=conv_kernels[layer_index],      # (tuple) size of the convolution window
            strides=conv_strides[layer_index],
            padding="same",
            name=f"encoder_conv_layer_{layer_number}"
        )
        x = conv_layer(x)
        x = Activation(conv_activations[layer_index], name=f"encoder_relu_{layer_number}")(x)
        x = BatchNormalization(name=f"encoder_bn_{layer_number}")(x)
    conv_layers = x
    # Create Bottleneck
    _shape_before_bottleneck = K.int_shape(conv_layers)[1:]
    x = Flatten()(conv_layers)
    x = Dense(latent_space_dim, name="encoder_output")(x)
    bottleneck = x
    # Define the model
    encoder = Model(encoder_input, bottleneck, name="encoder")

    # Decoder
    # Create Input layer
    decoder_input = Input(shape=(latent_space_dim,), name="decoder_input")
    # Create a Dense layer
    num_neurons = np.prod(_shape_before_bottleneck) # [1, 2, 4] -> 8
    dense_layer = Dense(num_neurons, name="decoder_dense")(decoder_input)
    # Create a Reshape layer
    reshape_layer = Reshape(_shape_before_bottleneck)(dense_layer)
    # Create all convolutional blocks in decoder
    x = reshape_layer
    for layer_index in reversed(range(1, _num_conv_layers)):
        layer_num = _num_conv_layers - layer_index
        conv_transpose_layer = Conv2DTranspose(
            filters=conv_filters[layer_index],
            kernel_size=conv_kernels[layer_index],
            strides=conv_strides[layer_index],
            padding="same",
            name=f"decoder_conv_transpose_layer_{layer_num}"
        )
        x = conv_transpose_layer(x)
        x = Activation(conv_activations[layer_index], name=f"decoder_relu_{layer_num}")(x)
        x = BatchNormalization(name=f"decoder_bn_{layer_num}")(x)
    conv_transpose_layers = x
    # Create output layer
    conv_transpose_layer = Conv2DTranspose(
            filters=input_shape[-1], # Get the latest dimension of the input shape
            kernel_size=conv_kernels[0],
            strides=conv_strides[0],
            padding="same",
            name=f"decoder_conv_transpose_layer_{_num_conv_layers}"
        )
    x = conv_transpose_layer(conv_transpose_layers)
    decoder_output = Activation("sigmoid", name="sigmoid_layer")(x)
    # Define the model
    decoder = Model(decoder_input, decoder_output, name="decoder")

    # Autoencoder = Encoder + Decoder
    model_input = encoder_input
    model_output = decoder(encoder(model_input))
    model = Model(model_input, model_output, name="autoencoder")

    return model, encoder, decoder



"""
This function creates a forward autoencoder model.
Parameters:
    input_dim: int, shape of the input data (height * width * channels)
    conv_filters: list of integers, number of filters in each convolutional layer
    conv_activations: list of strings, activations in each convolutional layer
    latent_space_dim: integer, dimension of the latent space
"""
def create_autoencoder(
        input_dim,
        filters,
        activations,
        latent_space_dim
):
    if len(filters) != len(activations):
        raise ValueError("The length of the following lists must be the same: conv_filters, conv_activations")

    # Internal variables
    _num_layers = len(filters)

    # Encoder
    # Create Input layer
    encoder_input = Input(shape=(input_dim,), name="encoder_input")
    # Create all blocks in encoder
    x = encoder_input
    for layer_index in range(_num_layers):
        layer_number = layer_index + 1
        #print(f"layer_index: {layer_index} | activation: {activations[layer_index]}")
        layer = Dense(
            filters[layer_index], 
            activation=activations[layer_index],
            name=f"encoder_layer_{layer_number}"
        )
        x = layer(x)
    layers = x
    # Create Bottleneck
    bottleneck = Dense(latent_space_dim, name="encoder_output")(layers)
    # Define the model
    encoder = Model(encoder_input, bottleneck, name="encoder")

    #encoder.summary()

    # Decoder
    # Create Input layer
    decoder_input = Input(shape=(latent_space_dim,), name="decoder_input")    
    # Create all convolutional blocks in decoder
    x = decoder_input
    for layer_index in reversed(range(0, _num_layers)):
        layer_num = _num_layers - layer_index
        layer = Dense(
            filters[layer_index], 
            activation=activations[layer_index],            
            name=f"decoder_layer_{layer_num}"
        )
        x = layer(x)
    layers = x
    # Create output layer
    sigmoid_layer = Dense(input_dim, activation="sigmoid", name="sigmoid_layer")(layers)
    decoder_output = sigmoid_layer
    # Define the model
    decoder = Model(decoder_input, decoder_output, name="decoder")

    #decoder.summary()

    # Autoencoder = Encoder + Decoder
    model_input = encoder_input
    model_output = decoder(encoder(model_input))
    model = Model(model_input, model_output, name="autoencoder")

    return model, encoder, decoder