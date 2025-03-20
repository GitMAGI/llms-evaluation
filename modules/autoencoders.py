import tensorflow as tf
from keras import Model
from keras.api.layers import Input, Dense, Conv2D, Lambda, Conv2DTranspose, Reshape, Flatten, BatchNormalization, Activation
import numpy as np
from datetime import datetime
import os
import pickle
from abc import ABC, abstractmethod

class Autoencoder(ABC):
    def __init__(
            self,            
            latent_space_dim,
            model_type,
            model_name=None
        ):        
        if model_name is None:
            model_name = datetime.now().strftime("%Y%m%d%H%M%S")           
        self._latent_space_dim = latent_space_dim
        self._model = None
        self._encoder = None
        self._decoder = None
        self._model_name = model_name
        self._model_type = model_type
        self._training_history = None
        self._build()

    @abstractmethod
    def _get_parameters_to_save(self):
        pass

    @abstractmethod
    def _build(self):
        pass

    def get_name(self):
        return self._model_name
    
    def get_type(self):
        return self._model_type

    def get_training_history(self):
        return self._training_history

    def get_encoder(self):
        return self._encoder

    def get_decoder(self):
        return self._decoder
    
    def get_autoencoder(self):
        return self._model
    
    def print_summaries(self):
        self._model.summary()
        self._encoder.summary()
        self._decoder.summary()

    def compile(self, loss, optimizer):
        self._model.compile(loss=loss, optimizer=optimizer)

    def train(self, x_train, x_val, batch_size, epochs, shuffle=True):
        self._training_history = self._model.fit(
            x_train, x_train,
            validation_data=(x_val, x_val),
            batch_size=batch_size,
            shuffle=shuffle,
            epochs=epochs
        )
        return self._training_history

    def save(self, save_folder):
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        self._save_parameters(save_folder)
        self._save_weights(save_folder)
        self._save_training_history(save_folder)

    def _save_training_history(self, save_folder):
        if self._training_history:
            save_path = os.path.join(save_folder, f"{self._model_type}_{self._model_name}.training_history.pkl")
            with open(save_path, "wb") as f:
                pickle.dump(self._training_history, f)

    def _save_parameters(self, save_folder):        
        parameters = self._get_parameters_to_save()
        save_path = os.path.join(save_folder, f"{self._model_type}_{self._model_name}.parameters.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(parameters, f)

    def _save_weights(self, save_folder):
        save_path = os.path.join(save_folder, f"{self._model_type}_{self._model_name}.weights.h5")
        self._model.save_weights(save_path)
    
    @classmethod
    def load(cls, save_folder, model_name):
        model_type = cls.__name__
        # Load parameters
        parameters = []
        parameters_path = os.path.join(save_folder, f"{model_type}_{model_name}.parameters.pkl")
        if not os.path.exists(parameters_path):
            raise ValueError(f"Parameters file for model '{model_name}' not found in '{save_folder}'. Do not use prefix '{model_type}' in model_name")
        with open(parameters_path, "rb") as f:
            parameters = pickle.load(f)
        # Load training history
        history = None
        history_path = os.path.join(save_folder, f"{model_type}_{model_name}.training_history.pkl")
        if os.path.exists(history_path):            
            with open(history_path, "rb") as f:
                history = pickle.load(f)
        # Load weights
        weights_path = os.path.join(save_folder, f"{model_type}_{model_name}.weights.h5")
        if not os.path.exists(weights_path):
            raise ValueError(f"Weights for model '{model_name}' not found in '{save_folder}'. Do not use prefix '{model_type}' in model_name")
        cls = cls(*parameters)
        cls._model.load_weights(weights_path)
        cls._training_history = history
        return cls

class VanillaAutoencoder(Autoencoder):
    """
    This class implements a Vanilla Autoencoder.
    Parameters:
        input_dim: int, input dimension
        neurons: tuple, number of neurons in each layer. e.g. (128, 64, 32) 
        activations: activation type for each layers. e.g. ('relu', 'relu', 'relu')
        latent_space_dim: int, dimension of the latent space
        model_name: str, name of the model (optional)
    """
    def __init__(
            self, 
            input_dim, 
            neurons, 
            activations, 
            latent_space_dim,
            model_name=None
        ):
        if len(neurons) != len(activations):
            raise ValueError("The length of the following lists must be the same: neurons, activations")
        
        self._input_dim = input_dim
        self._neurons = neurons
        self._num_layers = len(neurons)
        self._activations = activations

        super().__init__(latent_space_dim, self.__class__.__name__, model_name)

    def _build(self):        
        # Encoder
        # Create Input layer
        encoder_input = Input(shape=(self._input_dim,), name="encoder_input")        
        # Create all blocks in encoder
        x = encoder_input
        for layer_index in range(self._num_layers):
            layer_number = layer_index + 1
            layer = Dense(
                self._neurons[layer_index], 
                activation=self._activations[layer_index],
                name=f"encoder_layer_{layer_number}"
            )
            x = layer(x)
        prev_layers = x
        # Create Bottleneck
        bottleneck = Dense(self._latent_space_dim, name="encoder_output")(prev_layers)
        # Define the model
        encoder = Model(encoder_input, bottleneck, name=f"encoder")

        # Decoder
        # Create Input layer
        decoder_input = Input(shape=(self._latent_space_dim,), name="decoder_input")    
        # Create all blocks in decoder
        x = decoder_input
        for layer_index in reversed(range(0, self._num_layers)):
            layer_num = self._num_layers - layer_index
            layer = Dense(
                self._neurons[layer_index], 
                activation=self._activations[layer_index],            
                name=f"decoder_layer_{layer_num}"
            )
            x = layer(x)
        prev_layers = x
        # Create output layer
        sigmoid_layer = Dense(self._input_dim, activation="sigmoid", name="decoder_output")(prev_layers)
        decoder_output = sigmoid_layer
        # Define the model
        decoder = Model(decoder_input, decoder_output, name=f"decoder")

        # Autoencoder = Encoder + Decoder
        model_input = encoder_input
        model_output = decoder(encoder(model_input))
        model = Model(model_input, model_output, name=f"{self._model_type}_{self._model_name}_autoencoder")

        # Store components in class private attributes
        #self._model_input = model_input
        #self._model_output = model_output
        self._encoder = encoder
        self._decoder = decoder
        self._model = model

    def _get_parameters_to_save(self):
        return [
            self._input_dim,
            self._neurons,
            self._activations,
            self._latent_space_dim,            
            self._model_name
        ]

class ConvolutedAutoencoder(Autoencoder):
    """
    This class implements a Concoluted Autoencoder.
    Parameters:
        input_shape: tuple, shape of the input data (height, width, channels)
        conv_filters: tuple of integers, number of filters in each convolutional layer. e.g. (32, 64, 64, 64)
        conv_kernels: tuple of integers, size of the kernel in each convolutional layer. e.g. (3, 3, 3, 3)
        conv_strides: tuple of integers, strides in each convolutional layer. e.g. (1, 2, 2, 1)
        conv_activations: tuple of strings, activations in each convolutional layer. e.g. ('relu', 'relu', 'relu', 'relu')
        latent_space_dim: integer, dimension of the latent space
        model_name: str, name of the model (optional)
    """
    def __init__(
            self,
            input_shape,            
            filters,           # dimension of the output space
            kernels,           # size of the convolution window
            strides,
            activations,
            latent_space_dim,
            model_name=None
        ):
        if len(filters) != len(kernels) != len(strides) != len(activations):
            raise ValueError("The length of the following lists must be the same: filters, kernels, strides, activations")        

        self._input_shape = input_shape
        self._filters = filters
        self._kernels = kernels
        self._strides = strides
        self._activations = activations        
        self._num_layers = len(filters)
        self._shape_before_bottleneck = None

        super().__init__(latent_space_dim, self.__class__.__name__, model_name)

    def _build(self):   

        # Encoder
        # Create Input layer
        encoder_input = Input(shape=self._input_shape, name="encoder_input")
        # Create all convolutional blocks in encoder
        x = encoder_input
        for layer_index in range(self._num_layers):
            layer_number = layer_index + 1
            conv_layer = Conv2D(
                filters=self._filters[layer_index],          # (int) dimension of the output space
                kernel_size=self._kernels[layer_index],      # (tuple) size of the convolution window
                strides=self._strides[layer_index],
                padding="same",
                name=f"encoder_conv_layer_{layer_number}"
            )
            x = conv_layer(x)
            x = Activation(self._activations[layer_index], name=f"encoder_relu_{layer_number}")(x)
            x = BatchNormalization(name=f"encoder_bn_{layer_number}")(x)
        conv_layers = x

        # Create Bottleneck
        self._shape_before_bottleneck = conv_layers.shape[1:]
        x = Flatten()(conv_layers)
        x = Dense(self._latent_space_dim, name="encoder_output")(x)
        bottleneck = x

        # Define the model
        encoder = Model(encoder_input, bottleneck, name="encoder")

        # Decoder
        # Create Input layer
        decoder_input = Input(shape=(self._latent_space_dim,), name="decoder_input")
        # Create a Dense layer
        num_neurons = np.prod(self._shape_before_bottleneck) # [1, 2, 4] -> 8
        dense_layer = Dense(num_neurons, name="decoder_dense")(decoder_input)
        # Create a Reshape layer
        reshape_layer = Reshape(self._shape_before_bottleneck)(dense_layer)
        # Create all convolutional blocks in decoder
        x = reshape_layer
        for layer_index in reversed(range(1, self._num_layers)):
            layer_num = self._num_layers - layer_index
            conv_transpose_layer = Conv2DTranspose(
                filters=self._filters[layer_index],
                kernel_size=self._kernels[layer_index],
                strides=self._strides[layer_index],
                padding="same",
                name=f"decoder_conv_transpose_layer_{layer_num}"
            )
            x = conv_transpose_layer(x)
            x = Activation(self._activations[layer_index], name=f"decoder_relu_{layer_num}")(x)
            x = BatchNormalization(name=f"decoder_bn_{layer_num}")(x)
        conv_transpose_layers = x
        # Create output layer
        conv_transpose_layer = Conv2DTranspose(
                filters=self._input_shape[-1], # Get the latest dimension of the input shape
                kernel_size=self._kernels[0],
                strides=self._strides[0],
                padding="same",
                name=f"decoder_conv_transpose_layer_{self._num_layers}"
            )
        x = conv_transpose_layer(conv_transpose_layers)
        decoder_output = Activation("sigmoid", name="sigmoid_layer")(x)
        # Define the model
        decoder = Model(decoder_input, decoder_output, name="decoder")

        # Autoencoder = Encoder + Decoder
        model_input = encoder_input
        model_output = decoder(encoder(model_input))
        model = Model(model_input, model_output, name="autoencoder")

        # Autoencoder = Encoder + Decoder
        model_input = encoder_input
        model_output = decoder(encoder(model_input))
        model = Model(model_input, model_output, name=f"{self._model_type}_{self._model_name}_autoencoder")

        # Store components in class private attributes
        #self._model_input = model_input
        #self._model_output = model_output
        self._encoder = encoder
        self._decoder = decoder
        self._model = model

    def _get_parameters_to_save(self):
        return [
            self._input_shape,            
            self._filters,
            self._kernels,
            self._strides,
            self._activations,
            self._latent_space_dim,
            self._model_name
        ]

class VariationalAutoencoder(Autoencoder):
    """
    This class implements a Variational Autoencoder.
    Parameters:
        input_shape: tuple, shape of the input data (height, width, channels)
        filters: tuple of integers, number of filters in each convolutional layer. e.g. (32, 64, 64, 64)
        kernels: tuple of integers, size of the kernel in each convolutional layer. e.g. (3, 3, 3, 3)
        strides: tuple of integers, strides in each convolutional layer. e.g. (1, 2, 2, 1)
        activations: tuple of strings, activations in each convolutional layer. e.g. ('relu', 'relu', 'relu', 'relu')
        latent_space_dim: integer, dimension of the latent space
        model_name: str, name of the model (optional)
    """
    def __init__(
            self,
            input_shape,            
            filters,           # dimension of the output space
            kernels,           # size of the convolution window
            strides,
            activations,
            latent_space_dim,
            model_name=None
        ):
        if len(filters) != len(kernels) != len(strides) != len(activations):
            raise ValueError("The length of the following lists must be the same: filters, kernels, strides, activations")        

        self._input_shape = input_shape
        self._filters = filters
        self._kernels = kernels
        self._strides = strides
        self._activations = activations        
        self._num_layers = len(filters)
        self._shape_before_bottleneck = None
        self._bottleneck_mu = None
        self._bottleneck_log_variance = None
        self._reconstruction_loss_weight = None

        super().__init__(latent_space_dim, self.__class__.__name__, model_name)

    def _build(self):
        # Encoder
        # Create Input layer
        encoder_input = Input(shape=self._input_shape, name="encoder_input")
        # Create all convolutional blocks in encoder
        x = encoder_input
        for layer_index in range(self._num_layers):
            layer_number = layer_index + 1
            conv_layer = Conv2D(
                filters=self._filters[layer_index],          # (int) dimension of the output space
                kernel_size=self._kernels[layer_index],      # (tuple) size of the convolution window
                strides=self._strides[layer_index],
                padding="same",
                name=f"encoder_conv_layer_{layer_number}"
            )
            x = conv_layer(x)
            x = Activation(self._activations[layer_index], name=f"encoder_relu_{layer_number}")(x)
            x = BatchNormalization(name=f"encoder_bn_{layer_number}")(x)
        conv_layers = x

        # Create Bottleneck
        self._shape_before_bottleneck = conv_layers.shape[1:]        
        x = Flatten()(conv_layers)
        self._bottleneck_mu = Dense(self._latent_space_dim, name="mu")(x)
        self._bottleneck_log_variance = Dense(self._latent_space_dim, name="log_variance")(x)
        def _sample_point_from_normal_distribution(args):
            mu, log_variance = args
            epsilon = tf.random.normal(shape=tf.shape(mu), mean=0., stddev=1.)
            sampled_point = mu + tf.exp(log_variance / 2) * epsilon
            return sampled_point
        x = Lambda(function=_sample_point_from_normal_distribution, output_shape=(self._latent_space_dim,), name="encoder_output")([self._bottleneck_mu, self._bottleneck_log_variance])
        bottleneck = x

        # Define the model
        encoder = Model(encoder_input, bottleneck, name="encoder")

        # Decoder
        # Create Input layer
        decoder_input = Input(shape=(self._latent_space_dim,), name="decoder_input")
        # Create a Dense layer
        num_neurons = np.prod(self._shape_before_bottleneck) # [1, 2, 4] -> 8
        dense_layer = Dense(num_neurons, name="decoder_dense")(decoder_input)
        # Create a Reshape layer
        reshape_layer = Reshape(self._shape_before_bottleneck)(dense_layer)
        # Create all convolutional blocks in decoder
        x = reshape_layer
        for layer_index in reversed(range(1, self._num_layers)):
            layer_num = self._num_layers - layer_index
            conv_transpose_layer = Conv2DTranspose(
                filters=self._filters[layer_index],
                kernel_size=self._kernels[layer_index],
                strides=self._strides[layer_index],
                padding="same",
                name=f"decoder_conv_transpose_layer_{layer_num}"
            )
            x = conv_transpose_layer(x)
            x = Activation(self._activations[layer_index], name=f"decoder_relu_{layer_num}")(x)
            x = BatchNormalization(name=f"decoder_bn_{layer_num}")(x)
        conv_transpose_layers = x
        # Create output layer
        conv_transpose_layer = Conv2DTranspose(
                filters=self._input_shape[-1], # Get the latest dimension of the input shape
                kernel_size=self._kernels[0],
                strides=self._strides[0],
                padding="same",
                name=f"decoder_conv_transpose_layer_{self._num_layers}"
            )
        x = conv_transpose_layer(conv_transpose_layers)
        decoder_output = Activation("sigmoid", name="sigmoid_layer")(x)
        # Define the model
        decoder = Model(decoder_input, decoder_output, name="decoder")

        # Autoencoder = Encoder + Decoder
        model_input = encoder_input
        model_output = decoder(encoder(model_input))
        model = Model(model_input, model_output, name="autoencoder")

        # Autoencoder = Encoder + Decoder
        model_input = encoder_input
        model_output = decoder(encoder(model_input))
        model = Model(model_input, model_output, name=f"{self._model_type}_{self._model_name}_autoencoder")

        # Store components in class private attributes
        #self._model_input = model_input
        #self._model_output = model_output
        self._encoder = encoder
        self._decoder = decoder
        self._model = model

    def compile(self, optimizer, reconstruction_loss_weight=1000):
        self._reconstruction_loss_weight = reconstruction_loss_weight
        self._model.compile(
            optimizer=optimizer,
            loss=self._calculate_combined_loss,
            metrics=[self._calculate_reconstruction_loss, self._calculate_kl_loss]
        )
    
    def _calculate_combined_loss(self, y_target, y_predicted):
        reconstruction_loss = self._calculate_reconstruction_loss(y_target, y_predicted)
        kl_loss = self._calculate_kl_loss(y_target, y_predicted)
        combined_loss = self._reconstruction_loss_weight * reconstruction_loss + kl_loss
        return combined_loss

    def _calculate_reconstruction_loss(self, y_target, y_predicted):
        error = y_target - y_predicted
        reconstruction_loss = tf.reduce_mean(tf.square(error), axis=[1, 2, 3])
        return reconstruction_loss

    def _calculate_kl_loss(self, y_target, y_predicted):
        kl_loss = -0.5 * tf.reduce_sum(1 + self._bottleneck_log_variance - tf.square(self._bottleneck_mu) - tf.exp(self._bottleneck_log_variance), axis=1)
        return kl_loss

    def _get_parameters_to_save(self):
        return [
            self._input_shape,            
            self._filters,
            self._kernels,
            self._strides,
            self._activations,
            self._latent_space_dim,
            self._model_name
        ]