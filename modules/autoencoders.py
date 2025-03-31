import os
import tensorflow as tf
from keras import Model, layers
from keras.api.layers import Input, Concatenate, Dense, Conv2D, Conv2DTranspose, Reshape, Flatten, BatchNormalization, Activation
import numpy as np
from datetime import datetime
import pickle
import abc

class Autoencoder(Model):
    _metaclass__ = abc.ABCMeta
    
    """
    Autoencoder class inheriting Keras Model
    """
    def __init__(
            self,            
            latent_space_dim,
            model_name=None,
            **kwargs
        ):
        self._model_type = self.__class__.__name__

        self._encoder = None
        self._decoder = None
        self._training_history = None

        super().__init__(**kwargs)

        if model_name is None:
            model_name = datetime.now().strftime("%Y%m%d%H%M%S")
        self._latent_space_dim = latent_space_dim        
        self.name = model_name
        
        self._build()        
        self.built = True

    @abc.abstractmethod
    def _get_parameters_to_save(self):
        pass

    @abc.abstractmethod
    def _build(self):
        pass

    def get_name(self):
        return self.name
    
    def get_type(self):
        return self._model_type

    def get_training_history(self):
        return self._training_history

    def get_encoder(self):
        return self._encoder

    def get_decoder(self):
        return self._decoder
        
    def print_summaries(self):
        self.summary()
        self._encoder.summary()
        self._decoder.summary()

    def train(self, x_train, x_val, batch_size, epochs, shuffle=True):
        self._training_history = self.fit(
            x_train, 
            x_train,
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
        if self._training_history and self._training_history.history:
            save_path = os.path.join(save_folder, f"{self._model_type}_{self.name}.training_history.pkl")
            with open(save_path, "wb") as f:
                pickle.dump(self._training_history.history, f)

    def _save_parameters(self, save_folder):        
        parameters = self._get_parameters_to_save()
        save_path = os.path.join(save_folder, f"{self._model_type}_{self.name}.parameters.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(parameters, f)

    def _save_weights(self, save_folder):
        save_path = os.path.join(save_folder, f"{self._model_type}_{self.name}.weights.h5")
        self.save_weights(save_path)
    
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
            history = tf.keras.callbacks.History()         
            with open(history_path, "rb") as f:
                history.history = pickle.load(f)
        # Load weights
        weights_path = os.path.join(save_folder, f"{model_type}_{model_name}.weights.h5")
        if not os.path.exists(weights_path):
            raise ValueError(f"Weights for model '{model_name}' not found in '{save_folder}'. Do not use prefix '{model_type}' in model_name")
        cls = cls(*parameters)
        cls.load_weights(weights_path)
        cls._training_history = history
        return cls
    
    def call(self, x):
        encoded = self._encoder(x)
        decoded = self._decoder(encoded)
        return decoded

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
            model_name=None,
            **kwargs
        ):
        if len(neurons) != len(activations):
            raise ValueError("The length of the following lists must be the same: neurons, activations")
        
        self._input_dim = input_dim
        self._neurons = neurons
        self._num_layers = len(neurons)
        self._activations = activations

        super(VanillaAutoencoder, self).__init__(latent_space_dim, model_name, **kwargs)

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
        # Create Bottleneck
        z = Dense(self._latent_space_dim, name="encoder_output")(x)
        # Define the model
        encoder = Model(encoder_input, z, name=f"encoder")

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
        # Create output layer
        x = Dense(self._input_dim, activation="sigmoid", name="decoder_output")(x)
        decoder_output = x
        # Define the model
        decoder = Model(decoder_input, decoder_output, name=f"decoder")

        # Autoencoder = Encoder + Decoder
        model_input = encoder_input
        model_output = decoder(encoder(model_input))
        model = Model(model_input, model_output, name=f"{self._model_type}_{self.name}_autoencoder")

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
            self.name
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
            model_name=None,
            **kwargs
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

        super(ConvolutedAutoencoder, self).__init__(latent_space_dim, model_name, **kwargs)

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

        # Create Bottleneck
        self._shape_before_bottleneck = x.shape[1:]
        x = Flatten()(x)
        z = Dense(self._latent_space_dim, name="encoder_output")(x)

        # Define the model
        encoder = Model(encoder_input, z, name="encoder")

        # Decoder
        # Create Input layer
        decoder_input = Input(shape=(self._latent_space_dim,), name="decoder_input")
        # Create a Dense layer
        num_neurons = np.prod(self._shape_before_bottleneck) # [1, 2, 4] -> 8
        x = Dense(num_neurons, name="decoder_dense")(decoder_input)
        # Create a Reshape layer
        x = Reshape(self._shape_before_bottleneck)(x)
        # Create all convolutional blocks in decoder
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
        # Create output layer
        conv_transpose_layer = Conv2DTranspose(
                filters=self._input_shape[-1], # Get the latest dimension of the input shape
                kernel_size=self._kernels[0],
                strides=self._strides[0],
                padding="same",
                name=f"decoder_conv_transpose_layer_{self._num_layers}"
            )
        x = conv_transpose_layer(x)
        decoder_output = Activation("sigmoid", name="sigmoid_layer")(x)
        # Define the model
        decoder = Model(decoder_input, decoder_output, name="decoder")

        # Autoencoder = Encoder + Decoder
        model_input = encoder_input
        model_output = decoder(encoder(model_input))
        model = Model(model_input, model_output, name=f"{self._model_type}_{self.name}_autoencoder")

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
            self.name
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
            model_name=None,
            reconstruction_loss_weight = 1000,
            **kwargs
        ):
        if len(filters) != len(kernels) != len(strides) != len(activations):
            raise ValueError("The length of the following lists must be the same: filters, kernels, strides, activations")        

        self._input_shape = input_shape
        self._latent_space_dim = latent_space_dim
        self._filters = filters
        self._kernels = kernels
        self._strides = strides
        self._activations = activations        
        self._num_layers = len(filters)
        self._shape_before_bottleneck = None
        self._reconstruction_loss_weight = reconstruction_loss_weight

        #creating the 3 loss trackers
        self._total_loss_tracker = tf.keras.metrics.Mean(name="loss")
        self._reconstruction_loss_tracker = tf.keras.metrics.Mean(name="recon_loss")
        self._kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

        super(VariationalAutoencoder, self).__init__(latent_space_dim, model_name, **kwargs)

    class Sampling(layers.Layer):
        def call(self, inputs):
            z_mean, z_log_var = inputs
            epsilon = tf.keras.backend.random_normal(shape=tf.shape(z_mean))
            return z_mean + tf.exp(0.5* z_log_var) * epsilon

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
                activation=self._activations[layer_index],
                padding="same",
                name=f"encoder_conv_layer_{layer_number}_with_activation_{self._activations[layer_index]}"
            )
            x = conv_layer(x)

        # Create Bottleneck
        self._shape_before_bottleneck = x.shape[1:]        
        x = Flatten()(x)
        z_mu = Dense(self._latent_space_dim, name="mu")(x)
        z_log_var = Dense(self._latent_space_dim, name="log_variance")(x)
        z = self.Sampling(name="encoder_output")([z_mu, z_log_var])

        # Define the model (multiple output defined in the array [bottleneck, mu, log_var])
        encoder = Model(encoder_input, [z, z_mu, z_log_var] , name="encoder")

        # Decoder
        # Create Input layer
        decoder_input = Input(shape=(self._latent_space_dim,), name="decoder_input")
        # Create a Dense layer
        num_neurons = np.prod(self._shape_before_bottleneck) # [1, 2, 4] -> 8
        #dense_layer = Dense(num_neurons, name="decoder_dense")(decoder_input)
        x = Dense(num_neurons, activation="relu", name="decoder_dense")(decoder_input)
        # Create a Reshape layer
        x = Reshape(self._shape_before_bottleneck)(x)
        # Create all convolutional blocks in decoder
        for layer_index in reversed(range(1, self._num_layers)):
            layer_num = self._num_layers - layer_index
            conv_transpose_layer = Conv2DTranspose(
                filters=self._filters[layer_index],
                kernel_size=self._kernels[layer_index],
                strides=self._strides[layer_index],
                activation=self._activations[layer_index],
                padding="same",
                name=f"decoder_conv_transpose_layer_{layer_num}"
            )
            x = conv_transpose_layer(x)
        # Create output layer
        conv_transpose_layer = Conv2DTranspose(
                filters=self._input_shape[-1], # Get the latest dimension of the input shape
                kernel_size=self._kernels[0],
                strides=self._strides[0],
                activation="sigmoid",
                padding="same",
                name=f"decoder_conv_transpose_layer_{self._num_layers}_with_activation_sigmoid"
            )
        decoder_output = conv_transpose_layer(x)
        # Define the model
        decoder = Model(decoder_input, decoder_output, name="decoder")

        self._encoder = encoder
        self._decoder = decoder        

    def _get_parameters_to_save(self):
        return [
            self._input_shape,            
            self._filters,
            self._kernels,
            self._strides,
            self._activations,
            self._latent_space_dim,
            self.name
        ]

    @property
    def metrics(self):
        return [ self._total_loss_tracker, self._reconstruction_loss_tracker, self._kl_loss_tracker ]

    def _custom_loss(self, input_data):
        encoded, mean, log_var = self._encoder(input_data)
        decoded = self._decoder(encoded)
        recon_loss = tf.reduce_sum(tf.keras.losses.binary_crossentropy(input_data, decoded), axis =(1, 2))
        recon_loss = tf.reduce_mean(recon_loss)
        kl_loss = -0.5 * (1 + log_var - tf.square(mean) - tf.exp(log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        total_loss = self._reconstruction_loss_weight * recon_loss + kl_loss
        return total_loss, recon_loss, kl_loss

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            total_loss, recon_loss, kl_loss = self._custom_loss(self, x)
            grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self._total_loss_tracker.update_state(total_loss)
        self._reconstruction_loss_tracker.update_state(recon_loss)
        self._kl_loss_tracker.update_state(kl_loss)
        return { m.name : m.result() for m in self.metrics }

    def test_step(self, data):
        # Unpack the data
        x, y = data
        # Compute predictions
        y_pred = self(x, training=False)
        # Updates the metrics tracking the loss
        total_loss, recon_loss, kl_loss = self._custom_loss(self, x)
        self._total_loss_tracker.update_state(total_loss)
        self._reconstruction_loss_tracker.update_state(recon_loss)
        self._kl_loss_tracker.update_state(kl_loss)

        # Update the metrics.
        for metric in self.metrics:
            if metric.name != "loss":
                metric.update_state(y, y_pred)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}

    def call(self, x):
        encoded, mu, log_var = self._encoder(x)
        decoded = self._decoder(encoded)
        return decoded
