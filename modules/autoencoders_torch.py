import torch
from prettytable import PrettyTable
from torch import nn
from datetime import datetime
import pickle
import abc
from typing import Callable
from torch.nn.modules import activation
from collections import OrderedDict
import torch._dynamo

Activation = Callable[..., nn.Module]
def get_activation_fn(act: str) -> Activation:
    # get list from activation submodule as lower-case
    activations_lc = [str(a).lower() for a in activation.__all__]
    if (act := str(act).lower()) in activations_lc:
        # match actual name from lower-case list, return function/factory
        idx = activations_lc.index(act)
        act_name = activation.__all__[idx]
        act_func = getattr(activation, act_name)
        return act_func
    else:
        raise ValueError(f"Cannot find activation function for string <{act}>")

class Autoencoder(nn.Module):
    _metaclass__ = abc.ABCMeta

    """
    Autoencoder class inheriting PyTorch Module
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
        #self.built = True

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

    def train_one_epoch_(self, training_loader, optimizer, loss_fn):
        running_loss = 0.
        last_loss = 0.

        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        for i, data in enumerate(training_loader):
            # Every data instance is an input + label pair
            inputs = data

            # Zero your gradients for every batch!
            optimizer.zero_grad()

            # Make predictions for this batch
            self.mo
            outputs = self.forward(inputs)

            # Compute the loss and its gradients
            loss = loss_fn(outputs, inputs)
            loss.backward()

            # Adjust learning weights
            optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            if i % 1000 == 999:
                last_loss = running_loss / 1000 # loss per batch
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                running_loss = 0.

        return last_loss

    def train_model(self, x_train, x_val, batch_size, epochs, shuffle=True, loss_fn=None, optimizer=None):
        """
        Train the autoencoder model.
        Parameters:
            x_train: training data
            x_val: validation data
            batch_size: batch size for training
            epochs: number of epochs to train
            shuffle: whether to shuffle the training data
        """
        if loss_fn is None:
            loss_fn = nn.MSELoss()
        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self._optimizer = optimizer

        class History:
            def __init__(self):
                self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                self.history = {"loss": [], "val_loss": []}
            def add(self, loss, val_loss):
                self.history["loss"].append(loss)
                self.history["val_loss"].append(val_loss)
        self._training_history = History()

        training_loader = torch.utils.data.DataLoader(x_train, batch_size=batch_size, shuffle=shuffle)
        validation_loader = torch.utils.data.DataLoader(x_val, batch_size=batch_size, shuffle=False)

        epoch_number = 0
        for epoch in range(epochs):
            print('Epoch {}:'.format(epoch_number + 1))            
            # Make sure gradient tracking is on, and do a pass over the data
            self.train(True)            
            avg_loss = self.train_one_epoch_(training_loader, self._optimizer, loss_fn)
            running_vloss = 0.0
            # Set the model to evaluation mode, disabling dropout and using population statistics for batch normalization
            self.eval()            
            # Disable gradient computation and reduce memory consumption
            with torch.no_grad():
                for i, vdata in enumerate(validation_loader):
                    vinputs, vlabels = vdata
                    voutputs = self.forward(vinputs)
                    vloss = loss_fn(voutputs, vlabels)
                    running_vloss += vloss
            avg_vloss = running_vloss / (i + 1)
            print('Loss train {} valid {}'.format(avg_loss, avg_vloss))
            self._training_history.add(avg_loss, avg_vloss)
            epoch_number += 1

        return self._training_history

    def get_training_history(self):
        return self._training_history

    def get_encoder(self):
        return self._encoder

    def get_decoder(self):
        return self._decoder
    
    def count_parameters_(self, model):
        table = PrettyTable(["Layer (type)", "Output Shape", "Param #"])
        total_params = 0
        total_trainable_params = 0

        out_dict = {}
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad: continue
            main_name = name.replace(".weight", "").replace(".bias", "")
            if main_name not in out_dict:
                out_dict[main_name] = {}
            params = parameter.numel()
            if name.endswith(".weight"):
                total_params+=params
                total_trainable_params+=params
                out_dict[main_name]["Params"] = params
            elif name.endswith(".bias"):
                out_dict[main_name]["Size"] = params

        for k,v in out_dict.items():            
            table.add_row([k, v["Size"], v["Params"]])

        print(table)
        print(f"Total Params: {total_params}")
        print(f"Total Trainable Params: {total_trainable_params}")
        print(f"Non-trainable params: {total_params-total_trainable_params}")
        return total_params 

    def print_summaries(self):
        self.count_parameters_(self)
        self.count_parameters_(self._encoder)
        self.count_parameters_(self._decoder)

    @classmethod
    def load(cls, save_folder, model_name):
        pass

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
        encoder_layers_ = []
        # Encoder   
        for layer_index in range(0, self._num_layers):
            in_shape_ = self._input_dim if layer_index == 0 else self._neurons[layer_index-1]
            out_shape_ = self._neurons[layer_index]
            name_layer_=f"encoder_layer_{layer_index+1}"
            layer_ = nn.Linear(in_shape_, out_shape_)
            name_activation_=f"encoder_activation_{self._activations[layer_index]}_{layer_index+1}"
            activation_ = get_activation_fn(self._activations[layer_index])
            encoder_layers_.append((name_layer_, layer_))
            encoder_layers_.append((name_activation_, activation_()))
        # Create Bottleneck
        bottleneck_layer_ = nn.Linear(self._neurons[-1], self._latent_space_dim)        
        encoder_layers_.append(("encoder_bottleneck_layer", bottleneck_layer_))
        # Define the encoder model
        self._encoder = nn.Sequential(OrderedDict(encoder_layers_))

        decoder_layers_ = []
        for layer_index in reversed(range(0, self._num_layers)):
            layer_num = self._num_layers - layer_index
            in_shape_ = self._latent_space_dim if layer_num-1 == 0 else self._neurons[layer_index+1]
            out_shape_ = self._neurons[layer_index]
            name_layer_=f"decoder_layer_{layer_num}"
            layer_ = nn.Linear(in_shape_, out_shape_)
            name_activation_=f"decoder_activation_{self._activations[layer_index]}_{layer_num}"
            activation_ = get_activation_fn(self._activations[layer_index])
            decoder_layers_.append((name_layer_, layer_))
            decoder_layers_.append((name_activation_, activation_()))
        # Create output layer
        decoder_output_layer_ = nn.Linear(self._neurons[0], self._input_dim)
        decoder_output_activation_ = nn.Sigmoid()
        decoder_layers_.append(("decoder_output_layer", decoder_output_layer_))
        decoder_layers_.append(("decoder_output_activation", decoder_output_activation_))
        # Define the encoder model
        self._decoder = nn.Sequential(OrderedDict(decoder_layers_))

    def forward(self, x):
        z = self._encoder(x)
        x = self._decoder(z)
        return x

    def _get_parameters_to_save(self):
        return [
            self._input_dim,
            self._neurons,
            self._activations,
            self._latent_space_dim,
            self.name
        ]

