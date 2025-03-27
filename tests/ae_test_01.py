import sys
import os
import numpy as np
import matplotlib.pylab as plt
import tensorflow as tf

# Work-around for loading a module from a parent folder in Jupyter/Notebooks
parent_dir = os.path.abspath('.')
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from modules.autoencoders import VanillaAutoencoder

def main():
    N = 1000
    ds = np.random.randint(0,256,size=(N, 28, 28, 1)).astype('uint8')
    #print(ds.shape)

    # Fake Normalization
    norm_ds = ds

    x_train = norm_ds[0:int(N*.75)]
    #print(x_train.shape)
    x_val = norm_ds[int(N*.75):]
    #print(x_val.shape)

    flatten_x_train = x_train.reshape((len(x_train), -1))
    flatten_x_val = x_val.reshape((len(x_val), -1))

    neurons=(128, 64, 32)
    activations=('relu', 'relu', 'relu')
    latent_space_dim=2
    
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

    ae = VanillaAutoencoder(
        input_dim=flatten_x_train.shape[1],
        neurons=neurons, 
        activations=activations, 
        latent_space_dim=latent_space_dim
    )
    optimizer = tf.keras.optimizers.get("Adam")
    optimizer.learning_rate.assign(.0001)
    ae.compile(optimizer=optimizer, loss='mse')
    ae.print_summaries()
    ae_history = ae.train(x_train=flatten_x_train, x_val=flatten_x_val, batch_size=256, epochs=10, shuffle=True)

    # Plot training & validation loss values
    plt.figure(figsize=(10, 5))
    plt.plot(ae_history.history['loss'])
    plt.plot(ae_history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.show()

    return 

if __name__ == '__main__':
    #print(os.path.abspath('.'))
    #print(sys.path)
    main()