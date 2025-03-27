import sys
import os
import numpy as np
import matplotlib.pylab as plt
import tensorflow as tf

# Work-around for loading a module from a parent folder in Jupyter/Notebooks
parent_dir = os.path.abspath('.')
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from modules.autoencoders import VariationalAutoencoder

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

    vae_filters=(32, 64)
    vae_activations=('relu', 'relu')
    vae_kernels=(3, 3)
    vae_strides=(1, 2)
    vae_latent_space_dim=2
    
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

    vae = VariationalAutoencoder(
        ds.shape[1:], 
        vae_filters, 
        vae_kernels, 
        vae_strides, 
        vae_activations, 
        vae_latent_space_dim, 
        reconstruction_loss_weight=1000
    )
    optimizer = tf.keras.optimizers.get("Adam")
    optimizer.learning_rate.assign(.0001)
    vae.compile(optimizer=optimizer)
    vae.print_summaries()
    vae_history = vae.train(x_train=x_train, x_val=x_val, batch_size=256, epochs=10, shuffle=True)

    # Plot training & validation loss values
    plt.figure(figsize=(10, 5))
    plt.plot(vae_history.history['loss'])
    plt.plot(vae_history.history['val_loss'])
    plt.title('Model VAE loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.show()

    return 

if __name__ == '__main__':
    #print(os.path.abspath('.'))
    #print(sys.path)
    main()