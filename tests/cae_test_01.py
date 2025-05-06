import sys
import os
import numpy as np
import matplotlib.pylab as plt
import tensorflow as tf

# Work-around for loading a module from a parent folder in Jupyter/Notebooks
parent_dir = os.path.abspath('.')
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from modules.autoencoders_keras import ConvolutedAutoencoder

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

    cae_filters=(32, 64, 64, 64)
    cae_activations=('relu', 'relu', 'relu', 'relu')
    cae_kernels=(3, 3, 3, 3)
    cae_strides=(1, 2, 2, 1)
    cae_latent_space_dim=2
    
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

    cae = ConvolutedAutoencoder(
        ds.shape[1:], 
        cae_filters, 
        cae_kernels, 
        cae_strides, 
        cae_activations, 
        cae_latent_space_dim
    )
    optimizer = tf.keras.optimizers.get("Adam")
    optimizer.learning_rate.assign(.0001)
    cae.compile(optimizer=optimizer, loss='mse')
    cae.print_summaries()
    cae_history = cae.train_model(x_train=x_train, x_val=x_val, batch_size=256, epochs=10, shuffle=True)

    # Plot training & validation loss values
    plt.figure(figsize=(10, 5))
    plt.plot(cae_history.history['loss'])
    plt.plot(cae_history.history['val_loss'])
    plt.title('Model CAE loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.show()

    return 

if __name__ == '__main__':
    #print(os.path.abspath('.'))
    #print(sys.path)
    main()