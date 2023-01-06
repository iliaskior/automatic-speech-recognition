from typing import Union
from pathlib import Path
import numpy as np
import pickle
import h5py
import tensorflow.keras as keras
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model, save_model


LEARNING_RATE = 0.0001
BATCH_SIZE = 16
EPOCHS = 20


def load_data(data_path: Union[str, Path]) -> Union[np.ndarray, np.ndarray]:
    """Load extracted features and transform to the expected format

    Parameters
    ----------
    data_path : Union[str, Path]
        Extracted features path

    Returns
    -------
    Union[np.ndarray, np.ndarray]
        Features, Labels
    """
    

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    X = np.array(data["mfccs_conc"])
    y = np.array(data["user"])
    filename = np.array(data["filename"])

    print("Data loaded successfully!")
    return X, y, filename
    

def build_model(input_shape, n_classes):
    """Generates RNN-LSTM model
    :param input_shape (tuple): Shape of input set
    :return model: RNN-LSTM model
    """

    # build network topology
    model = keras.Sequential()

    # 2 LSTM layers
    model.add(keras.layers.LSTM(256, input_shape=input_shape, return_sequences=True))
    model.add(keras.layers.LSTM(256))

    # dense layer
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.5))

    # output layer
    model.add(keras.layers.Dense(n_classes, activation='softmax'))

    return model


def train_model(model, X_train, y_train, X_val=None, y_val=None, class_weights=None):

    # compile model
    optimiser = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimiser,
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

    model.summary()

    # Train model
    if X_val is not None:
        history = model.fit(
            X_train, y_train, 
            batch_size=BATCH_SIZE, 
            epochs=EPOCHS, 
            class_weight=class_weights,
        )
    else:
        history = model.fit(
            X_train, y_train, 
            validation_data=(X_val, y_val), 
            batch_size=BATCH_SIZE, 
            epochs=EPOCHS, 
            class_weight=class_weights,
        )


    return history


def plot_history(history, fullpath):

    if not isinstance(fullpath, Path):
        fullpath = Path(fullpath)

    epochs = len(history.history["accuracy"])
    xticks = np.arange(1, epochs+1, 1)

    fig, axs = plt.subplots(2)
    # Create accuracy subplot
    axs[0].plot(xticks, history.history["accuracy"], label="Train accuracy")
    axs[0].plot(xticks, history.history["val_accuracy"], label="Val accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    # Create error subplot
    axs[1].plot(xticks, history.history["loss"], label="Train error")
    axs[1].plot(xticks, history.history["val_loss"], label="Val error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.tight_layout()
    fullpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(fullpath), transparent=True, bbox_inches='tight')
    plt.show(block=False)


def load_model_ext(filepath, custom_objects=None):
    model = load_model(filepath, custom_objects=None)
    with h5py.File(filepath, mode='r') as f:
        metadata = f.attrs.get('labels', None)
    print('Model loaded succesfully!')
    return model, metadata


def save_model_ext(model, filepath, overwrite=True, metadata=None):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    save_model(model, filepath, overwrite)
    if metadata is not None:
        with h5py.File(filepath, mode='a') as f:
            f.attrs['labels'] = metadata
        print('Model saved succesfully!')