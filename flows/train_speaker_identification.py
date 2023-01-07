import os
import sys
import time
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupKFold
from sklearn.utils.class_weight import compute_class_weight

sys.path.append(os.path.abspath(os.path.join('utils')))
from utils import (
    format_time,
    save_pickle,
    plot_confusion_matrix
    )

sys.path.append(os.path.abspath(os.path.join('models')))
from model import (
    load_data,
    build_model,
    train_model,
    plot_history,
    save_model_ext
    )

#Setting up the paths
ROOT_PATH = Path.cwd()
ROOT_DATASET_PATH = ROOT_PATH.joinpath('dataset').joinpath('data')
OUTPUT_PATH = ROOT_PATH.joinpath('output')

def create_and_save_model(X, y):
    print('---- Creating general model for all ----')

    #Label encoding
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    #Calculate class weights for each class
    class_weights = compute_class_weight(
        class_weight = "balanced",
        classes = np.unique(y_enc),
        y = y_enc                                                   
    )
    class_weights = dict(zip(np.unique(y_enc), class_weights))

    # Create the deep neural network
    input_shape = (X.shape[1], X.shape[2]) # timesteps=63, features=39
    n_classes = len(list(le.classes_))

    #Initialize model
    model = build_model(input_shape, n_classes)

    # Train model
    train_model(
        model = model,
        X_train = X, 
        y_train = y_enc,  
        class_weights = class_weights
    )

    labels_string = json.dumps(list(le.classes_))

    # save model
    fname = OUTPUT_PATH.joinpath('speaker_identification.h5')
    save_model_ext(
        model=model, 
        filepath=fname, 
        metadata=labels_string
    ) 


def speaker_identification_experiment(path):
    #Load the extracted feature set for the SOLO dataset
    X, y, filename = load_data(data_path=path)

    #Label encoding
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    #List of dataframes in which will be stored the results of each testing fold
    results = []

    #We use GrouKfold experimentaion in order to be all the audio files in testing position in a circular manner
    #Furthermore we secure that audio segments of the same audio file will not be in train and test at the same time
    #protecting the model of getting biased
    #I used 5 splits, which means that the testing data in each fold is about the 20% of the initial set
    #The rest 80% will be used for training and validation set
    group_kfold_1 = GroupKFold(n_splits=5)
    for i, (idx1, test_idx) in enumerate(group_kfold_1.split(X=X, y=y_enc, groups=filename)):
        print(f"Fold {i+1}:")
        X1 , y1, filename1 = X[idx1], y_enc[idx1], filename[idx1]
        X_test , y_test, filename_test = X[test_idx], y_enc[test_idx], filename[test_idx]

        #In order to have a validation set without any type of data leakage (segments of the same audio file shared in train and val)
        #I had to use another groupkfold split with only 1 iteration
        #The 80% of X1 will be used as training set while the rest 20% as validation set (5 splits)
        group_kfold_2 = GroupKFold(n_splits=5)
        for j, (idx2, val_idx) in enumerate(group_kfold_2.split(X=X1, y=y1, groups=filename1)):
            if j > 0:
                break
            
            #Get the train and the validation set
            X_train , y_train = X1[idx2], y1[idx2]
            X_val , y_val = X1[val_idx], y1[val_idx]

            #Calculate the class weights (not totally necessary, the dataset is balanced in general)
            class_weights = compute_class_weight(
                class_weight = "balanced",
                classes = np.unique(y_train),
                y = y_train                                                   
            )
            class_weights = dict(zip(np.unique(y_train), class_weights))

            # Create the deep neural network
            input_shape = (X_train.shape[1], X_train.shape[2]) # timesteps=63, features=39
            n_classes = len(list(le.classes_))

            #Initialize model
            model = build_model(input_shape, n_classes)

            # Train model
            history = train_model(
                model, 
                X_train, y_train, 
                X_val, y_val, 
                class_weights
            )
            
            # plot accuracy/error for training and validation
            plot_history(history, fullpath=OUTPUT_PATH.joinpath(f'train_history_fold_{i+1}.jpg'))

            #Make predictions
            y_prob = model.predict(X_test)
            y_pred = np.argmax(y_prob, axis=1)

            y_test = le.inverse_transform(y_test)
            y_pred = le.inverse_transform(y_pred)
            
            #Create a dataframe, which contains the true labels, the predicted labels and the predicted probs for each label
            preds_df = pd.DataFrame({
                'source_file': filename_test,
                'label': y_test,
                'prediction': y_pred
            })
            probs_df = pd.DataFrame(data=y_prob, columns = list(le.classes_))
            results.append(pd.concat([preds_df, probs_df], axis=1))

    #List of dataframes -> dataframe
    results = pd.concat(results)

    #Plot and store confusion matrix
    plot_confusion_matrix(results['label'].values, results['prediction'].values, norm=False, fullpath=OUTPUT_PATH.joinpath('step1_cm.jpg'))
    plot_confusion_matrix(results['label'].values, results['prediction'].values, norm=True, fullpath=OUTPUT_PATH.joinpath('step1_normalized_cm.jpg'))

    #Save results locally in pickle format
    fname = OUTPUT_PATH.joinpath('step1_results_groupkfold.pickle')
    save_pickle(fname, results)

    #Train and store a model with all available data
    if save_model == 'yes':
        create_and_save_model(X, y)

def main():
    global save_model

    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", 
        "--input", 
        help="Path to training dataset", 
        dest="input_path", 
        required=True
    )

    parser.add_argument(
        "-n",
        "--new_model",
        help='Create and save a new model',
        dest="save_model",
        choices=["yes", "no"],
        required=True
    )

    args = parser.parse_args()
    path = Path(args.input_path)
    save_model = args.save_model

    #Trigger the experiment
    speaker_identification_experiment(path)
    print(f'Finished in {format_time(time.time() - start_time)}')


if __name__ == '__main__':
    main()