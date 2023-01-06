import os
import sys
import time
import json
import argparse
import pandas as pd
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join('utils')))
from utils import (
    format_time,
    save_pickle,
    plot_confusion_matrix
)

sys.path.append(os.path.abspath(os.path.join('models')))
from model import (
    load_data,
    load_model_ext
)

#Setting up the paths
ROOT_PATH = Path.cwd()
ROOT_DATASET_PATH = ROOT_PATH.joinpath('dataset').joinpath('data')
OUTPUT_PATH = ROOT_PATH.joinpath('output')
PRETRAINED_MODEL_PATH = OUTPUT_PATH.joinpath('speaker_identification.h5')


def predict(testing_dataset_path):

    #Load the pretrained model
    pretrained_model, metadata = load_model_ext(PRETRAINED_MODEL_PATH)
    labels = json.loads(metadata)

    #Load the extracted feature set for the testing dataset
    X, y, filename = load_data(testing_dataset_path)

    #Make predictions
    y_prob = pretrained_model.predict(X)
    probs_df = pd.DataFrame(data=y_prob, columns=labels)

    #Create a results dataframe, which contains the true labels, the predicted labels and the predicted probs for each label
    y_pred = probs_df.idxmax(axis=1)
    preds_df = pd.DataFrame({
                'label': y,
                'prediction': y_pred
            })
    results = pd.concat([preds_df, probs_df], axis=1)

    #Plot and save the confusion matrices
    plot_confusion_matrix(results['label'].values, results['prediction'].values, norm=False, fullpath=OUTPUT_PATH.joinpath('step2_cm.jpg'))
    plot_confusion_matrix(results['label'].values, results['prediction'].values, norm=True, fullpath=OUTPUT_PATH.joinpath('step2_normalized_cm.jpg'))

    #Store results locally in pickle format
    fname = OUTPUT_PATH.joinpath('step2_results.pickle')
    save_pickle(fname, results)


def main():
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", 
        "--input", 
        help="Path to testing dataset", 
        dest="input_path", 
        required=True
    )  

    args = parser.parse_args()
    path = Path(args.input_path)
    predict(path)
    print(f'Finished in {format_time(time.time() - start_time)}')

if __name__ == '__main__':
    main()