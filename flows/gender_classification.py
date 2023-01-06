import os
import sys
import time
import librosa
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from multiprocessing import Pool

sys.path.append(os.path.abspath(os.path.join('utils')))
from utils import (
    format_time,
    save_pickle,
    plot_confusion_matrix
)

sys.path.append(os.path.abspath(os.path.join('audio')))
from audioblock import audio_load

#Setting up the paths
ROOT_PATH = Path.cwd()
OUTPUT_PATH = ROOT_PATH.joinpath('output')

SAMPLE_RATE = 22050

mapping_name_gender = {
    'frf01': 'female',
    'frf02': 'female',
    'frf03': 'female',
    'frf04': 'female',
    'irf01': 'female',
    'irf02': 'female',
    'irf03': 'female',
    'irf04': 'female',
    'irf05': 'female',
    'irf06': 'female',
    'irf07': 'female',
    'irf08': 'female',
    'irf09': 'female',
    'irf10': 'female',
    'irf11': 'female',
    'irf12': 'female',
    'frm01': 'male', 
    'frm02': 'male',
    'frm03': 'male',
    'frm04': 'male',
    'irm01': 'male',
    'irm02': 'male', 
    'irm03': 'male', 
    'irm04': 'male', 
    'irm05': 'male', 
    'irm06': 'male', 
    'irm07': 'male', 
    'irm08': 'male', 
    'irm09': 'male', 
    'irm10': 'male', 
    'irm11': 'male', 
    'irm12': 'male', 
    'irm13': 'male', 
    'irm14': 'male', 
    'irm15': 'male', 
    'irm16': 'male',
}

def extract_f0(filepath):

    #Extract source info of the proccessing file
    filename, source, corpus = filepath.parts[-1], filepath.parts[-2], filepath.parts[-3]
    print(f'Processing file: {filename}')

    #Load the audio file as floating point time series
    sig = audio_load(filepath, SAMPLE_RATE)

    #Apply the PYIN algorithm for extracting the F0 of the signal
    f0, voiced_flag, voiced_probs = librosa.pyin(
        sig,
        fmin=50,
        fmax=300,
        sr=SAMPLE_RATE
    )

    #Edge case, may the f0 array contain only NaNs
    if np.isnan(f0).all():
        print(f'For the audio file {filename} all the values of f0 array are NaNs. Skipping...')
        return None
    
    #Store all extractes info in a dictionary
    data = {
       "filename": filename,
       "corpus": corpus,
       "user": source,
       "mean_f0": np.nanmean(f0),
       "median_f0": np.nanmedian(f0) 
    }

    return data


def create_dataset(path):

    #Create a list which contains all the audio files' that are under `path` directory
    audiofiles = list((path).glob('**/*.wav'))
    print(f'The total number of audio files is: {len(audiofiles)}')

    #`data` is a list in which will be appended dictionaries with the extracted info for each file
    #Iterate in parallel through all the audio files to extract the fundamental frequency f0
    pool = Pool()
    data = pool.map(extract_f0, audiofiles)
    pool.close()
    pool.join()

    #Remove possible None values from the list
    data = list(filter(lambda item: item is not None, data))

    #List of dicts -> pandas dataframe
    data = pd.DataFrame(data)

    #Map each row/user with the corresponding gender
    data['gender'] = data['user'].map(mapping_name_gender)

    #Store dataset locally in pickle format
    fname = OUTPUT_PATH.joinpath('step3_dataset.pickle')
    save_pickle(fname, data)

    return data

def predict_gender(data):
    #Predict gender based on threshold rule
    threshold = 160 #Hz
    data["prediction"] = data["median_f0"].apply(lambda x: 'female' if x>threshold else 'male')
    plot_confusion_matrix(data['gender'].values, data['prediction'].values, norm=False, fullpath=OUTPUT_PATH.joinpath('step3_cm.jpg'), figsize=(8, 5))
    plot_confusion_matrix(data['gender'].values, data['prediction'].values, norm=True, fullpath=OUTPUT_PATH.joinpath('step3_normalized_cm.jpg'), figsize=(8, 5))


def main():
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", 
        "--input", 
        help="Path to dataset", 
        dest="input_path", 
        required=True
    )  

    args = parser.parse_args()
    path = Path(args.input_path)
    data = create_dataset(path)
    predict_gender(data)
    print(f'Finished in {format_time(time.time() - start_time)}')

if __name__ == '__main__':
    main()