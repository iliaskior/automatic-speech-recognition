from pathlib import Path
from typing import Union
import numpy as np
import argparse
import time
import sys
import os

sys.path.append(os.path.abspath(os.path.join('audio')))
from audioblock import (
    audio_load,
    split_to_signals,
    extract_features
)

sys.path.append(os.path.abspath(os.path.join('utils')))
from utils import (
    format_time,
    save_pickle
    )

#Hyperparameters for audio feature extraction
SAMPLE_RATE = 16000 #Hz
WINDOW_SIZE = 2000 #milliseconds
WINDOW_STEP = 2000 #milliseconds
N_FFT = 128 #milliseconds
HOP_LENGTH = 32 #milliseconds
NUM_MFCC = 13

#Setting up the paths
ROOT_PATH = Path.cwd()
ROOT_DATASET_PATH = ROOT_PATH.joinpath('dataset').joinpath('data')


def create_features_dataset(filepaths: Union[str, Path], exportpath: Union[str, Path]) -> None:
    """Audio feature extraction for a given list of audio files path and storing them locally in a pickle file

    Parameters
    ----------
    filepaths : Union[str, Path]
        Root folder path under which the audio files are located 
    exportpath : Union[str, Path]
        Export path of the features set
    """

    #Store features under dictionary structure
    ftrs = {
        "filename": [],
        "corpus": [],
        "user": [],
        "segment_id": [],
        "mfccs_conc": []
    }

    audiofiles = list((filepaths).glob('**/*.wav'))

    #Iterate through all audio files of input dataset
    for f in audiofiles:
        try:
            #Extract source about the proccessing audio file
            filename, source, corpus = f.parts[-1], f.parts[-2], f.parts[-3]
            print(f'Processing file {filename} !!!')
            
            #Load the audio file as floating point time series
            sig = audio_load(f, sr=SAMPLE_RATE)

            #Split the signal into fixed size segments
            segs = split_to_signals(sig, sr=SAMPLE_RATE, size=WINDOW_SIZE, slide=WINDOW_STEP)

            #Extract audio features for each segment
            segment_id = 0
            for seg in segs:
                segment_id+=1
                mfccs, delta_mfccs, delta2_mfccs = extract_features(
                    sig = seg,
                    sr = SAMPLE_RATE,
                    num_mfcc=NUM_MFCC,
                    n_fft=N_FFT,
                    hop_length=HOP_LENGTH
                )
                
                #Vertical stack of the features
                mfccs_conc = np.vstack((mfccs, delta_mfccs))
                mfccs_conc = np.vstack((mfccs_conc, delta2_mfccs))

                #Store data
                ftrs["corpus"].append(corpus)
                ftrs["filename"].append(filename)
                ftrs["user"].append(source)
                ftrs["segment_id"].append(segment_id)
                ftrs["mfccs_conc"].append(mfccs_conc.T.tolist())

        except Exception as e:
            msg = f'Raised the exception: {e} for file {filename}'+ ' Skipping...'
            print(msg)
            continue

    #Save features locally in pickle format
    save_pickle(exportpath, ftrs)


def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Creates a dataset of audio features")

    parser.add_argument(
        "-n",
        "--name",
        help="Name of audio training dataset, e.g solo",
        dest="dataset_name",
        choices=["solo", "fast"],
        required=True
    )

    args = parser.parse_args()
    name = args.dataset_name
    
    #Setting up the paths
    dataset_path = ROOT_DATASET_PATH.joinpath(name)
    export_path = ROOT_DATASET_PATH.joinpath(f'{name}_ftrs.pickle')

    #Audio feature extraction process
    create_features_dataset(dataset_path, export_path)

    print(f'Finished in {format_time(time.time() - start_time)}')

if __name__ == '__main__':
    main()