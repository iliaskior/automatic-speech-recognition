# automatic-speech-recognition

## Installation
### Conda enviroment


Create an empty conda enviroment with Python 3.9
```
$ conda create -n <name> python=3.9
```
Activate the conda environment
```
conda activate <name>
```
#### Manually
Install the required libraries
```
 $ conda install -c conda-forge librosa
 $ conda install -c anaconda pandas
 $ conda install -c anaconda seaborn
 $ conda install -c anaconda jupyter
 $ conda install -c conda-forge tensorflow
```
#### Through requirements.txt file
```
$ conda install --file requirements.txt
```

### Docker

## Dataset
You can find the corpus's website [here](https://chains.ucd.ie/corpus.php)

You can download the corpus from here [here](https://chains.ucd.ie/ftpaccess.php)

You just need to download the following parts of the corpus:

`Solo condition`
- solo.tar.7z.001 (100 Mb)
- solo.tar.7z.002 (100 Mb)
- solo.tar.7z.003 (100 Mb)
- solo.tar.7z.004 (100 Mb)
- solo.tar.7z.005 (75 Mb)

`Fast Speech`
- fast.tar.7z.001 (100 Mb)
- fast.tar.7z.002 (100 Mb)
- fast.tar.7z.003 (100 Mb)
- fast.tar.7z.004 (16 Mb) 

After downloading the above files:
1. Create an empty folder under the root path of the current cloned repository and name it `dataset` 
```
$ cd ~/automatic-speech-recognition
$ mkdir dataset
```
2. Uncompress the compressed files under dataset folder

3. After that you should see the following structure
```
/automatic-speech-recogntion
    /dataset
        /data
            /fast
            /solo
    /audio
    /flows
    /models
    /utils

```

## Step 1)

*Purpose:* Perform speaker identification in SOLO dataset

At first, we have to extract the audio features 
Use the `flows/audio_feature_extraction.py`  
Inputs:
- Full dataset root path: root directory where audio files stands   

Outputs:
- Extracts and stores the audio features for audio files in pickle file. You can find it in
~/dataset/data/{dataset_name}_ftrs.pickle

```
$ python3 ~/flows/audio_feature_extraction.py -i ~/audio_files_root_path
```

e.g For extracting the audio features for SOLO dataset run this:
```
$ python3 ~/flows/audio_feature_extraction.py -i ~/dataset/data/solo
```

Experimentation

For running a full speaker indetification experiment and storing a trained model use the `flows/train_speaker_identification.py`

Inputs:
- Features dataset path
- A flag for training-storing or not a model with all available data

Outputs:
- Training history plot for each fold (GroupKFold experiment)
- The aggregated confusion matrix (normalized and not normalized) from all the folds
- The aggregated results of the experiment in pickle file. You can find it in
 ~/output/step1_results_groupkfold.pickle
 - Trained model to all available data. You can find the exported model in ~/output/speaker_identification.h5

```
$ python3 ~/flows/train_speaker_identification.py -i ~/dataset/data/solo_ftrs.pickle -n yes
```

## Step 2)
*Purpose:* Perform speaker identification in FAST dataset, using a model trained only in SOLO dataset

First you have to extract the audio features for the FAST dataset
```
$ python3 ~/flows/audio_feature_extraction.py -i ~/dataset/data/fast
```
For performing speaker identification in FAST dataset using the pretrained model of step1:
Use the `flows/predict_speaker.py`

Inputs:
- Path to the extracted features (testing dataset)

Outputs:
- The confusion matrix (normalized and not normalized) 
- The results of the experiment in pickle file. You can find it in
 ~/output/step2_results.pickle


 ## Step 3)
 *Purpose:* Perform speaker identification using both solo and fast datasets without using gender labels. 

 Use the `flows/gender_classification.py`
 Inputs:
 - Full dataset root path: root directory where audio files stands 

 Outputs:
 - A dataset which contains info about the source of each audio file of the dataset and the extracted mean & median fundamental frequency of it in a pickle file. You can find it in ~/output/step3_dataset.pickle
 - The confusion matrix (normalized and not normalized)

 ```
$ python3 ~/flows/gender_classification.py -i ~/dataset/data
```

