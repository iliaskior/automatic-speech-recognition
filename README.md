# automatic-speech-recognition

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

For running a full speaker indetification experiment and storing a trained model
Use the `flows/train_speaker_identification.py`

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

First you have to extract the audio features for FAST dataset
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
  
- The confusion matrix (normalized and not normalized) 
- The results of the experiment in pickle file. You can find it in
 ~/output/step2_results.pickle
 - A dataset which contains info about the source of each audio file of the dataset and the extracted mean & median fundamental frequency of it in a pickle file. You can find it in ~/output/step3_dataset.pickle
 - The confusion matrix (normalized and not normalized)

