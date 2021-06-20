Title:  TUT Sound events 2017, Development dataset

TUT Sound events 2017, Development dataset
==========================================
[Audio Research Group / Tampere University of Technology](http://arg.cs.tut.fi/)

Authors
- Toni Heittola (<toni.heittola@tut.fi>, <http://www.cs.tut.fi/~heittolt/>)
- Annamaria Mesaros (<annamaria.mesaros@tut.fi>, <http://www.cs.tut.fi/~mesaros/>)
- Tuomas Virtanen (<tuomas.virtanen@tut.fi>, <http://www.cs.tut.fi/~tuomasv/>)

Recording and annotation
- Eemi Fagerlund
- Aku Hiltunen

# Table of Contents
1. [Dataset](#1-dataset)
2. [Usage](#2-usage)
3. [Changelog](#3-changelog)
4. [License](#4-license)

1. Dataset
=================================
TUT Sound events 2017, development dataset consists of 24 audio recordings from a single acoustic scene: 

- Street (outdoor), totaling 1:32:08

The dataset was collected in Finland by Tampere University of Technology between 06/2015 - 01/2016. The data collection has received funding from the European Research Council under the ERC Grant Agreement 637422 EVERYSOUND.

[![ERC](https://erc.europa.eu/sites/default/files/content/erc_banner-horizontal.jpg "ERC")](https://erc.europa.eu/)

### Preparation of the dataset

The recordings were captured each in a different location (different streets). The equipment used for recording consists of a binaural [Soundman OKM II Klassik/studio A3](http://www.soundman.de/en/products/) electret in-ear microphone and a [Roland Edirol R-09](http://www.rolandus.com/products/r-09/) wave recorder using 44.1 kHz sampling rate and 24 bit resolution. 

For audio material recorded in private places, written consent was obtained from all people involved. Material recorded in public places (residential area) does not require such consent.

Individual sound events in each recording were annotated by a research assistant using freely chosen labels for sounds. The annotator was trained first on few example recordings. He was instructed to annotate all audible sound events, and choose event labels freely. This resulted in a large set of raw labels. Mapping of the raw labels was performed, merging sounds into classes described by their source before selecting target classes. Target sound event classes for the dataset were selected based on the frequency of the obtained labels, resulting in selection of most common sounds for the street acoustic scene, in sufficient numbers for learning acoustic models. Mapping of the raw labels was performed, merging sounds into classes described by their source, for example "car passing by", "car engine running", "car idling", etc into "car", sounds produced by buses and trucks into "large vehicle", "children yelling" and " children talking" into "children", etc.

Due to the high level of subjectivity inherent to the annotation process, a verification of the reference annotation was done using these mapped classes. Three persons (other than the annotator) listened to each audio segment annotated as belonging to one of these classes, marking agreement about the presence of the indicated sound within the segment. Agreement/disagreement did not take into account the sound event onset and offset, only the presence of the sound event within the annotated segment. Event instances that were confirmed by at least one person were kept, resulting in elimination of about 10% of the original event instances in the development set. 
 
The original metadata file is available in the directory `non_verified`. 

The ground truth is provided as a list of the sound events present in the recording, with annotated onset and offset for each sound instance. Annotations with only targeted sound events classes are in the directory `meta`. 

### File structure

```
dataset root
│   README.md				this file, markdown-format
│   README.html				this file, html-format
│   EULA.pdf				End user license agreement
│   meta.txt				meta data, csv-format, [audio file][tab][scene label][tab][event onset][tab][event offset][tab][event label][tab][event type]
│
└───audio					22 audio files, 24-bit 44.1kHz
│   │
│   └───street				acoustic scene label
│       │   a001.wav		name format: [original_recording_identifier].wav
│       │   a003.wav
│           ...

│
└───evaluation_setup		cross-validation setup, 4 folds
│   │   street_fold1_train.txt		training file list, csv-format: [audio file (string)][tab][scene label (string)][tab][event onset (float)][tab][event offset (float)][tab][event label (string)]
│   │   street_fold1_test.txt 		testing file list, csv-format: [audio file (string)][tab][scene label (string)]
│   │   street_fold1_evaluate.txt 	evaluation file list, fold1_test.txt with added ground truth, csv-format: [audio file (string)][tab][scene label (string)][tab][event onset (float)][tab][event offset (float)][tab][event label (string)]
│   │   ...        
│
└───meta					meta data, only with target sound classes
│   │
│   └───street				acoustic scene label
│       │   a001.ann		annotation data, csv-format (can be imported to audacity): [event onset (float)][tab][event offset (float)][tab][event label (string)]
│        │   a003.ann
│            ...
│
└───non_verified			non-verified raw meta data
	│ 
	└───meta				meta data, only with target sound classes
	│ 
	└───evaluation_setup	cross-validation setup, 4 folds
	meta.txt				meta data, csv-format, [audio file][tab][scene label][tab][event onset][tab][event offset][tab][event label][tab][event type]

```

### Event statistics

The sound event instance counts for the dataset are shown below. 

** Street **


| Event label           | Verified set, Event count  | Non-verified set, Event count      |
|-----------------------|----------------------------|------------------------------------|
| brakes squeaking      | 52                         | 59                                 |
| car                   | 304                        | 304                                |
| children              | 44                         | 58                                 |
| large vehicle         | 61                         | 61                                 |
| people speaking       | 89                         | 117                                |
| people walking        | 109                        | 130                                | 
| **Total**             | **659**                    | **729**                            |

2. Usage
=================================

Partitioning of data into **development dataset** and **evaluation dataset** was done based on the amount of examples available for each event class, while also taking into account recording location. Ideally the subsets should have the same amount of data for each class, or at least the same relative amount, such as a 70-30% split. Because the event instances belonging to different classes are distributed unevenly within the recordings, the partitioning of individual classes can be controlled only to a certain extent. 

The split condition was relaxed so that 65-75% of instances of each class were selected into the development set.  

Evaluation dataset is provided separately.

### Cross-validation setup

A cross-validation setup is provided in order to make results reported with this dataset uniform. The setup consists of four folds, so that each recording is used exactly once as test data. At this stage the only condition imposed was that the test subset does not contain classes unavailable in training. 

The folds are provided with the dataset in the directory `evaluation_setup`. 

#### Training

`evaluation setup\[scene_label]_fold[1-4]_train.txt`
: training file list (in csv-format)

Format:

    [audio file (string)][tab][scene label (string)][tab][event onset (float)][tab][event offset (float)][tab][event label (string)]

#### Testing

`evaluation setup\[scene_label]_fold[1-4]_test.txt`
: testing file list (in csv-format)

Format:

    [audio file (string)][tab][scene label (string)]

#### Evaluating

`evaluation setup\[scene_label]_fold[1-4]_evaluate.txt`
: evaluation file list (in csv-format), same as fold[1-4]_test.txt but added with ground truth information. These two files are provided separately to prevent contamination with ground truth when testing the system. 

Format: 

    [audio file (string)][tab][scene label][tab][event onset (float)][tab][event offset (float)][tab][event label (string)]
 
3. Changelog
=================================
#### 2.0 / 2017-06-20
* Verified meta data

#### 1.0 / 2017-03-20
* Initial commit

4. License
=================================

See file [EULA.pdf](EULA.pdf)