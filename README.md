# hmm_speech_recognition

### 0. Setup Environment
This demo project is running on python2.x, please install the following required packages as well:
- scikits.talkbox: Calculation of MFCC features on audio 
- hmmlearn: Hidden Markov Models in Python, with scikit-learn like API	
- scipy: Fundamental library for scientific computing

All the three python packages can be installed via `pip install`, on Python3.x, the package `scikits.talkbox` can't be installed correctly for me.


### 1. Description
#### 1.1 Problem
By utilizing the `GMMHMM` in `hmmlearn`, we try to model the audio files in 10 categories. `GMMHMM` model provides easy interface to train a HMM model and to evaluate the score on test set.

Please more details in the [doc](https://hmmlearn.readthedocs.io/en/latest/api.html#hmmlearn-hmm) of `hmmlearn`.
#### 1.2 Dataset
It's a demo project for simple isolated speech word recognition. There are only 100 audio files with extention of `.wav` for training, and 10 audio files for testing. To be more specified:

- Train: For digits 0-9, each with 10 sampels with Chinese pronunciation
- Test:  For digits 0-9, each with 1 sample with Chinese pronunciation

You can add some samples which the length is shorter than 1 sec in training data or testing data if you want.

#### 1.3 Demo running results:
In python2.x, run the script `hmm.py`, get the result below:
```
finish preparing the training data
finish training model
finish preparing the testing data
test on label  10 , predict result:  5
test on label  1 , predict result:  1
test on label  3 , predict result:  3
test on label  2 , predict result:  2
test on label  5 , predict result:  5
test on label  4 , predict result:  4
test on label  7 , predict result:  7
test on label  6 , predict result:  6
test on label  9 , predict result:  9
test on label  8 , predict result:  2
Final recognition rate is 80.00 %

```

### 2. Tricky bug workaround in hmmlearn
In the training of HMM model, there may lead to negative value of `startprob_`. Please refer to https://github.com/wblgers/hmm_speech_recognition_demo?fbclid=IwAR3Izc1oVjRoLEFMgRyyS6oFRw9h2Rz6G3cWVry8-oyofEDHTXw_NtFRR1s point 2 to get rid of this problem.