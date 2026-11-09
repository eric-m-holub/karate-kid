# Karate Kid

## Description

AI inference to infer famous motions from the film The Karate Kid.

## Setup

This code was written in 2021, so the dependencies are pretty old. I recommend you setup a virtual Python 3.8.10 environment, activate it, and then install the dependencies with:
```
pip install -r requirements.txt
```


## Usage

### Training

Modify the 'active_training_label' value in 'livedatacollection.py' to train each respective action:
| 0 | 1 | 2 | 3 | 4 |
|:-----|:------:|------:|
|No Action|Sand the Floor|Wax On / Wax Off|Paint the Fence|Paint the House|


Then run the following script to start training:
```
python3 livedatacollection.py
```


### Live Inference

```
python livedatainference.py
```


## Video Demo

Click [here](https://www.youtube.com/watch?v=A9JGNFzZ0ZA) to watch the video on YouTube.
