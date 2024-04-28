# Project 2: Recurrent Neural Networks
Speech commands classification with recurrent neural networks

## Dataset
[Speech Commands Dataset](https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/data)

### Classes
yes, no, up, down, left, right, on, off, stop, go, unknown, silence

## Usage
### Initialization
Clone this repository and install required packages using `pip install -r requirements.txt`.
Whenever you run a Python script, ensure that you are in `src` directory.

### Downloading data
If you need to access the data stored on W&B directly, you can download it using
`python download_data.py <dataset name>`. It will return the path to the downloaded data.

You can check the name of dataset on W&B website. The raw data with `.wav` files is called
`speech-raw`, so you to download it, you should run `python download_data.py speech-raw`.
