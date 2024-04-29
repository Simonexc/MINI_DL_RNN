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

### Training
You should use `single_run.py` to perform all training sessions. It takes one argument:
name of the YAML file located in `src/configs` directory, e.g. if you want to run trasnformer
training, you should execute `single_run.py transformer`.

This function ensures that all the required metrics are logged to W&B and the models' checkpoints
are saved.

#### Config YAML file
This file specifies the whole training process. The required attributes are:
- dataset <- name of the dataset to train on from W&B artifacts (always the latest version is used)
- model_name <- name of the model, can be any string, used for filtering of runs
- model_class <- should be the name of model class imported in `src/models/__init__.py`
- lr <- learning rate
- batch_size
- epochs
- model_params <- list of model parameters configured in this YAML, can be empty; if any parameter is add to this list,
it should also be added to the YAML

You can also specify custom feature processor (if required). The feature processor
function needs to be a [curried function](https://en.wikipedia.org/wiki/Currying) which doesn't
require any parameters (in the future we might need to add some parameters but for now it is not supported) and is located
in `src/dataset/feature_processors.py`.