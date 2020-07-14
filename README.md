# Computer Pointer Controller

The Computer Pointer Controller is an AI-driven app that controls the position of the mouse pointer on the screen with the gaze of the person in front of the screen. 

## Project Set Up and Installation

### Prerequisites

This project requires a local OpenVINO installation to run. It has been tested with the `2020.1` version, but it should run with more recent versions as well.

To install OpenVINO on your machine, follow the instructions for your operating system at the bottom of the [Official OpenVINO Overview](https://docs.openvinotoolkit.org/latest/index.html) page. 

### Virtual environment

As a first step, you should create a virtual environment to isolate the project from your local installation of Python. A detailed explanation can be found [here](https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/26/python-virtual-env/).

Then, you should install all the required libraries in the `requirements.txt` file.

TD;DR

```bash
virtualenv intelgaze
source ./intelgaze/bin/activate
pip install -r requirements.txt
```

### Start OpenVINO

Load the OpenVINO variables with the following command.

```bash
source /opt/intel/openvino/bin/setupvars.sh
```

### Download the models

The Gaze tracking app uses the following models from the OpenVINO Model Zoo:

- face-detection-adas-binary-0001
- head-pose-estimation-adas-0001
- gaze-estimation-adas-0002	
- landmarks-regression-retail-0009

You should download these models in the `models` folder. For more information on how to download an OpenVINO model locally, see the [OpenVINO Model Zoo](https://docs.openvinotoolkit.org/latest/omz_models_intel_index.html) page.

### Directory structure

The project's directory structure should look similar to this:

```
- README.md
- requirements.txt
- config.yml
- src
- bin
- models
  |- face-detection-adas-binary-0001
  |- head-pose-estimation-adas-0001
  |- gaze-estimation-adas-0002	
  |- landmarks-regression-retail-0009
```

The `models` folder should contain all the models you downloaded before.

## Running the model

To run the app, simply run this command:

```bash
python src/main.py --config config.yml
```

## Documentation

To run the program, you need a valid configuration file. An example of this is provided in the `config.yml` file. 

The format for the configuration is [YAML](https://www.tutorialspoint.com/yaml/index.htm), a lightweight and easy-to-read markdown language. 

To print a complete list of the parameters accepted by the configuration file, run the following command.

```bash
python src/main.py --print
```


## Benchmarks

### Loading times

Models with different accuracies have different loading times. The following table summarises the loading times.

|                     | FP32   | FP16   |
|---------------------|--------|--------|
| Face detection*     | 254 ms | N/A    |
| Landmarks detection | 252 ms | 129 ms |
| Headpose estimation | 140 ms | 113 ms |
| Gaze estimation     | 185 ms | 139 ms |

*The precision for the face detection model is FP32-INT1.

### Performance of the gaze estimation model

The inference times for the gaze estimation model vary according to the selected precision. The following table summarises the main results. 

| Precision | Inference time |
|-----------|----------------|
| FP16-INT8 | 1.95 ms        |
| FP16      | 2.11 ms        |
| FP32      | 2.18 ms        |

The results have been obtained as an average of the inference times over 100 inferences.

## Results

A lower accuracy of the models generally allows for higher performances. As can be seen from the previous table, running the gaze estimation model at FP16-INT8 precision reduces the inference time by almost 12\% with respect to FP32. 

### Moving the mouse

It is interesting to compare the inference times with the time spent running the rest of the code. 

A simple experiment shows that most part of the time (about 113 ms) is spent during the call to the `mouse.move()` function. At least on MacOS, this is the most expensive line of the main loop of the code. 

### Edge Cases

In certain cases, a face might not be present in the image. When the app cannot find a valid face in the image, it gives a warning and exits the program.

When multiple faces are present, only the one with the highest score is retained and processed. 
