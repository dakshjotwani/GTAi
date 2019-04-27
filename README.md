# GTAi: Self Driving Car in GTA V

A deep learning solution to a self driving car in GTA V. Currently, we use ResNet50 as a feature extractor, and a LSTM to get game controls from a sequence of frame embeddings.

## Updates

* April 23, 2019: First [video](https://youtu.be/G2as7jAU4LM) of GTAi working relatively well.

## Requirements

1. Windows 10. Since we needed windows to collect data from GTA V, we decided to do all our training and testing on Windows. We tried our best to keep most of the code platform agnostic, but we cannot guarantee that it works on other operating systems.

2. GTA V for data collection and visual evaluation

3. x360ce installed on GTA V for Xbox 360 controller emulation

4. Python libraries: Pytorch, keyboard, MSS, PyGame, numpy, vJoy, OpenCV

5. (Optional, Recommended) Nvidia GPU. We used a RTX 2080 for training and testing.

## Usage

Some strings and commented/uncommented lines will probably need to be changed to get the code to run.

Create the following directories in the root directory of this project:

1. ./models : This is where all models will be saved.
2. ./datasets/newdata : This is where all data collected will be saved.

Important scripts:

```bash
$ cd src
$ python ./frame_input_capture.py   # Data collection
$ python ./train_net.py             # Train CNN or LSTM by editing main() code appropriately
$ python ./tmp.py                   # Create embeddings for all frames using trained CNN. This is needed to train the LSTM.
$ python ./driver.py                # CNN only driver script
$ python ./temporal_driver.py       # CNN + LSTM driver script (locked framerate)
``` 