# Dataset Toolbox

A command line tool for dataset preparation.

## Features

- [ ] TODO

## Installing dependencies

- pip3 install numpy opencv-python dlib imutils scikit-image pandas
- pip3 install dependencies/detector-wrapper-0.1dev.tar.gz

## Usage example

Automatic generation of ROIs related to face location in dataset's images:

1. Visualization of ROIs:
    - python3 dataset_toolbox.py -d /path/to/database/classname/ -p -m models/face.prototxt models/face.caffemodel -t CvCaffe -vd
2. Save ROIs on disk, and split the data into training, validation and test sets:
    - python3 dataset_toolbox.py -d /path/to/database/classname/ -p -m models/face.prototxt models/face.caffemodel -t CvCaffe -dt -pf classname_cvcaffe_ -o /path/to/output/database/classname
3. OBS: Edit the implementation of the "postprocess_roi" function to change ROI position and shape.
