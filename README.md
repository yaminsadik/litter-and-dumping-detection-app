
# Streamlit App: Illegal Dumping and litter detection app.
This Streamlit app uses a custom-trained object detection model to detect illegal dumping, garbage, litter, and other related waste from aerial images and videos. The app is designed for easy use and demonstration of the AI model's capabilities and live object detection using YOLO models.

# Installation
To run the app locally, you need to clone this repository and install the required dependencies.

`git clone https://github.com/yaminsadik/litter-and-dumping-detection-app`

`pip install -r requirements.txt`

# Usage
Once the dependencies are installed, you can start the Streamlit app with the following command:

`streamlit run app.py`

This will launch the app in your default web browser.

# Features
* Object Detection: Upload an aerial image or video, and the app will detect and highlight areas with illegal dumping or litter.
* Interactive Interface: Easy-to-use interface with real-time results.
* Configurable Settings: Modify detection parameters via a configuration file.

# Configutration
To adjust the app's settings, you can modify the config.py file. This file allows you to set paths to your model files, tweak detection thresholds, and customize other parameters.
By default, model can be set on the weights directory. 

# Contributing
Contributions are welcome! Please feel free to submit a Pull Request or open an issue to discuss your ideas or report any bugs.

