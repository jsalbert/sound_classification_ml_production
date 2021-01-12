# UrbanSound8k 

In this repository you will find an end to end tutorial of a simple example of machine learning in production.
We will perform data exploration and analysis, followed by the training and evaluation of a machine learning model, and finally we will deploy an app on AWS using Docker. 

The repository contains:

- Colab notebooks with code to perform data exploration and train a machine learning model on the UrbanSound8k dataset 
- Instructions to create your own docker images and upload them to AWS Beanstalk. 

## Data Exploration

Observe the dataset statistics and visualize the data: 

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jsalbert/urban_sound8k_deep_learning/blob/main/notebooks/UrbanSound8k_data_exploration.ipynb)


<table style="width:100%">
  <tr>
    <th>Audio Files Waveform</th>
    <th>Audio Files Mel-Spectogram</th>
  </tr>
  <tr>
    <td><img src="https://github.com/jsalbert/urban_sound8k_deep_learning/blob/main/images/urban_sound_waveforms.png?raw=true" alt="fig_waveforms" width="500"/></td>
    <td><img src="https://github.com/jsalbert/urban_sound8k_deep_learning/blob/main/images/urban_sound_spectograms.png?raw=true" alt="fig_spectograms" width="500"/></td>
  </tr>
</table>

## Machine Learning Modelling

Train a Convolutional Neural Network to classify the audio samples: 

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jsalbert/urban_sound8k_deep_learning/blob/main/notebooks/UrbanSound8k_machine_learning.ipynb)


<img src="https://github.com/jsalbert/urban_sound8k_deep_learning/blob/main/images/urban_sound_confusion_matrix.png?raw=true" alt="fig_confusion" width="500"/>

