# UrbanSound8k 

In this repository you will find an end to end tutorial of an example of machine learning in production.

The objective will be to create and deploy in the cloud a machine learning application able to recognize and classify different audio sounds. We will use the UrbanSound8k Dataset, which contains the following 10 sounds: Air conditioner, car horn, children playing, dog bark, drilling, engine idling, gun shot, jackhammer, 
siren, street music. 

During this tutorial we will perform data exploration and analysis, followed by the training and evaluation of a machine learning model, and finally we will deploy an app on AWS using Docker. 

The repository contains:

1. [A Colab notebook to perform Data Exploration and Analysis](https://github.com/jsalbert/urban_sound8k_deep_learning#data-exploration)
2. [A Colab notebook to perform Training and Evaluation of a Machine Learning Model](https://github.com/jsalbert/urban_sound8k_deep_learning#machine-learning)
3. [Instructions to create your own Flask app](https://github.com/jsalbert/urban_sound8k_deep_learning#creating-the-flask-app)
4. [Instructions to create a Docker image and upload it to AWS Beanstalk to share your app with the world](https://github.com/jsalbert/urban_sound8k_deep_learning/blob/main/README.md#creating-a-docker-image-and-uploading-to-aws-beanstalk)

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

## Machine Learning 

Train a Convolutional Neural Network to classify the audio samples: 

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jsalbert/urban_sound8k_deep_learning/blob/main/notebooks/UrbanSound8k_machine_learning.ipynb)


<img src="https://github.com/jsalbert/urban_sound8k_deep_learning/blob/main/images/urban_sound_confusion_matrix.png?raw=true" alt="fig_confusion" width="500"/>

## Creating the Flask App

## Creating a Docker Image and Uploading to AWS Beanstalk

