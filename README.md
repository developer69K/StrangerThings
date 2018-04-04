# StrangerThings
+ Some Strange Projects involving Machine Learning Concepts and Papers
+ Structured Approach to put all my work in one Repo

# resources
+ https://www.amazon.com/Everybody-Lies-Internet-About-Really/dp/0062390856/ref=sr_1_1?ie=UTF8&qid=1522440665&sr=8-1&keywords=everybody+lies+big+dataaiindex.org
+ The Master Algorithm - Pedro Domingos
  - LEAPN []
+ Here is a link to that book: https://www.amazon.com/Surfaces-Essences-Analogy-Fuel-Thinking/dp/0465018475
  Link: https://playground.tensorflow.org
+ Another book referenced in the video - https://www.amazon.com/Master-Algorithm-Ultimate-Learning-Machine-ebook/dp/B012271YB2/
+ Blockchain , NLP , Driverless AI
+ CNN - http://cs231n.stanford.edu/syllabus.html
+ Latex - https://www.sharelatex.com/learn/Inserting_Images



## GANs
 + Generative Adversarial Networks
  - Introduction to GANs

  - **Mini-Project** - IMDB Keras - IMDB project using keras
  - Keras - Testing Keras module
  - **Project-1** - First Neural Network Design Project
  - **Project-2** - Dog Breed Classifier Project
  - **Mini-Project** - Sentiment Analysis Project by Trask
  - **Mini-Project** - Student Admissions - Keras Project
  + Study of Recurrent neural Networks
     - Recurrent Neural Networks
     - Long Short term Memory
     - Back Propagation through Time (BPTT)
       + Folded Networks to understand better
  + Practicing Tensor Flow
   - Training a RNN Network
  + **Mini-Project** - Sentiment Analysis using RNN/LSTMs
  + **Mini-Project** - Implementing a SkipGram Model using Tensor Flow
  + **Mini-Project** - Training a RNN/LSTM to predict the next word
  + **Project-3** - Generate TV Scripts
    - Use LSTM/RNNs to train a Model and Generate a TV script using Simpson's Dataset
    - Next step will be to improve the Training Loss and use the whole Dataset

## SmartCab
 + How to train a Naive smartcab training agent
 + Used Q-learning with multiple exploration factors to check where the Algorithm is converging
 + Results from the Project

|Attempt|	Epsilon	|Alpha	|Tolerance	|Safety	|Reliability	|n_test|
|-------|---------|-------|-----------|-------|-------------|------|
|1|	epsilon*=epsilon(0.98)|	0.5|	0.000001|	A+|	B|	40|
|2|	epsilon*=epsilon(0.98)|	0.5|	0.0001  |	A+|	A|	40|
|3|	epsilon*=epsilon(0.98)|	0.5|	0.001   |	A+|	B|	20|
|4|	epsilon=e^{-(0.05t)} |	  0.5|	0.01    |	A+|	B|	20|
|5|	epsilon=e^{-(0.005t)}|	0.5  |	0.01    |	A+|	B|	20|

**I choose the 2nd one from the top table**  

## RL
+ Re-enforcement Learning
+ Book - Reinforcement Learning: An Introduction

## Udacity Capstone

### Writing the Project Proposal
  + **Getting to know some topics from**
    - https://devpost.com
    - http://kaggle.com
  + **Topics**
    - DataScience Bowl - Identify a Nuclei in a dataset of Images
      + Do some exploratory data analysis , learn a bit about the dataset
      + Timeline - merger deadline: April 11th, 2018
      + Reading and thoughts :
        - U-Net: Convolutional Networks for Biomedical Image Segmentation : https://arxiv.org/pdf/1505.04597.pdf
        - Check other architectures that are comparable to Unet and why they perform better or lower
        - First step : Solve MNIST using tf
        - Second step : Do Analysis on the data
        - Write ups as you go
  + **How to write a Project Proposa**l:
      - https://github.com/udacity/machine-learning/blob/master/projects/capstone/capstone_proposal_template.md
      - rubric : https://review.udacity.com/#!/rubrics/410/view
      - **Project Proposal Submission**
      ```
      In this capstone project proposal, prior to completing the following Capstone Project, you you will leverage what you’ve learned throughout the Nanodegree program to author a proposal for solving a problem of your choice by applying machine learning algorithms and techniques. A project proposal encompasses seven key points:

      The project's domain background — the field of research where the project is derived;
      A problem statement — a problem being investigated for which a solution will be defined;
      The datasets and inputs — data or inputs being used for the problem;
      A solution statement — a the solution proposed for the problem given;
      A benchmark model — some simple or historical model or result to compare the defined solution to;
      A set of evaluation metrics — functional representations for how the solution can be measured;
      An outline of the project design — how the solution will be developed and results obtained.
      Think about a technical field or domain that you are passionate about, such as robotics, virtual reality, finance, natural language processing, or even artificial intelligence (the possibilities are endless!). Then, choose an existing problem within that domain that you are interested in which you could solve by applying machine learning algorithms and techniques. Be sure that you have collected all of the resources needed (such as datasets, inputs, and research) to complete this project, and make the appropriate citations wherever necessary in your proposal.
      ```
      - In addition, you may find a technical domain (along with the problem and dataset) as competitions on platforms such as Kaggle, or Devpost. This can be helpful for discovering a particular problem you may be interested in solving as an alternative to the suggested problem areas above. In many cases, some of the requirements for the capstone proposal are already defined for you when choosing from these platforms.

      - **Evaluation**

        Your project will be reviewed by a Udacity reviewer against the Capstone Project Proposal rubric. Be sure to review this rubric thoroughly and self-evaluate your project before submission. All criteria found in the rubric must be meeting specifications for you to pass.

      - **Submission Files**

        ```
        At minimum, your submission will be required to have the following files listed below. If your submission method of choice is uploading an archive (*.zip), please take into consideration the total file size. You will need to include

        A project proposal, in PDF format only, with the name proposal.pdf, addressing each of the seven key points of a proposal. The recommended page length for a proposal is approximately two to three pages.
        Any additional supporting material such as datasets, images, or input files that are necessary for your project and proposal. If these files are too large and you are uploading your submission, instead provide appropriate means of acquiring the necessary files in an included README.md file.
        Once you have collected these files and reviewed the project rubric, proceed to the project submission page.
        ```
