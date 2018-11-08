















## **ANLY-590 Neural Nets & Deep Learning**

Wei Liu, Yinzhi Xi, Jie He

<br />

### *The Piano Boy*

A deep neural network that can compose some piano music by learning from pinao music datasets contains Beethoven, Mozart, Bach and other famous composers' works using some deep learning methods. 

<br />

### Project Proposal

<div style="page-break-after: always;"></div>



## 1. Team

Our team is consist of three students.

Team member:

Wei Liu, Yinzhi Xi, Jie He

<br />

## 2. Project’s Goal and Objectives

#### 1) Background

As many composers developed famous and beautiful piano music, we want to  see that if we could use some deep learning tools to develop some classical music just like a composer.



#### 2) Goal

With the deep learning methods we learnt in class, our goal of this project is to build a deep neural network that can compose some piano music by learning from a ton of music from Beethoven, Mozart, Bach and other famous composers. 

What we plan to do:

• develop a novel neural network architecture, 

• train various configurations of the neural network on a dataset of piano music, 

• test the different generated music with different styles of piano music inputs, such as Classical, Jazz, etc.,

• generate a collection of musical segments for the various network configurations,

• survey a sample of peers to quantitatively assess the effectiveness of the various configurations.

What we want to achieve:

• a deep neural network that can compose some piano music with good performance (that is: making it difficult for listeners to distinguish between real-world music and generated music)



#### 3) Improvements compared with similar ideas

Ideas to be compared:

a. Deep Jammer: <https://github.com/justinsvegliato/deep-jammer>

Deep Jammer is a deep neural network that can endlessly compose classical piano music by learning from a ton of music from Bach, Beethoven, Mozart, and other famous composers. This model generates music with Theano library.

b. Magenta: https://github.com/tensorflow/magenta

Magenta is a research project exploring the role of machine learning in the process of creating art and music. Primarily this involves developing new deep learning and reinforcement learning algorithms for generating songs, images, drawings, and other materials.

c. https://cs224d.stanford.edu/reports/allenh.pdf

The goal is to be able to build a generative model from a deep neural network architecture to try to create music that has both harmony and melody and is passable as music composed by humans.



**Improvements:**

a. We can try to increase the time steps from 128 to bigger value so that our model may capture more information in the music.

b. We may test if convolutional layers can capture notes relations.

c. We are going to train our model with larger dataset than others similar ideas.

We can generate music without using random segments in the training set, but use a segment that in some specific music style, and we can see if the generated music can be of the same style. For example, we input a segment of Jazz music, the generated music is also in style of Jazz.

d. We may use Keras to train the model and we want to achieve good performance with this framework.



#### 4) Why we choose this idea

First of all, the idea of developing classical music is very creative. It is different from the common deep learning models, such as classifying images, etc., but with sounds as the input and output. In addition, we have knowledge of image classification, or text classification before, but we have not done related work with music, so this idea is very challenging for us.

<br />

## 3. Data

#### 1) Data sources

We will use 3 DataSets, listed as follows:

DataSet1: ~8.5MB 327 pieces of piano music in MIDI format

A music-generate project named Deep Jammer has used this dataset, and we will use this project as comparison later.

<https://github.com/justinsvegliato/deep-jammer/tree/master/theano-model/pieces>

DataSet2: ~4.2GB 174,154 multitrack pianorolls in MIDI format

we may need to select some more suitable pieces from this dataset.

<https://github.com/salu133445/lakh-pianoroll-dataset/blob/master/docs/dataset.md>

DataSet3: a website provides more than 20 famous composers’ works in MIDI format, including hundreds of pieces, ~7MB in total.

<http://www.piano-midi.de/>



#### 2) Data dimensions and if appropriate dictionary/schema

Our datasets contains many pieces of MIDI format music, which is the international standard of digital music. It defines the way music programs in computers, synthesizers and other electronic devices exchange information and uses electronic signals to solve the problem of incompatibility between different electronic musical instruments. A MIDI file contains notes, timing, and performance definitions for up to 16 channels. The file includes the note information for each channel: key channel number, length, volume, and velocity.

For our model, the input will be segments of piano music in MIDI format. 

For programming understanding, we can obtain some information from the MIDI files and transfer them to digital information.

After transformation, the input of our model may contain a matrix of 128 time steps, 88 notes and 78 attributes, these information comes from MIDI format files. Each attribute is defined below:

Position: The piano key position. 

Pitch Class: A categorical array of pitches. 

Vicinity: An array of neighboring note states. 

Beat: The location in a measure.

Because we can extract this useful information, we think that the data set we choose is an appropriate schema.



#### 3) Possible limitations of the data

As we are going to train with piano music datasets, the style of these music may only contains Classical and Jazz, so our generated music is probably like this. If we can train more music with various styles of music, our model may be more general,  and it may produce music in more styles.



#### 4) Areas where you could improve the collection of future data

We can try to absorb more music styles. We can try to use Lakh Dataset, which contains various kinds of music in MIDI format. The link is: <https://colinraffel.com/projects/lmd/>



#### 5) Why the current data source is appropriate

At first, the DataSet1 has been used by other researchers before, so we can reuse it and there is no need to collect data by ourselves. Secondly, the DataSet3 contains most of the famous composers’ works, which is very appropriate for our project. If the 2 datasets are not sufficient enough, we can use DataSet2 as supplements. Besides, these datasets contains more than thousands of pieces of music with reasonable lengths and all in MIDI format, so the current data source is appropriate for our model.



#### 6) If your team is working on a supervised problem, what is your input feature vector and output?

For our model, **the input** will be segments of piano music in MIDI format. 

We can obtain some information from the MIDI files and transfer them to digital information, including a matrix of 128 time steps, 88 notes and 78 attributes, these information comes from MIDI format files. Each attribute is defined below:

Position: The piano key position. 

Pitch Class: A categorical array of pitches. 

Vicinity: An array of neighboring note states. 

Beat: The location in a measure.



**The output** is a matrix of 88 notes and 2 predictions: 

Play Probability: The probability of the note being played.

Articulate Probability: The probability of the note being held.



Our model will output a following note, then we can repeatedly run the model with adding the output to inputs and it can finally generate a long piece of music(ideally, it can be endless).

<br />

## 4. Assessment Metrics

#### 1) What loss metric you will be using and why?

We will use negative log-likelihood.

The output is the possibility of whether or not 88 notes are played and thus this is a classification problem. So log-likelihood metric can be used for our problem.



#### 2) What baseline datasets will you be using to evaluate your model’s performance?

Since our model is to generate new music, there are not such baseline datasets which can be used to evaluate model performance. In order to evaluate our model’s performance, we can make some audiences listen to some generated music mixed with real-world music, and let them to rate these music. In this way, we can compare these ratings based on different datasets.



#### 3) What other models are used as baselines - how do you expect your approach to compare?

We have found a similar project named Deep Jammer, which has a result. And we can use it as the baseline model. In order to compare our model’s performance with theirs, we can make some audiences listen to some generated music by us mixed with real-world music and some generated music by Deep Jammer, and let them to rate these music. We can have a compare result based on the ratings.



#### 4) What is considered state of the art in the field and how does it compare to your method?

Recurrent Neural Networks combined with Restricted Boltzmann Machines and other similar recurrent energy based models can be regarded as the most common methods in this field since a lot of work has been done based on them. Our method is to only use deep neural networks.

<br />

## 5. Approach				

#### 1) At a high level what are the expected outcomes of your project 

– What approach will you be taking and why?

– Describe possible limitations of your approach

Our model is used to generate music and then we distribute the music segments to people for 

evaluation purpose. Literally, how good is our generated music? We can provide some real-world music segments and generated segments to some listeners, and then compare the scores between them.

The limitation would be the evaluation is too subjective, but we are doing a project related to art and the aesthetic of art is subjective indeed.



#### 2) Where are you going to train your model?

– Will you be using a cloud provider or running it locally?

We have an 1080 GPU locally, so we may try it locally at first. And we know that Google Cloud provides some credits to student and we can run it remotely if it is necessary.

– What are some limitations if running it on the cloud vs locally?

Cloud: may need to spend some money if credits are not sufficient. 

Locally: we are not sure if the GPU has enough performance to do our task.



#### 3) What API will you use to train your model (i.e. Tensorflow, Keras, PyTorch)

We will try Keras to train our model.

<br />