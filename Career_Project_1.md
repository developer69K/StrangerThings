## Project [ML Interview Practice]

## Question 1
```
In the A/B Testing,
    P(Liking Page A) = 0.20
    P(Liking Page B) = 0.2143

This statistic points that B is the better choice with the given data available,
but I am not yet confident that B is a better choice than A
with a difference between their % of liking so small,
I would rather consider a bigger confidence interval between them,
may be 20\% higher ie, B being higher than A by 20% or A being higher than B by 20%.
```

## Question 2
```
Since there is no restriction as to how to categorize the users,  
I would like to use the following categories for this case, Sports, Entertainment, Food, Health and Technology.

A list that would represent the above will be a 5 element list.
We use a word list of the topics and increment the count for each topic if the word appears for that tweet
For example, if the tweet would be something like,  
"I love Bryan Adams!!" , This contains the word "Bryan" and "Adams",
which will be part of the word list for the topic "Entertainment",
Hence this will output a distribution [0,0,1,0,0]

Now the Question is to categorize the words I would need to find the relation
of the words with that of the category, that is where the word2vec model will be helpful
My idea would be to do a word2vec on my text and check the cosine distance,
of some of the important/main words with the category word occurring per tweet per userid
and hence I will be able to categorize the userid's based on the tweet. I can fill up
the distribution matrix based on these similarities and then each userid will be assigned a
category to be printed out

I will train my model with the corpus to generate
```
## Describing the Whole Process
+ As there is a stream of Tweets coming in, I would assume there is a generator of sort, and I would need to store the Tweets in row-wise manner in a database file so that I can have corpus to train my model
+ I have to choose how much data I need to train and validate , so I accordingly I will feed a part of the text corpus as my input to the word2vec model.
+ The word2vec model will output a vector representation of words, called "word embeddings"
+ For Each tweet I will remove the punctuations and the stopwords/common words so that the important or uncommon words can be picked up for training
+ After training we can calculate the cosine distance of the priority words for each tweet with respect to the categories and thus can categorize them
  + Each tweet can have more than one priority word, In that case I will use the category lowest cosine distance
  + I have described an example below

  ```
  "user_id":11,
  "tweet":"I love Basketball games in LosAngeles!"

  After Preprocessing the final dataframe and the list of words will look something like
  [Sports, Entertainment, Food, Health and Technology]
  words = ["love", "Basketball", "games", "LosAngeles"]
  userid  tweet     Cosine_distance_from_categories
  11      love        [0.8, 0.7, 0.8, 0.9, 0.7]
  11      Basketball  [0.1, 0.7, 0.9, 0.6, 0.9]
  11      games       [0.1, 0.4, 0.8, 0.6, 0.7]
  11      LosAngeles  [0.4, 0.4, 0.6, 0.7, 0.8]

  Normalized Sum      [0.35, 0.55, 0.775, 0.7, 0.775] ==> This shows sports as the lowest distance and hence the userid "11" is put under the category "sports"
  ```
#### Some Preprocessing steps:
+ Code for counting words without the presence of any Punctuations

```(R)
library(dplyr)
library(janeaustenr)
library(tidytext)


text<-c("The world is flat!","There is no good in doing this","I love Basketball","There is something in the air", "where are you going sir?","Come on , lets do it!")
text_df<-data_frame(line=1:6, text=text)

tidyform<-text_df %>% unnest_tokens(word, text)

counts_<-tidyform %>% count(word, sort=TRUE)
```
+ Output:
```
> head(counts_,15)
# A tibble: 15 x 2
         word     n
        <chr> <int>
 1         is     3
 2         in     2
 3        the     2
 4      there     2
 5        air     1
 6        are     1
 7 basketball     1
 8       come     1
 9         do     1
10      doing     1
11       flat     1
12      going     1
13       good     1
14          i     1
15         it     1
```
## Question 3

+ In a classification problem, with labelled cases, if we are trying to fit a model not to over-fit
+ Rule of thumb is a good model should have a low training error and a low generalization error, there should always be a bias-variance tradeoff, low Bias and high Variance caused over-fitting
+ Some of the initial steps that we can take is not to make the model overly complex, for example for an image classification problem trying a simple 3-4 layers conv2D
  will be a good place to start with rather considering too many hidden layers and parameters
+ Some of the other process is introducing a validation set, or doing cross validation, precisely a k-fold CV, where you change the training and the validation sets, though a k-fold CV might be high on compute time
+ One of the steps that we can take is L2 regularization where we add more information(regularization term) so that the complexity of the model is reduced
+ Data augmentation techniques and Normalization also helps reduce overfitting in image classification problems

## Question 4

**Problem** I am tasked with is making a learning Agent, that learns for user's behavior while using a 3D modelling software based on click pattern and recommends changes
+ This is a simpler reinforcement Learning problem as I see it
+ My learning agent is created in a way such that for each state, I have to find what the next state can be and what will be the reward associated for going to the new state, At each step the agent will perform an action which leads to going to a new state(possibly) and receiving a reward, The goal of the agent is to define an optimal policy that maximizes the reward and hence the policy will be the decision for my agent to take       
+ I will define a State matrix of all the states that the user can be at based on a initial state
+ Initial State could be to Start a Model, Over the period of time I would need to collect the data for  determining the click pattern, to finalize what are the states that I could define, For example
```
File -> Project -> Load a Picture -> Rotate the picture (by a degree) ->
select a component (in the pic) -> Symmetrize -> Flip (by a degree) -> Save the model
```
+ **The Buttons used here**
  + File
  + Project
  + Load
  + Rotate
  + Select a Component
  + Symmetrize
  + Flip
  + Save
+ Our Training set will constitute of these states/buttons and the count of each states
+ Our Model will be based on the probability of going to the next state given in a certain state, such as
```
P(Load|Project) > P(Create|Project),
then the system will recommend the next state to be Load
This might be a prompt after start or a right click
```
+ I will start keeping rewards or penalties for reaching a new state, based on actual user responses to make the system learn.
+ As the system learns, the Policies will be optimized and the agent will start giving better suggestions
+ Based on the counter statistics we can also have a "Taskbar" on the software which will have the high hitters, for example:
  + Smoother
  + Rotate
  + Flip
  + Add Reference Planes
+ As the Software matures, I will thus have more data and will have stronger policies for my recommendation system, to recommend certain features, as taskbars, right clicks or prompts

## Question 5

Regularization is necessary for cases where we overfit easily to the training set, Anytime we try to fit the Noise along with the pattern we want to predict we will overfit.
+ The general cases where regularization is necessary is a supervised Learning setting, where we try to fit our models to predict such that we match the labels
+ We keep adding more features which results in addition of complexity and fits the training set better
+ Training set error decreases and test set error is higher
Regularization will not be ideal in cases of unsupervised learning, where there are no labels as such. Also for under-fitting scenarios, it is also not required

## Question 6

This problem is almost exactly similar to one of my Kaggle projects, it is called Instacart Buyer Prediction analysis.

**Problem description**
```

```
