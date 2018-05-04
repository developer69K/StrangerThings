---
title: "Udacity Career Project"
author: "Sandeep Anand"
date: "May 2, 2018"
output: html_document
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = FALSE, warning=FALSE, message=FALSE)
```

## Question 1

In the A/B Testing, we see the below information given to us

    P(Liking Page A) = 0.20
    P(Liking Page B) = 0.2143

This statistic points that B is the better choice with the given data available,
but I am not yet confident that B is a better choice than A
with a difference between their % of liking so small,
I would rather consider a bigger confidence interval between them,
may be 20% higher ie, B being higher than A by 20% or A being higher than B by 20%.
Also test plan durations can affect the conversion rate, by conversion rate I
mean coming to the page and clicking the button

Let's see if we can do a Hypothesis testing for this, Hypothesis test is one of
the best way to validate the claim made by a population. But for this case, There
is the complication of when to stop the test, do we keep running the tests on Page A
and Page B till a particular Page is significantly better in terms of probability
than the other

Consider A to be the control set and B to be the test set

**H0 (Null Hypothesis)**

* p(B)=p(A)

**H1 (Alternate Hypothesis)**

* p(B)!=p(A)

To Prove for or against the Null hypothesis, we would need to check the p-value
A p-value is a number, that you get by running the hypothesis test on your data.

if p-value>0.05, then we do not have enough evidence against the Null Hypothesis
and we would go with the Null Hypothesis
if p-value<0.05, then we reject the Null hypothesis

We will start the test with a simulated random experiment for a large number of iterations
under the Null hypothesis, to check the respective p-values and thus determine which page
is better. There are two sets of people here, control set(page A) and test set(page B).  

Below the theory for the analysis :

One of the decisions is to determine the number of data points needed to get a statistically significant result. This is called **statistical power**. Power has an inverse trade-off with size. The smaller the change you want to detect or the increased confidence you want to have in the result, means you have to run a larger experiment.

As you increase the number of samples, the confidence interval moves closer to the mean and we can get a better idea as to if we can reject the Null hypotheis
```
  α=P(reject null | null true)
  β=P(fail to reject null | null false)

1−β is referred to as the sensitivity of the experiment, or statistical power. People often choose high sensitivity, typically around 80%.

For a small sample, α is low and β is high

For a small sample, α is low and β is high. For a large sample α remains the same but β goes down (i.e. sensitivity increases). A good online calculator for determing the number of samples is here. As you change one of the parameters, your sample size will change as well. For example:

If you increase the baseline click through probability (under 0.5) then this increases the standard error, and therefore, you need a higher number of samples
If you increase the practical significance level, you require a fewer number of samples since larger changes are easier to detect
If you increase the confidence level, you want to be more certain that you are rejecting the null. At the same sensivitiy, this would require increasing the number of samples
If you want to increase the sensitivity, you need to collect more samples

```
### Comparing two samples

For comparing two samples, we calculate the pooled **standard error**.
For e.g., suppose Xcont and Ncont are the control number of users that click, and the total number of users in the control group. Let Xexp and Nexp be the values for the experiment. The pooled probability is given by

$$p_{pool}=(Xcont+Xexp)/(Ncont+Ntest)$$
$$SE_{pool}=√(p_{pool}∗(1−p_{pool})∗(1/Ncont+1/Ntest))$$
$$diff=p(exp)-p(control)$$

$$H0:diff=0, where \space diff^∼N(0,SE_{pool})$$


$$diff>1.96∗SE_{pool} \space or \space diff< −1.96∗SE_{pool}$$ then we can reject the null hypothesis and state that our difference represents a statistically significant difference


## Question2

Since there is no restriction as to how to categorize the users, I would like to use the following categories for this case, Sports, Entertainment, Food, Health and Technology.

A list that would represent the above will be a 5 element list. We use a word list of the topics and increment the count for each topic if the word appears for that tweet
For example, if the tweet would be something like,  
**"I love Bryan Adams!!" , This contains the word "Bryan" and "Adams"**
which will be part of the word list for the topic "Entertainment",
Hence this will output a distribution [0,0,1,0,0]

Now the Question is to categorize the words I would need to find the relation of the words with that of the category, that is where the word2vec model will be helpful
My idea would be to do a word2vec on my text and check the cosine distance,of some of the important/main words with the category word occurring per tweet per userid
and hence I will be able to categorize the userid's based on the tweet. I can fill up the distribution matrix based on these similarities and then each userid will be assigned a
category to be printed out


### Describing the Whole Process

* As there is a stream of Tweets coming in, I would assume there is a generator of sort, and I would need to store the Tweets in row-wise manner in a database file so that I can have corpus to train my model
* I have to choose how much data I need to train and validate , so I accordingly I will feed a part of the text corpus as my input to the word2vec model.
* The word2vec model will output a vector representation of words, called "word embeddings"
* For Each tweet I will remove the punctuations and the stopwords/common words so that the important or uncommon words can be picked up for training
* After training we can calculate the cosine distance of the priority words for each tweet with respect to the categories and thus can categorize them
  * Each tweet can have more than one priority word, In that case I will use the category lowest cosine distance
  * I have described an example below

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

## Clustering Technique to be used

* I will be finally using K-Means Clustering to see if I can categorize the groups better after the initial Cosine distance calculations  
* I have already a idea of the clustering so this step will be to initialize the clusters based on the five groups that I have currently
* Based on the classified points, we recompute the group center by taking the mean of the euclidean distance
* Repeat with a set of Iterations untill the group centers do not change a lot.

#### Some Preprocessing steps are shown below

* Code for taking a json file as input and counting words without the presence of any Punctuations
* I am still in the process to complete this project, but these are the starting steps

```math
{
 "user_id": [1,2,3,4,5,6],
 "timestamp": ["2016-03-22_11-21-20","2016-03-22_11-31-20","2016-05-22_11-31-20","2018-03-22_11-31-20","2016-03-27_11-31-20","2016-08-22_11-31-20"],
 "tweet": ["It's #dinner-time!", "It's going to be Fun!!", "I love New York!", "What is wrong with this Game!", "We are going to Atlanta!", "Jack Nicolson!!"]
}
```

```{r q2Countwords, echo=TRUE, warning=FALSE}
library(dplyr)
library(janeaustenr)
library(tidytext)
library(rjson)
library(ggplot2)

res<-fromJSON(file = "C:/Public/Code/json/tw1.json")
res1<-as.data.frame(res)

#text<-c("The world is flat!","There is no good in doing this","I love Basketball","There is something in the air", "where are you going sir?","Come on , lets do it!")
text_df<-data_frame(line=1:6, text=as.character(res1$tweet))

tidyform<-text_df %>% unnest_tokens(word, text)
counts_<-tidyform %>% count(word, sort=TRUE)
print(head(counts_))

plot1<-ggplot(data = counts_, aes(counts_$word, counts_$n))+ geom_bar(stat="identity")
print(plot1)
```

## Question 3

### In a classification problem, with labelled cases, that is a supervised problem

### Detect Overfitting
* Overfitting can be detected if, the Test error is high and the training error is low
* Cross validation can detect overfitting by partitioning the data
* Also calculating the R-squared values will help detect overfitting, A difference in predicted R-squared and regular R-squared is a good measure to detect OVerfitting
  * Predicted R Squared, can be calculated using LOO (Leave One out) approach, where you leave a particular data point and check how the model predicts the data point
  * Repeat this for all the data points

### Prevent Overfitting
* Rule of thumb is a good model should have a low training error and a low generalization error, there should always be a bias-variance tradeoff, low Bias and high Variance caused over-fitting
* Some of the initial steps that we can take is not to make the model overly complex, for example for an image classification problem trying a simple 3-4 layers conv2D
  will be a good place to start with rather considering too many hidden layers and parameters
* Some of the other process is introducing a validation set, or doing cross validation, precisely a k-fold CV, where you change the training and the validation sets, though a k-fold CV might be high on compute time
* One of the steps that we can take is L2 regularization where we add more information(regularization term) so that the complexity of the model is reduced
* Data augmentation techniques and Normalization also helps reduce overfitting in image classification problems

## Question 4

**Problem**
I am tasked with making a learning Agent, that learns for user's behavior while using a 3D modelling software based on click pattern and recommends changes

* This is a simpler reinforcement Learning problem as I see it
* I Will be using Q-learning for this problem where our purpose will be to follow the below steps, Q-Learning is a model-Free RL algorithm based on Bellman Equations.
    + Action(A) - All possible moves/decisions that the agent can make here, agent is the learning agent that I have
    + State(S)  - Current situation returned by the environment
    + Reward(R) - The immidiate return send back from the environment to evaluate the last action of the agent
    + Policy - The strategy that I will use for the next action based on the current state
    + Value(v) - The expected long term return with with discount, as opposed to the short term Reward(R)
    + Q-value() - Q value is the parameter we need to find out and maximize as the aim of q-learning here, it takes an extra parameter Q(s,a), this refers to the long term return of the current state, taking action a using policy
      + There are two ways of updating the value , and these are the **policy iteration** and **value iteration**
* My learning agent is created in a way such that for each state, I have to find what the next state can be and what will be the reward associated for going to the new state, At each step the agent will perform an action which leads to going to a new state(possibly) and receiving a reward, The goal of the agent is to define an optimal policy that maximizes the reward and hence the policy will be the decision for my agent to take       
* I will define a State matrix of all the states that the user can be at based on a initial state
* Initial State could be to Start a Model, Over the period of time I would need to collect the data for determining the click pattern, to finalize what are the states that I could define, For example

```
File -> Project -> Load a Picture -> Rotate the picture (by a degree) ->select a component (in the pic) -> Symmetrize -> Flip (by a degree) -> Save the model
```
* **The Buttons used here**
    + File
    + Project
    + Load
    + Rotate
    + Select a Component
    + Symmetrize
    + Flip
    + Save
* Our Training set will constitute of these states/buttons and the count of each states
* Our Model will be based on the probability of going to the next state given in a certain state, such as
```
P(Load|Project) > P(Create|Project),
then the system will recommend the next state to be Load, There could be a prompt after start or a right click
```
* I will start keeping rewards or penalties for reaching a new state, based on actual user responses to make the system learn, Q learning is a model-Free Algorithm and learns using hit and trial
* As the system learns, the Policies will be optimized and the agent will start giving better suggestions
* Based on the counter statistics we can also have a "Taskbar" on the software which will have the high hitters, for example:
  * Smoother
  * Rotate
  * Flip
  * Add Reference Planes
* As the Software matures, I will thus have more data and will have stronger policies for my recommendation system, to recommend certain features, as taskbars, right clicks or prompts


## Question 5

Regularization is necessary for cases where we overfit easily to the training set, Anytime we try to fit the Noise along with the pattern we want to predict we will overfit.
* The general cases where regularization is necessary is a supervised Learning setting, where we try to fit our models to predict such that we match the labels
* We keep adding more features which results in addition of complexity and fits the training set better
* Training set error decreases and test set error is higher
Regularization will not be ideal in cases of unsupervised learning, where there are no labels as such. Also for under-fitting scenarios, it is also not required


## Question 6

**Problem description**
```
Given the purchase history of customers and catalog of store items, predict what the customer will purchase as the next order and hence give a coupon to the user
```
**Approach**

* I do not know about the Dataset specifics, so I will start with an unupervised learning approach rather than using a "Tree Based (Gradient Boosting) supervised method"
* Sending a coupon to the user is a very similar problem as building a recommendation system based on the user choices
    + A Simple System will be to recommend what was the most bought item by the Particular user, this is basically where we collect the counts, but this does not work all the time as the user might be needing different items at a different time, Also picking up general user pattern is not efficient and does not work most of the time, We would need a more personalized way to do this
* We can use classification Algorithms or recommendation systems for solving this

  **Idea if we use a content and habit based recommendation system to design a system that recommends a coupon**

  + If the user likes an Item earlier, the person might like a "similar" item again
  + Based on similarity of items, the model can also recommend
  + Priority based approach where, you determine the probability of a particular item is higher to be picked by a particular user than the other. This approach could be using Baye's theorem, which checks for a pattern in the user's choices. Based on the information of the previous orders and the frequency of a particular pattern, we can also use the n-gram method to predict unigrams and bigrams such as (P(chocolate|milk) or P(milk|cereal, oreo)) , and determine the next item in the persom's list. Though we have to remember that this approach is compute heavy as it requires to calculate the unigram and bigram probabilities of a large number of combinations
  + The compute process can be made efficient by the catalog of store items present also, as if the items that have high probability but are not available could be taken out.
  + The compute process can also be made better by using a feedback system by which a user can "like/dislike", thus rating the particular coupon that is sent to him. More the feedback the better the system becomes.
  + Just to clarify also, by using the probability approach we are somewhat doing a filtering as well, as we are making sure , certain items are more prioritized over the others for a particular user.

### Testing and evaluating the Recommender System  
  + I would basically use Hypothesis testing or A/B testing to test if the customer liked the coupon. There could be online coupon systems, where the user can click "Like/dislike" buttons to rate a certain coupon received, and with high number of samples we can determine wether the user prefers a certain set of coupons ie, a Null Hypothesis or reject the Null Hypothesis based on
  + We can make sure that the same user is not getting the same coupon from our system all the time
  + To calculate the accuracy of the data, we will not have any test data, so we can use cross validation, just divide the current data into k-folds and see how the **RS** does, and average over K runs
  + If the online system is up and running the Key Indicator is **Click rate**, The **click rate** tells the count  the user clicks "Like or Dislike"


# Question 7
* If I am hired as a Machine Learning Engineer, I see myself becoming more adept and experienced with large scale Machine Learning, I see myself in a role where I am learning as well as building new systems and working as a team in the process. I see myself as an asset to the company who helps the company grow in the field of Analytics and data-driven Marketing. I see myself in a position where I can learn and collaborate with passionate data scientists and Machine learning engineers.
* I see the Job is focused on use of R, Python and Scala, My focus for the next year would be to get better at these languages, my language of preference is python though I would improve of the other languages as per requirement. Besides my current knowledge, I would focus on developing my expertise in NLP and Text mining and learn better data mining procedures using Hadoop and Spark.  
* My Long Term career goals is to work on leading Machine learning techniques that relate directly to all aspects of Marketing, yield optimization and customer experience, starting from generating data analysis pipelines to acquire new customers and retain existing customers to improving user experience through consistent feedback that is possible through large scale Machine learning. In the long term I am also passionate about using  high-performance computing, distributed systems and applied math These are going to help add to my skill set as I already come from a similar background.
