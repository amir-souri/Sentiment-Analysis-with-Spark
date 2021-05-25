**Sentiment Analysis with Spark**

Detecting emotion in language is one of most central and accessible tasks in Natural Language
Processing (NLP). The task is important for e-commerce and organisations in order to assess public
and consumer opinion and assess strategy accordingly. In this project I will use Amazon’s product
reviews to construct a simple sentiment analyser.

I will follow a simple method, in three stages:
1. Translate the review text into a vector of numbers (a so called word embedding)
2. Use the embeddings with known ratings to train a network (a multilayer perceptron classifier)
3. Use the perceptron to predict ratings for another set of reviews (validation)


1. Computing a word embedding model
A word embedding is an NLP model that attempts to capture the meaning of a word as a vector of real
numbers. Word embeddings are learned from large amounts of raw text data, such as a Wikipedia
dump or corpora scraped from the web. They are meant to represent distributional information of the
word in question, a representation of the word given its context.
I will use a precomputed set of vectors for 400 thousand English words made available by the
GLoVe project at Stanford. Get it at: https://nlp.stanford.edu/projects/glove/ (pre-trained vectors).
Each line in a GLoVe file consists of a word followed by vector of numbers. There are several files
with vectors of various sizes (50, 100, 200 and 300 numbers). Longer vectors can, in principle,
capture more information about the word, so they should give better precision of analysis. However,
it is not obviously clear, that this benefit is perceptible for the problem at hand, so feel free to
experiment with various files. Shorter embeddings will be faster to process (and it will be faster to
train a neural network using them).
I will perform the sentiment analysis for dumps of product reviews from Amazon.com. You can
get the data files here: http://jmcauley.ucsd.edu/data/amazon/links.html . The reviews are stored in JSON format.
I am interested in three fields from the JSON records: summary, reviewText, and overall. A
’summary’ is the title given to the product review. ’Overall’ is the rating given by a user. I will be
analyzing the summary and the text, trying to predict the overall rating. The ratings are on a 5 step
scale (from 1 to 5).
A simple way to capture meaning of a review is to take the average vector of its words. I first
tokenize the texts (turn them into words). Then I translate all the words in a reviewText and the
summary to vectors from the GLoVe dictionary. I sum all vectors for given review and divide the
result by the number of vectors summed (compute the average vector). Ignore the words that are not
found in the GLoVe corpus.

2. Training a multilayer perceptron classifier
Perceptron is a ’circuit’ that takes as the input features of the classified object, and
computes a class to which this object belongs.
The features are the average word embeddings for the reviews, and the classes are the
’overall’ user ratings (the sentiment value). I will only consider three classes: negative for overall
rating of 1 or 2, neutral for rating 3, and positive for rating 4 or 5. I remap the ’overall’
rating column in the data from five classes to three classes before training.
I will train a perceptron with 50 inputs/features (if using a 50-dimensioned GLoVe file) and with
three outputs (0/1/2, standing for negative/neutral/positive). The training will be performed by an
existing algorithm from a machine learning library.

3. 10-fold cross validation
Split the review file into 10 parts randomly. For every 9 parts of these, I perform the training of the
perceptron. Then I use the last part for the validation. In the validation stage, I compute the text
representation of average embedding vectors for each review, and ask the perceptron to predict the
sentiment (this again is done using an existing algorithm taken from a library). Then I compare the
predicted sentiment class, with the class originally stored in the Amazon files.

4. Implementation:
1- Use scala, Apache Spark, and its machine learning library MLlib.
2- While Spark provides a cool SQL interface to DataFrames, I avoid using it. I want to train
the functional style interface. So I don’t use the sql method.
3- Use map, flatMap or a for-comprehension.
4- Implement the entire program in a pure way (except for the spark API calls that are impure).
Make the functions referentially transparent.
