
# Toxicity Classifier Model
This is a toxic comment classifier model that classifies raw text into 6 labels:

-   `toxic`
-   `severe_toxic`
-   `obscene`
-   `threat`
-   `insult`
-   `identity_hate`

**Code**
All the code and annotation for building and training the model can be found at [toxicity_classifier_training_capstone.ipynb](https://github.com/C23-PS376/toxicity-classifier-model/blob/main/toxicity_classifier_training_capstone.ipynb)

This model is built using Tensorflow [Keras](https://keras.io/).

**Input**

The model from [toxicity_classifier_training_capstone.ipynb](https://github.com/C23-PS376/toxicity-classifier-model/blob/main/toxicity_classifier_training_capstone.ipynb) takes input of raw text with **50 words** maximum. If the model is fed with text more than **50 words**, only **50 words** will be taken.

**Output**

The model from [toxicity_classifier_training_capstone.ipynb](https://github.com/C23-PS376/toxicity-classifier-model/blob/main/toxicity_classifier_training_capstone.ipynb) will output a **2D array with each row of 6 numbers**. Each number denotes probability from 0 to 1. If the number is > 0.5, then it's positive to the corresponding label.

For example if the output is 
`[[0.9763663 0.19614795 0.79277015 0.01046047 0.81799424 0.7410841 ]]`
We apply threshold of > 0.5 for positive, else negative, then:

    [[positive, negative, positive, negative, positive, positive]]

Then we see the label in order, it means the output is:
-   `toxic = positive`
-   `severe_toxic = negative`
-   `obscene = positive`
-   `threat = negative`
-   `insult = positive`
-   `identity_hate = positive`

**Reference**

To build this model, we took reference and code from:

 - [TextVectorization layer (keras.io)](https://keras.io/api/layers/preprocessing_layers/core_preprocessing_layers/text_vectorization/)
 - [Using pre-trained word embeddings (keras.io)](https://keras.io/examples/nlp/pretrained_word_embeddings/)

# Dataset Source

The dataset for building this model is from Kaggle: [https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data).

To download the data in Google Colab notebook:

 1. Create Kaggle account
 2. Follow this [link](https://www.kaggle.com/general/74235) to create Kaggle token
 3. Use this command to download the dataset `! kaggle competitions download -c 'jigsaw-toxic-comment-classification-challenge'`

Complete code implementation is on [toxicity_classifier_training_capstone.ipynb](https://github.com/C23-PS376/toxicity-classifier-model/blob/main/toxicity_classifier_training_capstone.ipynb) file

## Dataset
The dataset consists of:

 - 159570 rows of train data 
 - 153164 rows of test data (only 63978 used)

**The dataset of train data has these columns:**

 - `id (string)`
 - `comment_text (string)`
 - `toxic (0 or 1)`
 - `severe_toxic (0 or 1)`
 - `obscene (0 or 1)`
 - `threat (0 or 1)`
 - `insult (0 or 1)`
 - `identity_hate (0 or 1)`

We preprocess the train data and remove the `id` column because it isn't used in training process.

**The dataset of test data has these columns:**

In the test data, there are two files, the text with columns:
 - `id (string)`
 - `comment_text (string)`

and the labels with columns:
 - `id (string)`
 - `toxic (0 or 1)`
 - `severe_toxic (0 or 1)`
 - `obscene (0 or 1)`
 - `threat (0 or 1)`
 - `insult (0 or 1)`
 - `identity_hate (0 or 1)`

We have to preprocess the data by joining them according to `id` then remove the `id` column. 
Then out of 153164 rows, only 63978 are labeled correctly. The rest which are not used, have `-1` as labels. These rows which have `-1` as labels come from [Kaggle](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data), not because we decided so.

## Word Embeddings
We use word embeddings trained using GloVe from Stanford University.

There are a couple of versions from this word embeddings and we choose the one that has 200 dimensions for each word.

Reference:

 - Word Embeddings: [GloVe: Global Vectors for Word Representation (stanford.edu)](https://nlp.stanford.edu/projects/glove/)
 - GloVe: [GloVe - Wikipedia](https://en.wikipedia.org/wiki/GloVe)

## Model Architecture
There are 2 models in the [saved_model](https://github.com/C23-PS376/toxicity-classifier-model/tree/main/saved_model) folder:

 - Model (No TextVectorization layer)

Layer (type) | Output Shape | Param # | 
--- | --- | --- |
(Embedding) | (None, 50, 200) | 4000400 |
(Bidirectional LSTM) | (None, 50, 64) | 59648 |
(Bidirectional LSTM) | (None, 64) | 24832 | 
(Dense) | (None, 32) | 2080 | 
(Dense) | (None, 6) | 198 |

 - End-to-end model (TextVectorization included)
 
Layer (type) | Output Shape | Param # | 
--- | --- | --- |
(Text Vectorization) | (None, 50) | 0 |
(Embedding) | (None, 50, 200) | 4000400 |
(Bidirectional LSTM) | (None, 50, 64) | 59648 |
(Bidirectional LSTM) | (None, 64) | 24832 | 
(Dense) | (None, 32) | 2080 | 
(Dense) | (None, 6) | 198 |

TextVectorization layer is used to convert raw text into sequences.

End-to-end model is just the base model with TextVectorization appended on top of the Keras sequential model.

If you use the model without TextVectorization layer, you have to convert raw text to sequences manually before feeding it to the model.

These 2 models are saved in form of Tensorflow **saved model** format.

## Model Evaluation

The model is evaluated using recall, precision and F1 score. It's because the dataset label is imbalance. There are too many negative labels compared to positive labels.

**Evaluation on train data:**

 - Precision = 0.8649898
 - Recall = 0.67704713
 - F1 score = 0.7595652663133241

**Evaluation on test data:**
 - Precision = 0.63464135
 - Recall = 0.68161124
 - F1 score = 0.6572882447429294
 
The F1 for test data is only 0.66 because the test data contains text not only in English but also in other languages and emoji. While in this model we use English word embedding because the intended use for application is for English only. The best practice is to filter so only English words in dataset. But, there are 63978 rows of text and we only have 1 month to complete the entire project. Also in the dataset, there are many rows which have more text more than 50 words so the model clips them. This will reduce the model's ability to classify them.
 
For more information about the evaluation metrics:

 - [Precision and Recall](https://en.wikipedia.org/wiki/Precision_and_recall)
 - [F1 score](https://en.wikipedia.org/wiki/F-score)
