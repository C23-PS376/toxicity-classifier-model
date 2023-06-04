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

The model from [toxicity_classifier_training_capstone.ipynb](https://github.com/C23-PS376/toxicity-classifier-model/blob/main/toxicity_classifier_training_capstone.ipynb) takes input of raw text with **200 words** maximum. If the model is fed with text more than **200 words**, only **200 words** will be taken.

**Thresholds**

The thresholds for output of the model are:
`[0.67, 0.31, 0.47, 0.4, 0.25, 0.27]`

We got this from using greedy search to find the best threshold for each label which maximizes F1 score.

**Output**

The model from [toxicity_classifier_training_capstone.ipynb](https://github.com/C23-PS376/toxicity-classifier-model/blob/main/toxicity_classifier_training_capstone.ipynb) will output a **2D array with each row of 6 numbers**. Each number denotes probability from 0 to 1. If the number is > 0.5, then it's positive to the corresponding label.

For example if the output is 
`[[0.9763663 0.19614795 0.79277015 0.01046047 0.81799424 0.7410841 ]]`
Then the thresholds:
`[0.67, 0.31, 0.47, 0.4, 0.25, 0.27]`
We apply threshold of (>) for positive, else negative, then:

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

# Saved Model

You can download the saved model here: [https://drive.google.com/drive/folders/10M7n8k2hZXLT-upBNei2CMUDxSRY_Ksm?usp=sharing](https://drive.google.com/drive/folders/10M7n8k2hZXLT-upBNei2CMUDxSRY_Ksm?usp=sharing)

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
We use word embeddings trained using GloVe from Stanford University. This is trained on Twitter comment dataset.

There are a couple of versions from this word embeddings and we choose the one that has 200 dimensions for each word.

Reference:

 - Word Embeddings: [GloVe: Global Vectors for Word Representation (stanford.edu)](https://nlp.stanford.edu/projects/glove/)
 - GloVe: [GloVe - Wikipedia](https://en.wikipedia.org/wiki/GloVe)

## Model Architecture
 
Layer (type) | Output Shape | Param # | 
--- | --- | --- |
(Text Vectorization) | (None, 200) | 0 |
(Embedding) | (None, 200, 200) | 4000400 |
(Bidirectional GRU) | (None, 256) | 253440 |
(Dense) | (None, 128) | 32896 | 
(Dense) | (None, 256) | 33024 | 
(Dense) | (None, 6) | 1542 |

TextVectorization layer is used to convert raw text into sequences.


## Model Evaluation

The model is evaluated using **F1 score**. It's because the dataset label is imbalance. There are too many negative labels compared to positive labels.

We also evaluate this for each label instead of all labels.

**Evaluation on train data:**

 F1 score:
 - `toxic = 79%`
 - `severe_toxic = 53%`
 - `obscene = 84%`
 - `threat = 56%`
 - `insult = 79%`
 - `identity_hate = 63%`

Average = 60%

**Evaluation on test data:**

 F1 score:
 - `toxic = 71%`
 - `severe_toxic = 42%`
 - `obscene = 70%`
 - `threat = 51%`
 - `insult = 67%`
 - `identity_hate = 58%`

 Average = 69%
 
The F1 score for test data is only 60% because the test data contains text not only in English but also in other languages and emoji. While in this model we use English word embedding because the intended use for application is for English only. The best practice is to filter so only English words in dataset. But, there are 63978 rows of text and we only have 1 month to complete the entire project.
 
For more information about the evaluation metrics:

 - [Precision and Recall](https://en.wikipedia.org/wiki/Precision_and_recall)
 - [F1 score](https://en.wikipedia.org/wiki/F-score)
