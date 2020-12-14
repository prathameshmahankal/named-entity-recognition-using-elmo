# Named Entity Recognition - Extracting Addresses From Eviction Notices

**Goal:** To extract addresses from the Eviction Notices using Deep Learning Approaches.

**Approach:** This is a Named Entity Recognition problem. To tackle this, I decided to use a popular NER algorithm - ELMo.

## ELMo Model

The current model architecture . Before building the model, we add every document with some padding so that every document is of a fixed maximum length. This padded document is embedded using the ELMo embedding and then passed through two Bidirectional layers separately. The results from these two layers are then summed and passed through a TimeDistributed layer before we get the final output. The optimizer we finalized on is adam and loss is sparse categorical cross entropy. Note that for my current model building process I am not making use of the GPU (so the training times would be much higher than you would expect).

For hyperparameter tuning, I am using Hyperas, which is a wrapper for the hyperopt package available in python. You can read more about it here. Using hyperparameter tuning we optimize our model for custom metrics like recall, precision and f1 score (since our data is highly imbalanced with only a few address words among a sea of other words). For validation, we consider both precision and recall for each document prediction and then average them out to get the score for each iteration.

Note: This model requires tensorflow 1.15, which is only available in python version <3.8. So I would suggest creating a virtual environment with the appropriate version for python (ELMo wasn’t available in Tensorflow 2+ while I was using it so you may ignore this step if ELMo is available in TF2 when you are reading this)


## Creating Training Data:

To train my model, we used a custom dataset made using lines from a book combined with some fake addresses which were inserted in between. With this setup, we built two datasets - Original and Tinkered.

### Original Dataset:

To build this dataset, we first collected all the different sentences from ‘The Project Gutenberg eBook, Tommy Remington's Battle, by Burton Egbert Stevenson’. From this list, we removed all non-alphanumeric characters. Next, we injected some fake addresses using the Faker package in python. Finally, we have a collection of documents, each containing a series of lines with one or more addresses present at different points.

Thus, the original files contain:
1. strings of text from the choice of book that is free of next line characters, punctuation, and websites
2. strings of addresses free from next line characters and punctuation
3. all together, free from whitespace

We then took different random samples of varied sizes from this dataset and trained an ELMo model on this sample dataset. We then tested this dataset on a test sample which was held out earlier. The table below shows the performance metrics for each of these iterations:

Training Size | Precision | Recall | F-Score | Training Time
--- | --- | --- | --- | ---
5 | 0.78 | 0.6 | | 1284 (5 epochs)
15 | 0.92 | 0.87 |  | 1765 (5 epochs)
50 | 0.86 | 0.88 |  | 1774 (3 epochs)
150 | 0.98 | 0.97 |  | 3654 (5 epochs)

Here, we only had letters from the book combined with the addresses. However, this created two problems:
1. Due to the absence of numbers, our model started predicting every number in the actual data as an address.
2. Again, since our model did not have enough Capitalized words, our model began predicting all such words as addresses.

### Tinkered Dataset:

To solve this problem, we created a new dataset where we introduced:
1. strings of text from the choice of book that is free of next line characters, punctuation, and websites
2. strings of addresses free from next line characters and punctuation
3. all together, free from whitespace
4. also includes capitalized words
5. Also includes phone numbers, however, the phone numbers that are injected include characters such as "-" and parenthesis

Below you can see how this new dataset actually performed compared to the previous one:

Training Size | Precision | Recall | F-Score | Training Time
--- | --- | --- | --- | ---
5 | 0.53 | 1 |  | 4983 (5 epochs)
25 | 0.79 | 0.94 |  | 2974
50 | 0.92 | 0.87 |  | 5844
150 | 0.95 | 0.99 |  | 8741

As you can see, the training durations for the models built on tinkered dataset are insane, but we will reap the benefits of this when we test this model on real data.

## Performance On Real Data:

Now that we have our model ready, we can start testing it on real eviction notices.

## Future Steps:

In addition to ELMo, I would like to try out the BERT model as well
The current model training was done entirely on synthetic data, so maybe adding a few real eviction notices might improve our model performance.
