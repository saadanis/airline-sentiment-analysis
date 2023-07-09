# Sentiment Analysis on Airline Tweets

This is an Assessed Coursework for the [Text-as-Data](https://www.gla.ac.uk/coursecatalogue/course/?code=COMPSCI5096) postgraduate course at the [University of Glasgow](https://www.gla.ac.uk) during the academic year 2022-2023.

## Coursework Specification

### Summary

The TaD coursework aims to assess your abilities relating to techniques discussed in the course. The objective is to assess your ability in text processing techniques, and applications to text classification. This is an individual exercise, and you should work independently. You will also read and critique a research paper.

## Tasks

### Task 1: Dataset

*Your first step is to choose a text classification dataset for this coursework. You will explore it, build and evaluate a classifier on it. The dataset should contain a number of documents (which could be short like a tweet or long like a research article). It should also have some labels (more than two) that you want to be able to predict automatically. Those labels may be categories, topic information, sentiments, authors, or other.*

The dataset, “Twitter US Airline Sentiment” is a collection of tweets that are aimed at each of the major United States airlines, each labeled with the sentiment portrayed in the tweet (negative, neutral, or positive), the name of the airline the tweet was aiming at (United, Virgin America, etc.), and if the sentiment was negative, the reason for the negativity (Bad Flight, Customer Service Issue, etc.). This dataset was chosen to develop a sentiment analysis classifier. 

### Task 2: Clustering

*Your next step is to perform k-means clustering over your data. Write your own implementation of k-means clustering over sparse TF-IDF vectors of your dataset.*

Every cluster apart from cluster 0 is dominated by negative labels. Only the 0th cluster is dominated by positive labels. Over 70% of the 1st, the 2nd, and the 3rd clusters have negative labels. Over 60% of the labels of the 4th cluster are negative. In every cluster, neutral tweets are neither the majority nor the minority. The 1st, 2nd, 3rd, and the 4th clusters have only 4-13% of positive labels. From this information, it can be revealed there is a strong separation between positive and negative labels, while very little effect on the neutral labels. Primarily, clusters seem to have been divided by topics rather than sentiment, but coincidently clustering most of the positive labels into the 0th cluster.

### Task 3: Comparing Classifiers

*Use the text in your dataset to train baseline classification models with the Scikit Learn package. Conduct experiments using the following combinations of classifier models and feature representations:*  
* *Dummy Classifier with strategy="most_frequent"*
* *Dummy Classifier with strategy="stratified"*
* *LogisticRegression with One-hot vectorization*
* *LogisticRegression with TF-IDF vectorization (default settings)*
* *SVC Classifier with One-hot vectorization (SVM with RBF kernel, default settings)*

The Dummy Classifiers had F1 scores of 0.257 and 0.320 on the validation datasets, which all the other classifiers easily beat. The SVC model trained with the one-hot vectorized training set resulted in a validation F1 score of 0.686. The regularization parameter was set to its default value of 1.0. The kernel was also set to its default ‘rbf’ value. The model has a high recall value (0.93) for negative labels, but relatively lower recall values (0.44 and 0.60) for the other labels suggesting that the model was more likely to label tweets as negative. The Logistic Regression model trained on the TF-IDF vectorized data performed worse with a validation F1 score of 0.659. The recall values had an even greater disparity than the SVC model with the recall for negative labels being 0.95 and the recall for neutral and positive being 0.36 and 0.56 respectively. This also shows a high bias for labeling tweets as negative. The best performing model was the Logistic Regression model trained on the one-hot vectorized dataset. Its training F1 score was 0.920 and its validation F1 score was 0.706. This model does not share the negative labeling bias as strongly as the other two classifiers, having a 0.87 recall for negative labels, 0.53 for neutral labels, and 0.68 for positive labels.

### Task 4: Parameter Tuning

*In this task you will improve the effectiveness of the LogisticRegression with TF-IDF vectorisation from Task 3. Tune the parameters for both the vectorizer and classifier on the validation set.*  
1. *Classifier - Regularisation C value (typical values might be powers of 10 (from 10^-3 to 10^5)*
2. *Vectorizer - Parameters: sublinear_tf and max_features (vocabulary size) (in a range None to 50k)*
3. *Select another parameter of your choice from the classifier or vectorizer*

The following parameters were chosen in the parameter grid:
* For C, the regularization parameter of the classifier: 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000.
* For ‘max_features’, the number of tokens to use: None (all), 915, 1830, 3660, 7318.
* For ‘sublinear_tf’, whether replace term frequency (tf) with 1 + log(tf): True, False.
* For ‘tokenizer’, which tokenizer to use: None, ‘text_pipeline_spacy’ (custom-made tokenizer using spaCy).

The search produced the following parameters as the best option.
* C:1
* max_features: 1830
* sublinear_tf: False
* tokenizer: None

### Task 5: Context Vectors Using BERT

*Now you will explore whether a deep learning-based approach can improve the performance compared to the earlier more traditional approaches. Encode the text of your documents using the 'feature-extraction' pipeline from the HuggingFace library with the ‘roberta_base’ model. Train an end-to-end classifier using the ‘trainer’ function from the HuggingFace library, again using the ‘roberta_base’ model. Try different values for the model, learning_rate, epochs and batch_size.*

The first approach vectorized the text using a base model and used the vector of the start token in a Logistic Regression model to perform the classification. The second approach fine-tuned the existing base model with the new training data, thereby improving the existing model’s performance on the current task. Hence, all the end-to-end Trainer models performed better than the pipeline model.

### Task 6: Conclusions

*You will now take your best model from Tasks 3, 4, and 5, and evaluate it on the test set. You will then explore the performance of your classifier and discuss the strengths and weaknesses of your machine learning pipeline.*

The best performing model was the roBERTa-base model which was finetuned for sentiment analysis on Twitter data. This model resulted in an F1 score of 0.831 on the validation dataset. On the test set, it performed slightly worse with an F1 of 0.820.

### Task 7: Research Paper Report

*Write a short report of a maximum of 500 words on a research paper.*

[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://aclanthology.org/N19-1423.pdf)
