# Project-3

# UTOR: FinTech Bootcamp - Project 3: Yelp Review Analysis (Natural Language Processing and Machine Learning)

## Problem Statement

The project aims to utilize NLP techniques and machine learning algorithms to gain insights from the Yelp review data. NLP will be used to process and analyze the natural language text of the reviews, which can be complex and unstructured. The goal is to extract relevant information and insights from the review text data and use machine learning to build models that can predict or classify the sentiment or other attributes of the reviews.

## Background

Sentiment Analysis, also known as Opinion Mining, is a field of study that deals with identifying and extracting subjective information from text, such as reviews or social media posts. The goal of sentiment analysis is to classify a given text into positive, negative or neutral sentiment.

Sentiment analysis is an important application of NLP (Natural Language Processing) and machine learning because it helps businesses and organizations understand their customers' opinions and feedback. By analyzing the sentiment of reviews, businesses can gauge customer satisfaction and identify areas that need improvement.

Sentiment analysis involves several steps, including text preprocessing, feature extraction, and classification. Text preprocessing includes tasks such as cleaning and normalizing the text, removing stop words and punctuation, and stemming or lemmatizing words. Feature extraction involves transforming the text into numerical features that can be used as input to a machine learning model. Finally, the machine learning model is trained on a labeled dataset, and used to predict the sentiment of new reviews.

There are several techniques used in sentiment analysis, including lexicon-based approaches, machine learning algorithms such as Naive Bayes, Support Vector Machines, and Deep Learning models like Long Short-Term Memory (LSTM) networks. The choice of technique depends on the size of the dataset, the complexity of the problem, and the available computational resources.

## Contributor 

* [Aizhen Dong]()

## Technologies Used

* Python
* Pandas
* Numpy
* Matplotlib and seaborn
* Wordcloud
* nltk
* Sklearn (CountVectorizer, MultinomialNB, train_test_split, classification_report, confusion_matrix)
* pickle

## Dataset

The Yelp dataset can be obtained by downloading it from the official Yelp website at [Yelp Dataset](https://www.yelp.com/dataset). This is the official source for the Yelp review data, which includes information about businesses, users, and their reviews. Once the users have access, they can download the dataset as a compressed file in JSON format. The size of the dataset is quite large, so it is important to have adequate storage space and a fast internet connection before starting the download.

[Yelp Dataset](https://www.yelp.com/dataset)

The Yelp dataset I have chosen for this project is the review dataset, which includes information about individual reviews written by Yelp users. This dataset is particularly useful for NLP and machine learning that aim to gain insights into customer opinions, preferences, and experiences.

![Yelp Dataset](/images/yelp_dataset.png)

## Steps
### **(1) Json-csv converter**

The first step in processing the Yelp review data is to convert it from the JSON format to the CSV format. Converting the Yelp review data from JSON to CSV can make it easier to load and manipulate the data with Python.

[json-csv converter](https://github.com/JD-Yue/Project-3/blob/main/1_collecting%20data/json-csv_original_data.ipynb)

The following steps/code perform the conversion of the Yelp review data from JSON to CSV, and loads the resulting data into a Pandas dataframe for further analysis and processing.

* Open the JSON file and extract the header information. This is done by reading the first line of the file, decoding it using the json.loads function, and extracting the keys of the resulting dictionary, which correspond to the header names.
* Write the header names to the CSV file using the csv.DictWriter class. This class is initialized with the file handle and header information, and the header information is written to the file using the writeheader method.
* Read the contents of the JSON file line by line, and decode each line using the json.loads function. For each line, the decoded line contents are written to the CSV file using the writerow method of the csv.DictWriter object.
* Finally, the CSV file is loaded into a Pandas dataframe using the pd.read_csv function, which provides a convenient and efficient way to work with tabular data in Python.

### **(2) Downsize the data**

The second step is to downsize the data, as the file is too big to process efficiently with standard computing resources. The size of the file (Json file: 5.34GB and csv file: 4.69GB) can make it challenging to load the data into memory, slow down processing times, and limit the ability to work with the data.

[Downsize data](https://github.com/JD-Yue/Project-3/blob/main/1_collecting%20data/downsized_data.ipynb)

The code downsize the large CSV file 'yelp_academic_dataset_review.csv' by randomly sampling only 0.25% of its rows (SAMPLE_SIZE = 0.0025). It uses the skiprows argument in pd.read_csv to randomly select rows based on the outcome of the random number generator, with seed set to 42. After reading the data into a dataframe df_downsized, the code adds headers to the columns using the columns attribute and sets the column names in the headers list. Finally, it calculates the descriptive statistics of the dataframe using the describe() method.

* Before downsizing (6990280 rows)

![Descriptive statistics](/images/descriptive_statistics.png)

* After downsizing (17409 rows)

![Downsized descriptive statistics](/images/downsized_descriptive_statistics.png)

 The mean, median, mode, standard deviation, etc. of the 'stars' column in the original dataset and the downsized dataset are very close to each other, which suggests that the distribution of the 'stars' column in both datasets is very similar. Hence, using the 'stars' column to differentiate between different groups of reviews (e.g. positive reviews vs. negative reviews) in future analysis should not have a significant impact on the results.

### **(3) Visualizing and understanding and creating datasets**

[Visualize, understand and create datasets](https://github.com/JD-Yue/Project-3/blob/main/2_understanding%20data.ipynb)

**Length of text**

 I plot a histogram of the frequency of the length of the text reviews in the 'yelp_df' dataframe. The purpose of this plot is to give an overview of the distribution of the length of the reviews in the dataframe.  Understanding the distribution of review length can help in choosing appropriate parameters for further analysis or modeling, such as setting a threshold for the maximum review length to be considered, or selecting an appropriate text representation method based on the average review length.

 ![text frequency](/images/text_frequency.png)

**Stars**

 To understand stars, I create a count plot of the 'stars' column in the dataframe using the seaborn library and to display the distribution of the 'length' column of the dataframe as histograms, faceted by the 'stars' column. The count plot of the 'stars' column will give an overview of the number of reviews with each rating (1 to 5 stars). The histograms of the 'length' column, faceted by 'stars', will give an overview of the distribution of the length of reviews for each rating.

![star count](/images/star_count.png)

![star length frequency](/images/star_length_frequency.png)

Decision to use 1 star and 5 stars in the analysis

Based on the results, the two categories (1 star and 5 stars) seem to have enough representation in the data to allow for meaningful analysis. Using only 1 star and 5 star reviews may have made it easier to distinguish between negative and positive reviews, respectively, compared to including more categories of reviews like 2, 3, and 4 stars. It is also common in sentiment analysis to focus on the extremes of a rating scale to differentiate between negative and positive sentiments. The choice of using only 1 and 5 stars is made to simplify the analysis and to make a clear distinction between negative and positive feedback.

**Create dataframes**

I create two separate dataframes, one for reviews with 1 star rating (yelp_df_1) and another for reviews with 5 star rating (yelp_df_5), by filtering the original dataframe yelp_df based on the value of the 'stars' column. Then, a third dataframe yelp_df_1_5 is created by concatenating the two dataframes yelp_df_1 and yelp_df_5 using the pd.concat() function. The resulting dataframe yelp_df_1_5 only includes reviews with either 1 star or 5 star rating. 

![1 star vs 5 stars](/images/1_star_5_stars.png)

I then do a count of the 1 star and 5 stars reviews to get a further understanding. The count of the number of 1-star reviews is smaller compared to the count of the number of 5-star reviews in the dataset. This information could suggest that there are fewer negative reviews than positive reviews in the dataset. It also highlights the imbalance in the distribution of the 'stars' column in the dataset and could impact the results of further analysis if not handled properly.

I then use wordcount to visualize the 1 star and 5 star reviews. This helps to understand the most common words used in the positive (5 stars) and negative (1 star) reviews and thus gives a deeper understanding of the characteristics of each group. The frequency of each word is visualized in the wordcloud, with the most frequently used words appearing larger. This provides a quick and easy way to see what customers generally like and dislike about the product or service being reviewed.

1 star

![1 star wordcount](/images/1_star_wordcount.png)

5 stars

![5 stars wordcount](/images/5_stars_wordcount.png)

### **(3) Data cleaning and applying NLP**

[Clean data and apply NLP](https://github.com/JD-Yue/Project-3/blob/main/3_data_cleaning.ipynb)

The next step is performed as a preprocessing step for natural language processing. The goal is to clean the text data and convert it into a format that can be used as input for NLP algorithms. This includes removing punctuation marks and removing stopwords (common words that do not carry much meaning, such as "the" and "and"). 

The step of cleaning the text data is important in preparing the data for NLP modeling. This cleaning process helps to improve the accuracy of the model by removing irrelevant information and reducing the noise in the data. It also helps to standardize the text data, making it easier for the NLP model to process and analyze. By removing punctuation and stopwords, the data is made more concise and focused, reducing the dimensionality of the feature space and allowing the model to train on only the most relevant information. This, in turn, can lead to better model performance and more accurate predictions.

The cleaned data is then vectorized, which means it is transformed into numerical data that can be processed by machine learning algorithms. This is done using the CountVectorizer method from the scikit-learn library, which tokenizes the text data and converts it into a sparse matrix representation.

### **(4) Model training**

[Model training](https://github.com/JD-Yue/Project-3/blob/main/4_model_training%20(1).ipynb)

The next step is training the machine learning model (Multinomial Naive Bayes classifier (NB_classifier)) for sentiment analysis on the Yelp dataset. The goal of this model is to predict the star rating (1 to 5) based on the text reviews.

Data preparation: The first step is to prepare the data that will be used to train and test the model. The code defines two variables X and y. X is the input feature matrix and y is the target variable. In this case, X is represented by the output of the vectorizer applied on the text reviews (yelp_vectorizer) and y is represented by the star ratings (yelp_df_1_5['stars'].values).

Train/Test Split: The next step is to split the data into training and testing sets. The train_test_split function is used to randomly split the data into two parts, with X_train and y_train being used for training the model and X_test and y_test being used for evaluating the model's performance.

Model Training: The fit method is then used to train the NB_classifier model using the training data (X_train, y_train).

Multinomial Naive Bayes is a popular choice for text classification tasks, such as sentiment analysis, because it makes the assumption that the features (in this case, the words in the text reviews) are conditionally independent given the target class (the star rating). This means that the presence or absence of a word in a review does not depend on the presence or absence of any other word in the same review.

The "Naive" part of the name comes from this independence assumption, which is not always valid in practice, but works well in many cases. The "Multinomial" part of the name comes from the fact that the algorithm is designed to work with discrete features (such as word counts in text documents), rather than continuous features.

Another advantage of the Multinomial Naive Bayes algorithm is that it is computationally efficient and easy to implement, making it a good choice for quickly prototyping a model or working with large datasets. Additionally, it has been shown to perform well on many text classification tasks, including sentiment analysis.

In this case, the choice of using a Multinomial Naive Bayes model for sentiment analysis on the Yelp dataset may have been made due to its simplicity, efficiency, and prior success on similar tasks.

## Results

**Train**

![train heatmap](/images/train_heatmap.png)

**Test**

![test heatmap](/images/test_heatmap.png)

**Classification Report**

![classification report](/images/classification_report.png)


**Test**
[Streamlit](https://github.com/JD-Yue/Project-3/blob/main/app.py)

## Conclusion

The use of NLP (Natural Language Processing) techniques in conjunction with the Multinomial Naive Bayes model can lead to very effective results in sentiment analysis. The confusion matrix and accurate classification report are valuable tools for evaluating the performance of a model and determining how well it is able to correctly predict the target class.

The confusion matrix show that the model is able to accurately predict the target class for the majority of instances in the train and test set, with low numbers of false positive and false negative predictions. The classification report further provide more detailed information on the model's performance, including precision, recall, and F1 score and shows the predictions with high accuracy. In this case, the model performs really well with an accuracy of 0.93.

Overall, the combination of NLP and the Multinomial Naive Bayes model can be a powerful tool for sentiment analysis and the use of a good confusion matrix and accurate classification report can help ensure that the model is performing well.

## Postmortem

**(1) Json to csv converter**

The "json_to_csv_converter.py" file ([Yelp/dataset-examples](https://github.com/Yelp/dataset-examples)) provided by Yelp is being "too old" (created more than 8 years ago and no updates to it yet) which means that it no longer works with the recent version of Python. Thus the converter file is considered as unusable as of now.

**(2) 1 star review data size**
The distribution of the lengths of 1 star and 5 star reviews is significantly different, it might affect the validity of the analysis. E.g., A text of 1 word or two words reviews will probably show a rating of 5 stars as the dataset for 5 stars used in this project is much bigger than the 1 star used.

## Reference

* Yelp Dataset: https://www.yelp.com/dataset
* Generating Word Cloud in Python: https://www.geeksforgeeks.org/generating-word-cloud-python/
* sklearn.naive_bayes.MultinomialNB: https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html
* Preprocessing steps in NLP: https://www.educative.io/answers/preprocessing-steps-in-natural-language-processing-nlp
* sklearn.feature_extraction.text.CountVectorizer: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
* NLP Punctuation, Lower-Case and StopWords Pre-Processing: https://medium.com/@LauraHKahn/nlp-punctuation-lower-case-and-stopwords-pre-processing-d4888c4da940
