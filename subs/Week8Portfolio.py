# Assessment Lab — Naïve Bayes Text Classification
# Independently implement a Naïve Bayes text classification system using PySpark
# to detect spam emails.
# Building on the techniques learned in the Guided Lab, apply the same machine
# learning workflow to the Enron Email Dataset, which contains real-world email
# messages labeled as spam or ham (not spam).

from pyspark.sql import SparkSession

from src.mlonbigdata import show_df, title_xy_labels

# File location defined
file = '/Users/wizrdm/Desktop/UEL/Machine Learning on Big Data/emails.csv'

# Spark session defined
spark = SparkSession.builder.appName("NaiveBayes").getOrCreate()

# Dataframe defined
df = spark.read.csv(file, header=True, inferSchema=True)

# Show dataframe
show_df(df, rows=5, print_str="Showing data sample:")
print(f"Number of rows: {df.count()}, Number of columns: {len(df.columns)}")
# this csv counts instances of each word in each email

# Investigate balance of spam and ham emails
show_df(df.groupBy("Prediction").count(), print_str="Showing data balance:")

# The task requires you to load the dataset into a Spark DataFrame, perform text
# preprocessing, convert the email content into numerical features using TF–IDF,
# and train a classification model capable of predicting whether an email is spam.
# Implement the full Spark ML pipeline, including label indexing, tokenisation,
# stop-word removal, feature extraction, model training, prediction, and
# evaluation.
# In addition to implementing the pipeline for the provided dataset, you must also
# train and compare three versions of the Naïve Bayes algorithm (as discussed in
# the lecture slides).
# The models should be evaluated using appropriate classification metrics such as
# accuracy, precision, recall, and F1-score, and the results should be analysed
# to determine which version performs best for spam detection.
# Your submission should clearly demonstrate the complete workflow by providing
# screenshots of each stage, including Spark setup, dataset loading, preprocessing,
# model training, prediction outputs, and evaluation metrics. The goal of this
# assessment is to test your ability to apply distributed machine learning
# techniques in PySpark and to adapt the Guided Lab pipeline to a new dataset
# while critically comparing the performance of different Naïve Bayes models

# Dataset Description
# The Enron Email Dataset (The Enron Email Dataset) contains real email messages
# labelled as spam or ham (not spam). Each record contains a label and the
# corresponding email text.
# Submission Requirements
# Students must submit:
# 1. Pdf/Word file with Screenshots showing all steps as needed