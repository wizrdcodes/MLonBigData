from sklearn.datasets import fetch_20newsgroups
from pyspark.sql import SparkSession
from pyspark.ml.feature import (Tokenizer, HashingTF, IDF, StringIndexer,
                                StopWordsRemover)
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.feature import Word2Vec
from pyspark.sql.functions import udf, col
from pyspark.sql.types import ArrayType, StringType
from nltk.stem import PorterStemmer, WordNetLemmatizer
import pandas as pd
import numpy as np
from src.mlonbigdata import show_df

dataframe = 1 # 1 = show dataframe, 0 = hide dataframe
show_dataframes = (dataframe == 1)

# Start Spark Session
spark = SparkSession.builder.appName("DocumentClassification").getOrCreate()

# Fetch 20 Newsgroups Data
newsgroups = fetch_20newsgroups(subset='all')

# Convert the dataset to a DataFrame for PySpark processing
data = pd.DataFrame({'text': newsgroups.data, 'category': newsgroups.target})
df = spark.createDataFrame(data)
if show_dataframes: show_df(df, rows=5, truncate=50, print_str="20 Newsgroups Data:")

# Show some details about the data
print(f"Total number of documents: {len(newsgroups.data)}")
print(f"Categories: {newsgroups.target_names}")
print(f"Number of categories: {len(newsgroups.target_names)}")

# Display the distribution of categories
if show_dataframes: show_df(df.groupBy('category').count(), rows=20, truncate=50,
                            print_str="Category Distribution:")

# Filter 25% of documents from each category
fractions = {i: 0.25 for i in range(len(newsgroups.target_names))}
df_sampled = df.sampleBy("category", fractions=fractions, seed=42)
if show_dataframes: show_df(df_sampled, rows=5, truncate=50,
                            print_str="\n25% of each category of Newsgroups:")

# Show the number of documents after sampling
print(f"Total number of documents after sampling: {df_sampled.count()}")
if show_dataframes: show_df(df_sampled.groupBy('category').count(), rows=20, truncate=50,
                            print_str="Category Distribution after sampling:")

# Prepare for Document Classification

# Step 1: Define Tokenizer
tokenizer = Tokenizer(inputCol="text", outputCol="words")

# Step 2: Define HashingTF
hashingTF = HashingTF(inputCol="words", outputCol="raw_features", numFeatures=1000)

# Step 3: Define IDF (Inverse Document Frequency)
idf = IDF(inputCol="raw_features", outputCol="features")

# Step 4: Define Indexer
indexer = StringIndexer(inputCol="category", outputCol="label")

# Step 5: Define classifier (Logistic Regression)
lr = LogisticRegression(featuresCol="features", labelCol="label")

# Set up pipeline with stages
pipeline = Pipeline(stages=[tokenizer, hashingTF, idf, indexer, lr])

# Split data into training and testing sets (80% train, 20% test)
train_data, test_data = df_sampled.randomSplit([0.8, 0.2], seed=42)

# Step 6: Train the model using the pipeline
model = pipeline.fit(train_data)

# Step 7: Make predictions on the test data
predictions = model.transform(test_data)

# Show predictions
show_df(predictions, 'text', 'category', 'prediction', rows=5, truncate=50,
                print_str="\nPredictions (selected columns):")
# show_df(predictions, rows=5, truncate=50, print_str="\nPredictions DF:")

# Step 8: Evaluate the model's accuracy
evaluator = MulticlassClassificationEvaluator(labelCol="label",
                        predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)

# Display the accuracy
print(f"Model Accuracy: {accuracy:.2f}")

# Apply the pipeline to the sampled data (df_sampled) to get the 'features'
# column (TF-IDF vectors)
processed_data = model.transform(df_sampled)

# Extract the "features" column as an RDD
tdm_rdd = processed_data.select("features").rdd.map(lambda row: row[0])

# Convert the RDD of vectors into a numpy array
tdm_array = np.array(tdm_rdd.collect())

# Convert the numpy array into a DataFrame (this is our Term-Document Matrix)
tdm_df = pd.DataFrame(tdm_array)

# Show the Term-Document Matrix (first few rows)
print("Term-Document Matrix (TDM):")
print(tdm_df.head())

# Word2Vec
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def stem_words(words):
    return [stemmer.stem(word) for word in words]

def lemmatize_words(words):
    return [lemmatizer.lemmatize(word) for word in words]

stem_udf = udf(stem_words, ArrayType(StringType()))
lemma_udf = udf(lemmatize_words, ArrayType(StringType()))

# df processing
df_labeled = indexer.fit(df_sampled).transform(df_sampled)
df_tokenized = tokenizer.transform(df_labeled) # tokenizer already defined above
remover = StopWordsRemover(inputCol="words", outputCol="filtered_words") # not yet defined
df_filtered = remover.transform(df_tokenized)
df_stemmed = df_filtered.withColumn("stemmed_words", stem_udf(col("filtered_words")))
df_lemmatized = df_stemmed.withColumn("lemmatized_words", lemma_udf(col("filtered_words")))

df_process = [
    ("df_tokenized", df_tokenized), ("df_filtered", df_filtered), ("df_stemmed", df_stemmed),
    ("df_lemmatized", df_lemmatized),]
for name, df in df_process:
    if show_dataframes: show_df(df, rows=5, truncate=50, print_str=f"{name}:")

word2Vec = Word2Vec(vectorSize=25, minCount=2, seed=42,
                    inputCol="lemmatized_words", outputCol="featuresW2Vector")
word2Vec_model = word2Vec.fit(df_lemmatized)
df_w2v = word2Vec_model.transform(df_lemmatized)
if show_dataframes:
    show_df(df_w2v, 'text', 'featuresW2Vector', truncate=50, print_str="\nWord2Vec Features:")
    # show_df(df_w2v, truncate=100, print_str="\nWord2Vec Features:")

train_data_w2v, test_data_w2v = df_w2v.randomSplit([0.8, 0.2], seed=42)

# Train Model
lr_w2v = LogisticRegression(featuresCol="featuresW2Vector", labelCol="label")
lr_w2v_model = lr_w2v.fit(train_data_w2v)

# Predictions
predictions_w2v = lr_w2v_model.transform(test_data_w2v)
show_df(predictions_w2v, 'text', 'category', 'prediction', rows=10, truncate=50,
        print_str="\nWord2Vec Predictions:")

# Evaluate Model Accuracy
accuracy_w2v = evaluator.evaluate(predictions_w2v)

# Compare accuracy of word2vec and logistic regression
print(f"Logistic Regression Accuracy: {accuracy:.2f}")
print(f"Word2Vec Accuracy: {accuracy_w2v:.2f}")

