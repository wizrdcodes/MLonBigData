import nltk
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, StringType
from nltk.stem import PorterStemmer, WordNetLemmatizer
from src.mlonbigdata import show_df

nltk.download('wordnet')
nltk.download('omw-1.4')

# Define UDFs for Stemming & Lemmatization
# Initialize Stemmer and Lemmatizer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Define UDF for Stemming
def stem_words(words):
    return [stemmer.stem(word) for word in words]

# Define UDF for Lemmatization
def lemmatize_words(words):
    return [lemmatizer.lemmatize(word) for word in words]

# Convert Python functions to PySpark UDFs
stem_udf = udf(stem_words, ArrayType(StringType()))
lemma_udf = udf(lemmatize_words, ArrayType(StringType()))

# Create a Spark session
spark = SparkSession.builder.appName("DocumentClassificationTFIDF").getOrCreate()

# Load & Preprocess Data (Tokenization, Stopword Removal, Stemming, Lemmatization)
# Sample dataset
data = [(0,"Cloud computing is becoming increasingly important for businesses","Technology"),
 (1, "Basketball players are preparing for the next tournament", "Sports"),
 (2,"Machine learning has revolutionized the way data is processed","Technology"),
 (3, "Political campaigns are gearing up for the upcoming elections","Politics"),
 (4, "The football team has been training hard for the upcoming season", "Sports"),
 (5,"International relations are being discussed in diplomatic meetings","Politics"),
 (6, "The president addressed the nation in a live broadcast", "Politics"),
 (7, "Tennis players are practicing for the grand slam matches", "Sports"),
 (8,"Machine learning has revolutionized the way data is processed","Technology"),
 (9, "Stock trading has become a popular way to build wealth", "Finance"),
 (10, "5G technology is expected to significantly improve communication speeds","Technology"),
 (11,"Political campaigns are gearing up for the upcoming elections","Politics"),
 (12,
  "Machine learning has revolutionized the way data is processed",
  "Technology"),
 (13,
  "Investors are looking for high-yield bonds in the current market",
  "Finance"),
 (14,
  "Quantum computing holds promise for solving complex problems",
  "Technology"),
 (15,
  "Cloud computing is becoming increasingly important for businesses",
  "Technology"),
 (16, "The Formula 1 race track is set to host the next grand prix", "Sports"),
 (17,
  "Financial experts are advising on diversifying investment portfolios",
  "Finance"),
 (18, "Stock trading has become a popular way to build wealth", "Finance"),
 (19, "Stock trading has become a popular way to build wealth", "Finance"),
 (20,
  "International relations are being discussed in diplomatic meetings",
  "Politics"),
 (21, "The Formula 1 race track is set to host the next grand prix", "Sports"),
 (22,
  "Political campaigns are gearing up for the upcoming elections",
  "Politics"),
 (23,
  "Machine learning has revolutionized the way data is processed",
  "Technology"),
 (24,
  "The football team has been training hard for the upcoming season",
  "Sports"),
 (25, "Tennis players are practicing for the grand slam matches", "Sports"),
 (26, "Tennis players are practicing for the grand slam matches", "Sports"),
 (27,
  "Political campaigns are gearing up for the upcoming elections",
  "Politics"),
 (28,
  "The real estate market has seen significant growth in the past decade",
  "Finance"),
 (29,
  "Financial experts are advising on diversifying investment portfolios",
  "Finance")]

df = spark.createDataFrame(data, ["id", "text", "category"])
show_df(df, rows=5, print_str="\nOG Data Sample:")

# Tokenization
tokenizer = Tokenizer(inputCol="text", outputCol="words")
df = tokenizer.transform(df)
show_df(df, 'text', 'words', rows=5, truncate=False, print_str="\nTokenized Data:")

# Stopword Removal
remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
df = remover.transform(df)
show_df(df, 'text', 'filtered_words', rows=5, truncate=False,
        print_str="\nStopword Removed Data:")

# Apply Stemming
df = df.withColumn("stemmed_words", stem_udf(col("filtered_words")))
show_df(df, 'text', 'filtered_words', 'stemmed_words', rows=5, truncate=False,
        print_str="\nStemmed Data:")

# Apply Lemmatization
df = df.withColumn("lemmatized_words", lemma_udf(col("filtered_words")))
show_df(df, 'text', 'filtered_words', 'stemmed_words', 'lemmatized_words', rows=5,
        truncate=False, print_str="\nLemmatized Data:")

# Show Results
show_df(df, 'text', 'filtered_words', 'stemmed_words', 'lemmatized_words',
        rows=5, truncate=False, print_str="\nShowing results of preprocessing:")

# Compute TF-IDF After Preprocessing
# Apply HashingTF to the lemmatized words
hashingTF = HashingTF(inputCol="lemmatized_words", outputCol="raw_features",
                      numFeatures=500)
df = hashingTF.transform(df)

# Compute IDF
idf = IDF(inputCol="raw_features", outputCol="features")
idf_model = idf.fit(df)
df = idf_model.transform(df)

# Show TF-IDF Features
show_df(df, 'text', 'features', rows=10, truncate=False,
        print_str="\nTF-IDF Features:")

#Step 5: Train & Evaluate Logistic Regression Classifier

# Convert category labels to numerical labels
indexer = StringIndexer(inputCol="category", outputCol="label")
df = indexer.fit(df).transform(df)
show_df(df, 'category', 'label', rows=10, distinct=True,
        print_str="\nLabel Distribution:")

# Split Data
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

# Train Logistic Regression Model
lr = LogisticRegression(featuresCol="features", labelCol="label")
lr_model = lr.fit(train_data)

# Predictions
predictions = lr_model.transform(test_data)
show_df(predictions, 'text', 'category', 'prediction', rows=10, truncate=False,
        print_str="\nTF-IDF Predictions:")

# Evaluate Model Accuracy
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
                                              metricName="accuracy")
accuracy_tf_idf = evaluator.evaluate(predictions)
print(f"TF-IDF Model Accuracy: {accuracy_tf_idf:.2f}")

show_df(df, 'label', 'category', truncate=False, distinct=True,
        print_str="\nLabel Distribution:")

#Train Word2Vec Model
#import Word2Vec
from pyspark.ml.feature import Word2Vec # Import Word2Vec here

show_df(df, truncate=False, print_str="\nData Sample:")
word2Vec = Word2Vec(vectorSize=100, minCount=1, seed=42, inputCol="lemmatized_words",
                    outputCol="featuresW2Vector")
word2Vec_model = word2Vec.fit(df)
df = word2Vec_model.transform(df)

show_df(df, 'text', 'featuresW2Vector', truncate=False, print_str="\nWord2Vec Features:")
# show_df(df, truncate=False, print_str="\nWord2Vec Features:")

#Step 7: Train & Evaluate Logistic Regression Classifier on Word2Vec Features
# Split Data
train_data_w2v, test_data_w2v = df.randomSplit([0.8, 0.2], seed=42)

# Train Model
lr_w2v = LogisticRegression(featuresCol="featuresW2Vector", labelCol="label")
lr_w2v_model = lr_w2v.fit(train_data_w2v)

# Predictions
predictions_w2v = lr_w2v_model.transform(test_data_w2v)
show_df(predictions_w2v, 'text', 'category', 'prediction', rows=10, truncate=False,
        print_str="\nWord2Vec Predictions:")

# Evaluate Model Accuracy
accuracy_w2v = evaluator.evaluate(predictions_w2v)
print(f"Word2Vec Model Accuracy: {accuracy_w2v:.2f}")

