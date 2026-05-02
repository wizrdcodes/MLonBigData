# ---------------------------------------------------------------------------------
# Assessment Lab — Naïve Bayes Text Classification
# ---------------------------------------------------------------------------------

# The selected version of the Enron dataset was already converted into
# word-frequency columns. Therefore, the pipeline begins from precomputed
# term-frequency features rather than raw email text. TF–IDF was still applied by
# assembling the word-count columns into a feature vector and applying IDF.

import re
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, IDF, Binarizer, StringIndexer
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import col
from scipy.signal.windows import gaussian

from src.mlonbigdata import show_df, title_xy_labels

# File location defined
file = '/Users/wizrdm/Desktop/UEL/Machine Learning on Big Data/emails.csv'

# Spark session defined
spark = SparkSession.builder.appName("NaiveBayes").getOrCreate()

# Control Spark output
spark.sparkContext.setLogLevel("ERROR")

# Dataframe defined
df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(file))

# df viewer helper - first and last 5 columns
def show_first_last_5_cols(df, rows=5,
            print_str="Showing first and last 5 columns of dataframe:"):
    first_5_cols = df.columns[:5]
    last_5_cols = df.columns[-5:]
    selected_cols = first_5_cols + last_5_cols
    show_df(
        df,*[col(f"`{c}`") for c in selected_cols], rows=rows,
        print_str=print_str)

show_first_last_5_cols(df)
print(f"Number of rows: {df.count()}, Number of columns: {len(df.columns)}")

# Investigate balance of spam and ham emails
show_df(df.groupBy("Prediction").count(), print_str="\nShowing data balance:")

# Clean column names
def clean_column_name(name):
    name = name.strip().lower()
    name = re.sub(r"[^a-zA-Z0-9_]+", "_", name)
    name = re.sub(r"_+", "_", name)
    name = name.strip("_") # final output example: Email 1 -> email_1
    if not name:
        name = "blank"
    if name[0].isdigit():
        name = "col_" + name
    return name

cleaned_names = []
seen = {}

for original_name in df.columns:
    base_name = clean_column_name(original_name)
    if base_name in seen:
        seen[base_name] += 1
        new_name = f"{base_name}_{seen[base_name]}"
    else:
        seen[base_name] = 0
        new_name = base_name
    cleaned_names.append(new_name)

df = df.toDF(*cleaned_names)

print("Cleaned columns preview:")
print(df.columns[:20])
print("Total columns:", len(df.columns))

# Separate label column from word-count columns
id_col = df.columns[0]
label_source_col = df.columns[-1]
show_df(df, df.columns[-1])

feature_cols = [
    col_name for col_name in df.columns
    if col_name not in [id_col, label_source_col]]

print("\nNumber of feature columns:", len(feature_cols))
print("Example feature columns:", feature_cols[:20])

df_model = df.select(
    F.col(id_col),
    F.col(label_source_col).cast("string").alias("target_text"),
    *[
        F.coalesce(F.col(c).cast("double"), F.lit(0.0)).alias(c)
        for c in feature_cols])

print("\nPrepared modelling dataframe:")
df_model.select(id_col, "target_text", *feature_cols[:10]).show(5,
                                                            truncate=False)

print("\nLabel distribution:")
df_model.groupBy("target_text").count().orderBy("target_text").show()
show_first_last_5_cols(df)

# split into train/test
train_df, test_df = df_model.randomSplit([0.8, 0.2], seed=42)

print("\nTraining rows:", train_df.count())
print("Testing rows:", test_df.count())

# Helpers
def evaluate_predictions(predictions, model_name, spam_label=1.0):
    evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction")

    accuracy = evaluator.setMetricName("accuracy").evaluate(predictions)
    weighted_precision = evaluator.setMetricName("weightedPrecision").evaluate(predictions)
    weighted_recall = evaluator.setMetricName("weightedRecall").evaluate(predictions)
    f1 = evaluator.setMetricName("f1").evaluate(predictions)

    spam_precision = (
        evaluator.setMetricName("precisionByLabel")
        .setMetricLabel(spam_label)
        .evaluate(predictions))
    spam_recall = (
        evaluator.setMetricName("recallByLabel")
        .setMetricLabel(spam_label)
        .evaluate(predictions))
    spam_f1 = (
        evaluator.setMetricName("fMeasureByLabel")
        .setMetricLabel(spam_label)
        .evaluate(predictions))

    print(f"\n--- {model_name} Results ---")
    print(f"Accuracy:            {accuracy:.4f}")
    print(f"Weighted Precision:  {weighted_precision:.4f}")
    print(f"Weighted Recall:     {weighted_recall:.4f}")
    print(f"Weighted F1:         {f1:.4f}")
    print(f"Spam Precision:      {spam_precision:.4f}")
    print(f"Spam Recall:         {spam_recall:.4f}")
    print(f"Spam F1:             {spam_f1:.4f}")

    return {
        "model": model_name,
        "accuracy": accuracy,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": f1,
        "spam_precision": spam_precision,
        "spam_recall": spam_recall,
        "spam_f1": spam_f1}

def common_stages():
    """Create fresh shared stages for each pipeline."""
    label_indexer = StringIndexer(
        inputCol="target_text",
        outputCol="label",
        stringOrderType="alphabetAsc")
    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="rawFeatures",
        handleInvalid="keep")
    return label_indexer, assembler

def train_idf_nb(model_name, model_type):
    """Train a Naive Bayes model using TF-IDF and word-frequency features."""
    label_indexer, assembler = common_stages()

    idf = IDF(
        inputCol="rawFeatures",
        outputCol="tfidfFeatures")

    nb = NaiveBayes(
        labelCol="label",
        featuresCol="tfidfFeatures",
        modelType=model_type,
        smoothing=1.0)

    pipeline = Pipeline(stages=[
        label_indexer,
        assembler,
        idf,
        nb])

    model = pipeline.fit(train_df)
    predictions = model.transform(test_df)

    print(f"\n{model_name} prediction preview:")

    predictions.select(
        id_col,
        "target_text",
        "label",
        "prediction",
        "probability"
    ).show(10, truncate=False)

    results = evaluate_predictions(predictions, model_name)
    return model, predictions, results

def train_bernoulli_nb():
    """Train Bernoulli Naive Bayes using binary word-presence features."""
    label_indexer, assembler = common_stages()

    binarizer = Binarizer(
        inputCol="rawFeatures",
        outputCol="binaryFeatures",
        threshold=0.0)

    nb = NaiveBayes(
        labelCol="label",
        featuresCol="binaryFeatures",
        modelType="bernoulli",
        smoothing=1.0)

    pipeline = Pipeline(stages=[
        label_indexer,
        assembler,
        binarizer,
        nb])

    model = pipeline.fit(train_df)
    predictions = model.transform(test_df)

    print("\nBernoulli Naive Bayes prediction preview:")

    predictions.select(
        id_col,
        "target_text",
        "label",
        "prediction",
        "probability"
    ).show(10, truncate=False)

    results = evaluate_predictions(predictions, "Bernoulli Naive Bayes")
    return model, predictions, results

# Model 1: Multinomial Naive Bayes with TF-IDF
multinomial_model, multinomial_predictions, multinomial_results = train_idf_nb(
    model_name="Multinomial Naive Bayes",
    model_type="multinomial")

# Model 2: Complement Naive Bayes with TF-IDF
gaussian_model, gaussian_predictions, gaussian_results = train_idf_nb(
    model_name="Complement Naive Bayes",
    model_type="gaussian")

# Model 3: Bernoulli Naive Bayes with binary word-presence features
bernoulli_model, bernoulli_predictions, bernoulli_results = train_bernoulli_nb()

# compare all models
results = [
    multinomial_results,
    gaussian_results,
    bernoulli_results]

results_df = spark.createDataFrame(results)

print("\nModel comparison:")
results_df.orderBy(F.desc("weighted_f1")).show(truncate=False)

# confusion matrix for best model
best_model_name = results_df.orderBy(F.desc("weighted_f1")).first()["model"]
print("\nBest model by weighted F1:", best_model_name)

# stop Spark
spark.stop()

