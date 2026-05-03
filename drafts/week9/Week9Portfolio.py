#%%

from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pyspark.ml import Pipeline
from pyspark.ml.classification import LinearSVC, LogisticRegression
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)
from pyspark.ml.feature import HashingTF, IDF, StopWordsRemover, Tokenizer
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, lower, regexp_replace, trim

# Plot style
plt.rcParams.update({
    "figure.figsize": (8, 5),
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,})
sns.set_palette("tab10")

#%%

spark = (
    SparkSession.builder
    .master("local[*]")
    .appName("Week9_IMDB_LinearSVM_TextClassification")
    .config("spark.driver.memory", "4g")
    .getOrCreate())

spark.sparkContext.setLogLevel("ERROR")

print(f"Spark {spark.version} started.")
print(f"Master: {spark.sparkContext.master}")
print(f"App:    {spark.sparkContext.appName}")

#%%

DATASET_ROOT = Path("/Users/wizrdm/Desktop/UEL/Machine Learning on Big Data/aclImdb")

#%%

def folder_uri(folder_path):
    """Return a normal local path for Spark."""
    return str(folder_path.resolve())

def load_review_folder(split_name, sentiment_name, label_value):
    """Load all .txt reviews from one IMDB split/sentiment folder."""
    review_folder = DATASET_ROOT / split_name / sentiment_name
    if not review_folder.exists():
        raise FileNotFoundError(f"Missing expected review folder: {review_folder}")
    txt_files = list(review_folder.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(
            f"No .txt files found in: {review_folder}\n"
            "Check that this folder contains the IMDB review text files.")
    file_pattern = str(review_folder.resolve() / "*.txt")
    review_rdd = spark.sparkContext.wholeTextFiles(file_pattern)

    review_df = spark.createDataFrame(review_rdd, ["file_path", "text_raw"])
    review_df = (
        review_df
        .withColumn("split", lit(split_name))
        .withColumn("sentiment", lit(sentiment_name))
        .withColumn("label", lit(float(label_value))))
    return review_df

train_pos_df = load_review_folder("train", "pos", 1.0)
train_neg_df = load_review_folder("train", "neg", 0.0)
test_pos_df = load_review_folder("test", "pos", 1.0)
test_neg_df = load_review_folder("test", "neg", 0.0)

train_raw_df = train_pos_df.unionByName(train_neg_df)
test_raw_df = test_pos_df.unionByName(test_neg_df)
raw_df = train_raw_df.unionByName(test_raw_df)

print("\nSchema:")
raw_df.printSchema()

print("\nSample rows:")
raw_df.select("split", "sentiment", "label", "file_path", "text_raw").show(5, truncate=80)

print(f"\nTotal reviews loaded: {raw_df.count():,}")

#%%

# Dataset distribution
split_dist_pd = (
    raw_df.groupBy("split", "sentiment", "label")
    .count()
    .orderBy("split", "sentiment")
    .toPandas())

print("\nDataset distribution:")
print(split_dist_pd.to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(
    data=split_dist_pd,
    x="split",
    y="count",
    hue="sentiment",
    ax=ax,)
ax.set_title("IMDB Review Counts by Split and Sentiment", fontweight="bold")
ax.set_xlabel("Dataset Split")
ax.set_ylabel("Number of Reviews")
for container in ax.containers:
    ax.bar_label(container, fmt="%d", padding=3)
plt.tight_layout()
plt.show()

#%%

# Text cleaning
clean_df = (
    raw_df
    .withColumn("text", lower(col("text_raw")))
    .withColumn("text", regexp_replace(col("text"), r"<br\s*/?>", " "))
    .withColumn("text", regexp_replace(col("text"), r"[^a-z\s]", " "))
    .withColumn("text", regexp_replace(col("text"), r"\s+", " "))
    .withColumn("text", trim(col("text")))
    .select("split", "sentiment", "label", "file_path", "text")
    .na.drop())

train_df = clean_df.filter(col("split") == "train").select("text", "label")
test_df = clean_df.filter(col("split") == "test").select("text", "label")

n_train = train_df.count()
n_test = test_df.count()

print("\nPreprocessing complete.")
print(f"Training reviews: {n_train:,}")
print(f"Testing reviews:  {n_test:,}")

print("\nCleaned text preview:")
clean_df.select("split", "sentiment", "label", "text").show(5, truncate=100)

#%%

# Train/test split visualisation
split_counts = pd.DataFrame({
    "Split": ["Train", "Test"],
    "Count": [n_train, n_test],})

fig, ax = plt.subplots(figsize=(7, 3.5))
bars = ax.bar(split_counts["Split"], split_counts["Count"])
ax.set_title("Official IMDB Train / Test Split", fontweight="bold")
ax.set_xlabel("Split")
ax.set_ylabel("Number of Reviews")
for bar, value in zip(bars, split_counts["Count"]):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 200,
        f"{value:,}",
        ha="center",
        va="bottom",)
plt.tight_layout()
plt.show()

#%%

# Define TF-IDF pipeline stages
tokenizer = Tokenizer(inputCol="text", outputCol="tokens")
stop_rm = StopWordsRemover(inputCol="tokens", outputCol="filtered")
hash_tf = HashingTF(inputCol="filtered", outputCol="tf", numFeatures=2**18)
idf = IDF(inputCol="tf", outputCol="features")

# Fit on training data only to avoid leakage
featurizer = Pipeline(stages=[tokenizer, stop_rm, hash_tf, idf])
feat_model = featurizer.fit(train_df)
train_fe = feat_model.transform(train_df)

print("\nTF-IDF vectors created.")
print(f"Feature dimensions: {hash_tf.getNumFeatures():,}")
print("\nSample sparse feature vectors:")
train_fe.select("text", "features").show(3, truncate=80)

#%%

# Define classifiers
lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=50)
svm = LinearSVC(featuresCol="features", labelCol="label", maxIter=50, regParam=0.1)

# Logistic Regression baseline
print("Training Logistic Regression baseline...")
t0 = time.time()
pipeline_lr = Pipeline(stages=[tokenizer, stop_rm, hash_tf, idf, lr])
lr_model = pipeline_lr.fit(train_df)
lr_pred = lr_model.transform(test_df)
lr_time = time.time() - t0
print(f"Logistic Regression completed in {lr_time:.1f}s")

# Linear SVM
print("\nTraining Linear SVM...")
t0 = time.time()
pipeline_svm = Pipeline(stages=[tokenizer, stop_rm, hash_tf, idf, svm])
svm_model = pipeline_svm.fit(train_df)
svm_pred = svm_model.transform(test_df)
svm_time = time.time() - t0
print(f"Linear SVM completed in {svm_time:.1f}s")

#%%

# Evaluators
acc_eval = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy",)
f1_eval = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="f1",)
weighted_precision_eval = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="weightedPrecision",)
weighted_recall_eval = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="weightedRecall",)
auc_eval = BinaryClassificationEvaluator(
    labelCol="label",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC",)


def evaluate_model(predictions, model_name, train_time):
    return {
        "Model": model_name,
        "Accuracy": acc_eval.evaluate(predictions),
        "Weighted Precision": weighted_precision_eval.evaluate(predictions),
        "Weighted Recall": weighted_recall_eval.evaluate(predictions),
        "F1 Score": f1_eval.evaluate(predictions),
        "AUC-ROC": auc_eval.evaluate(predictions),
        "Train Time (s)": train_time,}


results = pd.DataFrame([
    evaluate_model(lr_pred, "Logistic Regression", lr_time),
    evaluate_model(svm_pred, "Linear SVM", svm_time),])

print("\nEvaluation Results")
for _, row in results.iterrows():
    print(f"\n{row['Model']}")
    print(f"Accuracy:           {row['Accuracy']:.4f}")
    print(f"Weighted Precision: {row['Weighted Precision']:.4f}")
    print(f"Weighted Recall:    {row['Weighted Recall']:.4f}")
    print(f"F1 Score:           {row['F1 Score']:.4f}")
    print(f"AUC-ROC:            {row['AUC-ROC']:.4f}")
    print(f"Train Time:         {row['Train Time (s)']:.1f}s")

#%%

# Bar chart comparison
metrics = ["Accuracy", "Weighted Precision", "Weighted Recall", "F1 Score", "AUC-ROC"]
x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(11, 5))
lr_values = [results.loc[results["Model"] == "Logistic Regression", metric].iloc[0] for metric in metrics]
svm_values = [results.loc[results["Model"] == "Linear SVM", metric].iloc[0] for metric in metrics]

bars1 = ax.bar(x - width / 2, lr_values, width, label="Logistic Regression")
bars2 = ax.bar(x + width / 2, svm_values, width, label="Linear SVM")

for bar in list(bars1) + list(bars2):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.003,
        f"{bar.get_height():.3f}",
        ha="center",
        va="bottom",
        fontsize=9,)

ax.set_ylim(0.0, 1.05)
ax.set_xticks(x)
ax.set_xticklabels(metrics, rotation=20, ha="right")
ax.set_ylabel("Score")
ax.set_title("Model Comparison: Logistic Regression vs Linear SVM", fontweight="bold")
ax.legend()
ax.yaxis.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

#%%

def plot_confusion_matrix(pred_df, model_name, ax):
    rdd = pred_df.select("prediction", "label").rdd.map(
        lambda row: (float(row[0]), float(row[1])))
    cm_metrics = MulticlassMetrics(rdd)
    cm = cm_metrics.confusionMatrix().toArray()
    cm_norm = cm / cm.sum(axis=1, keepdims=True)

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2%",
        cmap="Blues",
        ax=ax,
        xticklabels=["Pred: Negative", "Pred: Positive"],
        yticklabels=["True: Negative", "True: Positive"],
        linewidths=0.5,
        linecolor="grey",
        cbar=False,)

    for i in range(2):
        for j in range(2):
            ax.text(
                j + 0.5,
                i + 0.72,
                f"n={int(cm[i, j]):,}",
                ha="center",
                va="center",
                fontsize=9,
                color="grey",
            )

    ax.set_title(model_name, fontweight="bold", fontsize=12)
    return cm


fig, axes = plt.subplots(1, 2, figsize=(13, 5))
cm_lr = plot_confusion_matrix(lr_pred, "Logistic Regression", axes[0])
cm_svm = plot_confusion_matrix(svm_pred, "Linear SVM", axes[1])

plt.suptitle("Confusion Matrices - Row Normalised", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

print("\nLogistic Regression raw confusion matrix counts:")
print(cm_lr.astype(int))
print("\nLinear SVM raw confusion matrix counts:")
print(cm_svm.astype(int))

#%%

def compute_roc(pred_df, pos_class_idx=1):
    roc_pd = (
        pred_df.select("rawPrediction", "label").rdd
        .map(lambda row: (float(row["rawPrediction"][pos_class_idx]), float(row["label"])))
        .toDF(["score", "label"])
        .toPandas()
        .sort_values("score", ascending=False))

    positives = (roc_pd["label"] == 1.0).sum()
    negatives = (roc_pd["label"] == 0.0).sum()

    tpr = []
    fpr = []
    true_positives = 0
    false_positives = 0
    previous_score = None

    for score, label in zip(roc_pd["score"], roc_pd["label"]):
        if previous_score is None or score != previous_score:
            tpr.append(true_positives / positives if positives else 0.0)
            fpr.append(false_positives / negatives if negatives else 0.0)
            previous_score = score

        if label == 1.0:
            true_positives += 1
        else:
            false_positives += 1

    tpr.append(1.0)
    fpr.append(1.0)
    return np.array(fpr), np.array(tpr)


fpr_lr, tpr_lr = compute_roc(lr_pred)
fpr_svm, tpr_svm = compute_roc(svm_pred)

lr_auc = results.loc[results["Model"] == "Logistic Regression", "AUC-ROC"].iloc[0]
svm_auc = results.loc[results["Model"] == "Linear SVM", "AUC-ROC"].iloc[0]

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr_lr, tpr_lr, lw=2, label=f"Logistic Regression (AUC = {lr_auc:.3f})")
ax.plot(fpr_svm, tpr_svm, lw=2, label=f"Linear SVM (AUC = {svm_auc:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.5, label="Random (AUC = 0.500)")

ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves - Logistic Regression vs Linear SVM", fontweight="bold")
ax.legend(loc="lower right")
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.02])
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()

#%%

spark.stop()
print("\nSpark session stopped.")

