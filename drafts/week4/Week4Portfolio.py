from pyspark.sql import SparkSession
from src.mlonbigdata import (
    show_df, describe_df, show_tight_layout, title_xy_labels,
    show_null_counts, show_vector_x_decimals, outputs_week_plots)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql.functions import col, isnan, when, count
from pyspark.ml.feature import QuantileDiscretizer, VectorAssembler, StandardScaler, ChiSqSelector
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

output = 1 # 1 = show outputs, 0 = hide outputs
show_outputs = (output == 1)
graph = 1 # 1 = show plots, 0 = hide plots
show_plots = (graph == 1)

WEEK = 4
PLOTS_FOLDER = outputs_week_plots(WEEK)

spark = SparkSession \
    .builder \
    .appName("Python Spark SQL basic example") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

file = '/Users/wizrdm/Desktop/UEL/Machine Learning on Big Data/titanic_synthetic_data.csv'

# Provided data
data = spark.read.csv(file, header=True, inferSchema=True)

# seaborn data
# data = sns.load_dataset("titanic")
# data = spark.createDataFrame(data)
# show_df(data,truncate=True)

columns = data.columns
if show_outputs:
    print(f"columns: {columns}") # see column list
    show_df(data,print_str=True)
    describe_df(data,"Pclass","Age","SibSp","Parch",print_str=True)
    describe_df(data,"Fare","Sex","Survived",print_str=True)
    describe_df(data, "Embarked_Q", "Embarked_S",print_str=True)

    print("showing null counts:")
    show_null_counts(data)

# rows where age is 0
age0 = data.filter(col("Age")==0).count()
print(f"num of rows where age is 0: {age0}")
print(f"percentage of rows where age is 0: {((age0/data.count())*100):.2f}%") # likely infants

cols = data.columns
cols.remove("Survived")
assembler = VectorAssembler(inputCols=cols, outputCol="features")
data = assembler.transform(data)
if show_outputs: show_df(data,"features", truncate=False, print_str=True)

# Scale features
standardscaler=(StandardScaler().setInputCol("features")
                .setOutputCol("Scaled_features"))
data=standardscaler.fit(data).transform(data)
if show_outputs:
    print("showing Scaled_features:")
    # show_df(data,"features","Scaled_features")
    # use my function to show less decimals
    show_vector_x_decimals(data.select("features","Scaled_features"),
        ["features","Scaled_features"],1)

train, test = data.randomSplit([0.8, 0.2], seed=42)

data_size = float(train.select("Survived").count())
numPos = train.select("Survived").where("Survived == 1").count()
per_ones = (float(numPos)/float(data_size))*100 # percent of ones over total
numNeg = float(data_size - numPos)
if show_outputs:
    print(f"The number of ones are: {numPos}")
    print(f"The percentage of ones (over total) is: {per_ones}%")

BalancingRatio = numNeg/data_size
if show_outputs: print(f"Balancing Ratio = {BalancingRatio}")

# balance data (makeshift weights using ratio)
train = train.withColumn("classWeights",
            when(train.Survived == 1, BalancingRatio).otherwise(1-BalancingRatio))
if show_outputs:
    print("showing classWeights:")
    show_df(train,"classWeights", rows=20)

from pyspark.sql.functions import approx_count_distinct, col

for c in ["Pclass","Age","SibSp","Parch","Fare","Sex","Embarked_Q","Embarked_S"]:
    data.select(approx_count_distinct(col(c)).alias(c)).show()

# Building a classification model using Logistic Regression (LR)
lr = LogisticRegression(labelCol="Survived", featuresCol="Scaled_features",
                        weightCol="classWeights",maxIter=10)
model=lr.fit(train)
predict_train=model.transform(train)
predict_test=model.transform(test)
if show_outputs:
    print("showing predictions:")
    show_df(predict_test,"Survived","prediction", rows=10)

# Evaluating the model
evaluator=BinaryClassificationEvaluator(rawPredictionCol='prediction',
                                        labelCol="Survived")
# We have only two choices: area under ROC and PR curves :-(
auroc = evaluator.evaluate(predict_test,
                           {evaluator.metricName: "areaUnderROC"})

if show_outputs:
    print("Area under ROC Curve: {:.4f}".format(auroc))
    show_df(predict_test,"Survived","prediction","probability", rows=15)

    print(model.summary)

pr = model.summary.pr.toPandas()
plt.plot(pr['recall'],pr['precision'])
plt.ylabel('Precision')
plt.xlabel('Recall')
if show_plots: show_tight_layout(show=True, save=True, folder=PLOTS_FOLDER)
if show_outputs:
    print("Model Accuracy",model.summary.accuracy)
    print("FalsePos rate",model.summary.falsePositiveRateByLabel)
    print("TruePos rate",model.summary.truePositiveRateByLabel)

print("Total True Positive (i.e. diabetes):",predict_test.select("Survived")
              .where('Survived == 1.0').count())
print("Total True  Negative (i.e. without diabetes):",predict_test.select("Survived")
              .where('Survived == 0.0').count())

pr = predict_test.toPandas()
TruePositive =0
FalsePositive=0
TrueNegative=0
FalseNegative=0
Postive=1.0
Negative=0.0
pos=0
Neg=0

if show_outputs: print("Total",len(pr["Survived"]))
for lbl in range(len(pr["Survived"])):
  if  pr["prediction"][lbl]==Postive:
    pos+=1
    if pr["prediction"][lbl]==pr["Survived"][lbl]:
      TruePositive+=1
    else:
      FalsePositive+=1
  if  pr["prediction"][lbl]==Negative:
    Neg+=1
    if pr["prediction"][lbl]==pr["Survived"][lbl]:
      TrueNegative+=1
    else:
      FalseNegative+=1

print("Total Positive & Negative predicted, diabetes: ",pos,",Non Diabetes",Neg)

if show_outputs:
    print("TruePostive",TruePositive,"FalsePostive",FalsePositive)
    print("TrueNegative",TrueNegative,"FalseNegative",FalseNegative)

TN = TrueNegative
FP = FalsePositive
FN = FalseNegative
TP = TruePositive
conf_matrix = np.array([[TN, FP],
                        [FN, TP]])
labels = ["Negative (0)", "Positive (1)"]

plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True,  cmap="Blues",
            xticklabels=labels, yticklabels=labels)
title_xy_labels("Confusion Matrix", "Predicted Label",
                "True Label")
if show_plots: show_tight_layout(show=True, save=True, folder=PLOTS_FOLDER)

precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) \
    if (precision + recall) > 0 else 0

if show_outputs:
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1_score:.4f}")

#