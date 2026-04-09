from pyspark.sql import SparkSession
from src.mlonbigdata import (show_df, describe_df, show_df_stats,
                             title_xy_labels, show_tight_layout)
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import OneVsRest
from pyspark.ml.classification import LogisticRegression
from pyspark.mllib.evaluation import MulticlassMetrics
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

output = 1 # 1 = show outputs, 0 = hide outputs
show_outputs = (output == 1)
graph = 1 # 1 = show plots, 0 = hide plots
show_plots = (graph == 1)

spark = SparkSession.builder.master("local[*]").getOrCreate()

data = spark.read.csv(
    '/Users/wizrdm/Desktop/UEL/Machine Learning on Big Data'
    '/bezdekIris.data.txt',
    inferSchema=True, header =True).toDF(
    "sep_len", "sep_wid", "pet_len", "pet_wid", "label")

if show_outputs:
    show_df(data, "label", rows=5, print_str=True)
    describe_df(data, print_str=True)
    show_df_stats(data)

vectorAssembler = VectorAssembler(inputCols=["sep_len", "sep_wid",
                        "pet_len", "pet_wid"], outputCol="features")

df_temp = vectorAssembler.transform(data)
if show_outputs: show_df(df_temp, print_str=True)

df = df_temp.drop("sep_len", "sep_wid", "pet_len", "pet_wid")
if show_outputs: show_df(df, rows=3, print_str=True)

labelIndexer = StringIndexer(inputCol="label", outputCol="labelIndex")
df = labelIndexer.fit(df).transform(df)
if show_outputs: show_df(df, "label", "labelIndex", rows=3,
                         distinct=True, print_str="Showing df after labelIndexer:")

(trainingData, testData) = df.randomSplit([0.7, 0.3], seed=42)

dt = DecisionTreeClassifier(labelCol="labelIndex", featuresCol="features",
                            impurity='entropy', maxDepth=4,seed=1234)
model = dt.fit(trainingData)
predictions = model.transform(testData)

evaluator = MulticlassClassificationEvaluator(
    labelCol="labelIndex", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test Accuracy = ", accuracy)
print(model.toDebugString)

train, test = df.randomSplit([0.7, 0.3], seed=42)
lr = LogisticRegression(maxIter=100, featuresCol="features", labelCol="labelIndex",
                        predictionCol="prediction")
ovr = OneVsRest(classifier=lr, labelCol="labelIndex", featuresCol="features")

#from pyspark.ml import Pipeline
#pipeline_ovr = Pipeline(stages=[vecAssembler, stdScaler, ovr])
#pipelineModel_ovr = pipeline_ovr.fit(trainDF)

ovrModel = ovr.fit(train)
predictionsovr = ovrModel.transform(test)

predictionsovr.show(truncate=False)

prediction_and_labels = (predictionsovr.select("prediction", "labelIndex")
                         .withColumnRenamed("labelIndex", "label")
                         .toPandas())

confusion_matrix = pd.crosstab(prediction_and_labels["label"],
                    prediction_and_labels["prediction"], rownames=["Actual"],
                    colnames=["Predicted"])
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues",cbar=False)
title_xy_labels("Confusion Matrix", "Predicted",
                "Actual")
show_tight_layout(show_plots, save=False)

accuracy_ovr = evaluator.evaluate(predictionsovr)
print("Test Accuracy = ", accuracy_ovr)

# Test Accuracy = 0.9782608695652174

