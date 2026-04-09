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
learn = 0
learning = (learn == 0)

spark = (SparkSession.builder
         .master("local[*]")
         .config("spark.driver.memory", "4g")
         .getOrCreate())

data = spark.read.csv(
    '/Users/wizrdm/Desktop/UEL/Machine Learning on Big Data/'
    'customer_segmentation_dataset.csv',
    inferSchema=True, header=True).toDF(
    "age", "annual_income", "spending_score", "years_as_customer",
    "number_of_purchases", "customer_segment")

if show_outputs:
    show_df(data, rows=5, print_str=True)
    describe_df(data, print_str=True)
    show_df_stats(data, print_str=True)

vectorAssembler = VectorAssembler(inputCols=["age", "annual_income",
                "spending_score","years_as_customer","number_of_purchases"],
                                  outputCol="features")

df_temp = vectorAssembler.transform(data)
if show_outputs: show_df(df_temp, print_str="Data + features:")

df = df_temp.drop("age", "annual_income", "spending_score", "years_as_customer",
                  "number_of_purchases")
if show_outputs: show_df(df, rows=5, print_str="Target + features:")

labelIndexer = StringIndexer(inputCol="customer_segment", outputCol="value_tier")
df = labelIndexer.fit(df).transform(df)
if show_outputs: show_df(df, "customer_segment", "value_tier",
                         distinct=True, print_str="Indexed target column:")

(trainingData, testData) = df.randomSplit([0.7, 0.3], seed=42)

dt = DecisionTreeClassifier(labelCol="value_tier", featuresCol="features",
                            impurity='entropy', maxDepth=4,seed=1234)
model = dt.fit(trainingData)
predictions = model.transform(testData)

evaluator = MulticlassClassificationEvaluator(
    labelCol="value_tier", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
if show_outputs:
    print("Test Accuracy = ", accuracy)
    # print(model.toDebugString)

train, test = df.randomSplit([0.7, 0.3], seed=42)
lr = LogisticRegression(maxIter=100, featuresCol="features", labelCol="value_tier",
                        predictionCol="prediction")
ovr = OneVsRest(classifier=lr, labelCol="value_tier", featuresCol="features")

#from pyspark.ml import Pipeline
#pipeline_ovr = Pipeline(stages=[vecAssembler, stdScaler, ovr])
#pipelineModel_ovr = pipeline_ovr.fit(trainDF)

ovrModel = ovr.fit(train)
predictionsovr = ovrModel.transform(test)

if show_outputs: predictionsovr.show(5, truncate=False)

prediction_and_labels = (predictionsovr.select("prediction", "value_tier")
                .withColumnRenamed("value_tier", "customer_segment")
                .toPandas())

confusion_matrix = pd.crosstab(prediction_and_labels["customer_segment"],
                    prediction_and_labels["prediction"], rownames=["Actual"],
                    colnames=["Predicted"])
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues",cbar=False)
title_xy_labels("Confusion Matrix", "Predicted",
                "Actual")
show_tight_layout(show_plots, save=True, folder="plots")

evaluator = MulticlassClassificationEvaluator(
    labelCol="value_tier", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictionsovr)
if show_outputs: print("Test Accuracy = ", accuracy)

