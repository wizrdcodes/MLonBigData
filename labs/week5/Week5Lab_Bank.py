from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from src.mlonbigdata import (show_df, describe_df, show_df_stats,
                             title_xy_labels, show_tight_layout)
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.classification import (LogisticRegression, GBTClassifier,
    DecisionTreeClassifier, RandomForestClassifier, OneVsRest)
from pyspark.ml.evaluation import (MulticlassClassificationEvaluator,
    BinaryClassificationEvaluator)
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

output = 1 # 1 = show outputs, 0 = hide outputs
show_outputs = (output == 1)
graph = 1 # 1 = show plots, 0 = hide plots
show_plots = (graph == 1)
learn = 0
learning = (learn == 0)

# Initialize Spark session
spark = SparkSession.builder \
    .appName("Logistic Regression") \
    .getOrCreate()

dff = spark.read.csv('/Users/wizrdm/Desktop/UEL/'
        'Machine Learning on Big Data/bank.csv',inferSchema=True, header =True)

if show_outputs:
    print(f"row count: {dff.count()}") # count rows

# see possible values in deposit column
    show_df(dff, "deposit", distinct=True,
            print_str="Showing distinct deposit values: ")
    print("Within the 'deposit' column:")
    print("Instances of 'Yes':",
          dff.select("deposit").where("deposit=='yes'").count())
    print("Instances of 'No':",
          dff.select("deposit").where("deposit=='no'").count())
# info about dataframe
    show_df(dff, print_str="Showing full dataframe:")
    print("Showing schema:")
    dff.printSchema() # learn about the schema of dataframe

# show distinct values in string columns of df (using functional programming)
    for name, dtype in dff.dtypes:
        if dtype == 'string':
            print(f"Distinct values in {name.capitalize()} column:")
            show_df(dff, name, distinct=True)

# taking only integer features
numeric_features = [t[0] for t in dff.dtypes if t[1] == 'int']

if show_outputs:
    print(f"Showing numeric features: ")
    summary = dff.select(numeric_features).describe().toPandas().transpose()
    print(summary)

# find out if there are any null values in any column
if show_outputs:
    print("\nChecking for null values:")
    for col in dff.columns:
        print(col,"Total null values: ",dff.where(dff[col].isNull()).count())

df = dff.drop('day', 'month')
df_cols = df.columns

if show_outputs:
    print("\nShowing schema of df after dropping 'day' and 'month' columns:")
    df.printSchema()

# apply string indexing to categorial variables using pipeline
categorColumns = ['job', 'marital', 'education', 'default', 'housing', 'loan',
                      'contact', 'poutcome']
stages = []
for categorCol in categorColumns:
    strIndexer = StringIndexer(inputCol=categorCol,
                               outputCol=categorCol+'Index') # '[col]Index'
    encoder = OneHotEncoder(inputCols=[strIndexer.getOutputCol()],
                    outputCols=[categorCol + "classVec"]) # list of '[col]classVec'

    if show_outputs: print(strIndexer.getOutputCol())

    stages += [strIndexer, encoder]
label_stringIdx = StringIndexer(inputCol = 'deposit', outputCol = 'label')

if show_outputs: print(f"\nString Indexer: {label_stringIdx}")

stages += [label_stringIdx]
numericCols = ['age','balance', 'duration', 'campaign', 'pdays', 'previous']
assemblerInputs = [c + "classVec" for c in categorColumns] + numericCols

if show_outputs: print("\n",assemblerInputs)

assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
stages += [assembler]

pipeline = Pipeline(stages = stages)
pipelineModel = pipeline.fit(df)
df = pipelineModel.transform(df)

if show_outputs:
    show_df(df, truncate=True, rows=12, print_str="\nShowing pipeline df:")
    show_df(df.select("features"), rows=1,
            print_str="\nShowing example of features vector:")
    print("\nPipeline columns:",df.columns)

# feature vector is in features from assembler.
selectedCols = ['label', 'features'] + df_cols
dfSelected = df.select(selectedCols)

if show_outputs:
    print("\nShowing schema of selected columns:")
    dfSelected.printSchema() # label, features + most columns (minus day/month)

# code for multiple classification using logistic Regression
train, test = dfSelected.randomSplit([0.7, 0.3], seed = 2018)
lr = LogisticRegression(maxIter=100, \
                        featuresCol="features", \
                        labelCol='label')
ovr = OneVsRest(classifier=lr, \
                labelCol='label', \
                featuresCol='features')

ovrModel = ovr.fit(train)
predictionsovr = ovrModel.transform(test)

if show_outputs:
    show_df(predictionsovr, rows=10, print_str="Showing predictions:")

# Create MulticlassMetrics object
# Convert predictions and labels to RDD for MulticlassMetrics
prediction_and_labels = predictionsovr.select("prediction", "label") \
    .withColumnRenamed("indexedLabel", "label") \
    .toPandas()  # Convert to Pandas DataFrame for easier manipulation
# Create a confusion matrix using Pandas
confusion_matrix = pd.crosstab(prediction_and_labels['label'],
    prediction_and_labels['prediction'],rownames=['Actual'],colnames=['Predicted'])

# Plot the confusion matrix using Seaborn and Matplotlib
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
title_xy_labels("Confusion Matrix", "Predicted", "Actual")
show_tight_layout(show_plots, save=False)

multi_evaluator = MulticlassClassificationEvaluator(\
labelCol="label", predictionCol="prediction",\
metricName="accuracy")
accuracy = multi_evaluator.evaluate(predictionsovr)

if show_outputs: print(f"Test accuracy =  {accuracy}")

train, test = dfSelected.randomSplit([0.7, 0.3], seed = 2018)

if show_outputs:
    print(f"Training Dataset Count: {train.count()}")
    print(f"Test Dataset Count: {test.count()}")

dt = DecisionTreeClassifier(featuresCol='features',labelCol='label',maxDepth=3)
dtModel = dt.fit(train)
predictions = dtModel.transform(test)

if show_outputs:
    show_df(predictions, 'age','job','label','rawPrediction','prediction',
            'probability', rows= 10, print_str='Decision Tree Predictions:')
    print("Total Actual Positive:",predictions.select("label")
                       .where('label == 1.0').count())
    print("Total Actual Negative:",predictions.select("label")
                       .where('label == 0.0').count())

pr = predictions.toPandas()
TruePositive =0
FalsePositive=0
TrueNegative=0
FalseNegative=0
Postive=1.0
Negative=0.0
pos=0
neg=0

if show_outputs: print("Total Predictions:",len(pr["label"]))

for lbl in range(len(pr["label"])):
  if pr["prediction"][lbl]==Postive:
    pos+=1
    if pr["prediction"][lbl]==pr["label"][lbl]:
      TruePositive+=1
    else:
      FalsePositive+=1
  if pr["prediction"][lbl]==Negative:
    neg+=1
    if pr["prediction"][lbl]==pr["label"][lbl]:
      TrueNegative+=1
    else:
      FalseNegative+=1

if show_outputs:
    print(f"Total Positive & Negative Predictions: Pos: {pos}, Neg: {neg}")
    print(f"TruePostive: {TruePositive}, FalsePostive: {FalsePositive}")
    print(f"TrueNegative: {TrueNegative}, FalseNegative: {FalseNegative}")

# Evaluate Decision Tree model.
binary_evaluator = BinaryClassificationEvaluator()

if show_outputs: print("Test Area Under ROC: "+
    str(binary_evaluator.evaluate(predictions,
                           {binary_evaluator.metricName: "areaUnderROC"})))

# RandomForest
rf = RandomForestClassifier(featuresCol='features', labelCol='label',
                            maxDepth = 4,numTrees=20)
train, test = dfSelected.randomSplit([0.7, 0.3], seed = 2018)
rfModel = rf.fit(train)
predictions = rfModel.transform(test)
pr = predictions.toPandas()

if show_outputs:
    print("Total Actual Positive:",predictions.select("label")
          .where('label == 1.0').count())
    print("Total Actual Negative:",predictions.select("label")
          .where('label == 0.0').count())
    print("Test Area Under ROC: "+str(binary_evaluator.evaluate(predictions,
                        {binary_evaluator.metricName: "areaUnderROC"})))

TruePositive =0
FalsePositive=0
TrueNegative=0
FalseNegative=0
Postive=1.0
Negative=0.0
pos=0
neg=0

if show_outputs: print("Total",len(pr["label"]))

for lbl in range(len(pr["label"])):
  if pr["prediction"][lbl]==Postive:
    pos+=1
    if pr["prediction"][lbl]==pr["label"][lbl]:
      TruePositive+=1
    else:
      FalsePositive+=1
  if pr["prediction"][lbl]==Negative:
    neg+=1
    if pr["prediction"][lbl]==pr["label"][lbl]:
      TrueNegative+=1
    else:
      FalseNegative+=1

if show_outputs:
    print(f"Total positive & Negative in Predictions: Pos: {pos}, Neg: {neg}")
    print(f"TruePostive: {TruePositive}, FalsePostive: {FalsePositive}")
    print(f"TrueNegative: {TrueNegative}, FalseNegative: {FalseNegative}")

ml = GBTClassifier(maxIter=10,featuresCol='features',labelCol='label',maxDepth=10)
train, test = dfSelected.randomSplit([0.7, 0.3], seed = 2018)
mlModel = ml.fit(train)
predictions = mlModel.transform(test)

binary_evaluator = BinaryClassificationEvaluator()

if show_outputs:
    print("Test Area Under ROC: "+str(binary_evaluator.evaluate(predictions,
                        {binary_evaluator.metricName: "areaUnderROC"})))

# Calculate accuracy and F-1 score
accuracy_evaluator = MulticlassClassificationEvaluator(metricName='accuracy')
accuracy = accuracy_evaluator.evaluate(predictions.select('label', 'prediction'))

f1_score_evaluator = MulticlassClassificationEvaluator(metricName='f1')
f1_score = f1_score_evaluator.evaluate(predictions.select('label', 'prediction'))

if show_outputs:
    print("GBT Classifier:")
    print(accuracy, f1_score)

gbt = GBTClassifier(maxIter=10)
paramGrid = (ParamGridBuilder()
             .addGrid(gbt.maxDepth, [2, 4, 6])
             .addGrid(gbt.maxBins, [20, 60])
             .addGrid(gbt.maxIter, [10, 20])
             .build())
cv = CrossValidator(estimator=gbt, estimatorParamMaps=paramGrid,
                    evaluator=binary_evaluator, numFolds=5)
# Run cross validations. This can take about 6 minutes since it is
# training over 20 trees!
cvModel = cv.fit(train)
predictions = cvModel.transform(test)
binary_evaluator.evaluate(predictions)

if show_outputs: predictions.select('label', 'prediction').show(13)

