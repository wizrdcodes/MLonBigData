from pyspark.sql import SparkSession
from src.mlonbigdata import show_df, describe_df
from src.mlonbigdata import show_null_counts, show_x_decimals, show_vector_x_decimals
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql.functions import col, isnan, when, count
from pyspark.ml.feature import Imputer, VectorAssembler, StandardScaler, ChiSqSelector
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

output = 1 # 1 = show outputs, 0 = hide outputs
show_outputs = (output == 1)
graph = 0 # 1 = show plots, 0 = hide plots
show_plots = (graph == 1)
learn = 0 # 0 = show
learning = (learn == 0)

spark = SparkSession \
    .builder \
    .appName("Python Spark SQL basic example") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

file = '/Users/wizrdm/Desktop/UEL/Machine Learning on Big Data/diabetes.csv'
data = spark.read.csv(file, header=True, inferSchema=True)

columns = data.columns # see column list
if show_outputs:
    print(f"columns: {columns}")
    show_df(data,print_str=True)
    describe_df(data,"Pregnancies","Glucose","BloodPressure","Age",print_str=True)
    describe_df(data,"SkinThickness","Insulin",print_str=True)
    describe_df(data, "BMI","DiabetesPedigreeFunction", "Age",print_str=True)

    print("showing null counts:")
    show_null_counts(data)

# replace min value of zeros with Nan as data cleaning processs
data=data.withColumn("Glucose",when(data.Glucose==0,np.nan)
                     .otherwise(data.Glucose))
data=data.withColumn("BloodPressure",when(data.BloodPressure==0,np.nan)
                     .otherwise(data.BloodPressure))
data=data.withColumn("SkinThickness",when(data.SkinThickness==0,np.nan)
                     .otherwise(data.SkinThickness))
data=data.withColumn("BMI",when(data.BMI==0,np.nan).otherwise(data.BMI))
data=data.withColumn("Insulin",when(data.Insulin==0,np.nan)
                     .otherwise(data.Insulin))
if show_outputs:
    print("showing null counts after cleaning:")
    show_df(data,"Insulin","Glucose","BloodPressure",
                             "SkinThickness","BMI", print_str=True)

    print("showing null counts:")
    show_null_counts(data)

# replaces null values with mean
imputer=Imputer(
    inputCols=["Glucose","BloodPressure","SkinThickness","BMI","Insulin"],
    outputCols=["Glucose","BloodPressure","SkinThickness","BMI","Insulin"]
)
model=imputer.fit(data)
data=model.transform(data)
if show_outputs:
    print("showing data after imputation:")
    show_df(data)

cols = data.columns
cols.remove("Outcome")
assembler = VectorAssembler(inputCols=cols, outputCol="features")
data = assembler.transform(data)
if show_outputs: show_df(data,"features", truncate=False)

# Scale features
standardscaler=(StandardScaler().setInputCol("features")
                .setOutputCol("Scaled_features"))
data=standardscaler.fit(data).transform(data)
if show_outputs:
    print("showing Scaled_features:")
    show_df(data,"features","Scaled_features")
    # use my function to show less decimals
    show_vector_x_decimals(data.select("features","Scaled_features"),
        ["features","Scaled_features"],1)

train, test = data.randomSplit([0.8, 0.2], seed=42)

data_size = float(train.select("Outcome").count())
numPos = train.select("Outcome").where("Outcome == 1").count() # positive diabetes
per_ones = (float(numPos)/float(data_size))*100 # percent of positives
numNeg = float(data_size - numPos) # negative diabetes
if show_outputs:
    print("The number of ones are: ", numPos)
    print("The percentage of ones (positive for diabetes) is: ", per_ones, "%")

BalancingRatio = numNeg/data_size
if show_outputs: print(f"Balancing Ratio = {BalancingRatio}")

# balance data (makeshift weights using ratio)
train = train.withColumn("classWeights",
            when(train.Outcome == 1, BalancingRatio).otherwise(1-BalancingRatio))
if show_outputs:
    print("showing classWeights:")
    show_df(train,"classWeights", rows=20)

# Feature selection
css = ChiSqSelector(featuresCol='Scaled_features',outputCol='Aspect',
                    labelCol='Outcome',fpr=0.05)
train=css.fit(train).transform(train)
test=css.fit(test).transform(test)
if show_outputs:
    print("showing selected features:")
    show_df(test,"Aspect",truncate=False)

# Building a classification model using Logistic Regression (LR)
lr = LogisticRegression(labelCol="Outcome", featuresCol="Aspect",
                        weightCol="classWeights",maxIter=10)
model=lr.fit(train)
predict_train=model.transform(train)
predict_test=model.transform(test)
if show_outputs:
    print("showing predictions:")
    show_df(predict_test,"Outcome","prediction", rows=10)

#Evaluating the model
evaluator=BinaryClassificationEvaluator(rawPredictionCol='prediction',
                                        labelCol="Outcome")
# We have only two choices: area under ROC and PR curves :-(
auroc = evaluator.evaluate(predict_test,
                           {evaluator.metricName: "areaUnderROC"})

if learning: print("Area under ROC Curve: {:.4f}".format(auroc))
if learning: show_df(predict_test,"Outcome","prediction","probability", rows=15)

if learning: print(model.summary)
pr = model.summary.pr.toPandas()
plt.plot(pr['recall'],pr['precision'])
plt.ylabel('Precision')
plt.xlabel('Recall')
if show_plots: plt.show()
if learning:
    print("Model Accuracy",model.summary.accuracy)
    print("FalsePos rate",model.summary.falsePositiveRateByLabel)
    print("TruePos rate",model.summary.truePositiveRateByLabel)

#print("Total True Positive i.e. diabetes",predict_test.select("Outcome")
#               .where('Outcome == 1.0').count())
#print("Total True  Negative,i.e. without diabetes",predict_test.select("Outcome")
#               .where('Outcome == 0.0').count())
pr = predict_test.toPandas()
TruePositive =0
FalsePositive=0
TrueNegative=0
FalseNegative=0
Postive=1.0
Negative=0.0
pos=0
Neg=0

if learning: print("Total",len(pr["Outcome"]))
for lbl in range(len(pr["Outcome"])):
  if  pr["prediction"][lbl]==Postive:
    pos+=1
    if pr["prediction"][lbl]==pr["Outcome"][lbl]:
      TruePositive+=1
    else:
      FalsePositive+=1
  if  pr["prediction"][lbl]==Negative:
    Neg+=1
    if pr["prediction"][lbl]==pr["Outcome"][lbl]:
      TrueNegative+=1
    else:
      FalseNegative+=1
#print("Total Positive & Negative predicted,  diabetes: ",pos,",Non Diabetes",Neg)
if learning:
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
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
if show_plots: plt.show()

precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) \
    if (precision + recall) > 0 else 0

if learning:
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1_score:.4f}")

# ----------------------------
# Attica and Kata's solutions
# ----------------------------
#
# # Attica's solution:
# from pyspark.ml.feature import UnivariateFeatureSelector
#
# selector = UnivariateFeatureSelector(
#     featuresCol="Scaled_features",
#     outputCol="selected_features",
#     labelCol="Survived",
#     selectionMode="fpr"
# )
#
# selector.setFeatureType("continuous").setLabelType("categorical")
# selector.setSelectionThreshold(0.5)
#
# model=selector.fit(train)
# train=model.transform(train)
# test=model.transform(test)
#
# # Kata's solution:
# # Feature selection using chisquareSelector
# from pyspark.ml.feature import ChiSqSelector, VectorSlicer
#
# # Define indices of categorical/ordinal features suitable for ChiSqSelector
# # Original features: ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex', 'Embarked_Q', 'Embarked_S']
# # Exclude 'Age' (index 1) and 'Fare' (index 4) which are continuous and cause the error
# categorical_feature_indices = [0, 2, 3, 5, 6, 7] # Pclass, SibSp, Parch, Sex, Embarked_Q, Embarked_S
#
# # Drop 'ChiSq_Selected_features' column if it exists to avoid error on re-execution
# if "ChiSq_Selected_features" in train.columns:
#     train = train.drop("ChiSq_Selected_features")
# if "ChiSq_Selected_features" in test.columns:
#     test = test.drop("ChiSq_Selected_features")
#
# # Drop 'Aspect' column from the original train/test dataframes before slicing
# if "Aspect" in train.columns:
#     train = train.drop("Aspect")
# if "Aspect" in test.columns:
#     test = test.drop("Aspect")
#
# # Apply VectorSlicer to select only these features from Scaled_features
# slicer = VectorSlicer(inputCol="Scaled_features", outputCol="ChiSq_Selected_features", indices=categorical_feature_indices)
# train_sliced = slicer.transform(train)
# test_sliced = slicer.transform(test)
#
# css = ChiSqSelector(featuresCol='ChiSq_Selected_features', outputCol='Aspect', labelCol='Survived', fpr=0.05)
#
# # Fit and transform using the sliced dataframes
# train = css.fit(train_sliced).transform(train_sliced)
# test = css.fit(test_sliced).transform(test_sliced)
#
# test.select("Aspect").show(5,truncate=False)