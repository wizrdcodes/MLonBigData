from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
from pyspark.sql.types import NumericType
from pyspark.sql.functions import isnan, col, when, count
import psutil

output = 1 # 1 = show outputs, 0 = hide outputs
show_outputs = (output == 1)

spark = SparkSession.builder \
                    .appName("LinearRegression_spark") \
                    .master("local[*]") \
                    .config("spark.executor.memory", "4g") \
                    .config("spark.driver.memory", "2g") \
                    .config("spark.executor.cores", "2") \
                    .config("spark.sql.inMemoryColumnarStorage.compressed", "true") \
                    .getOrCreate()

if show_outputs:
    print(f"CPU Usage: {psutil.cpu_percent()}%")
    print(f"Memory Usage: {psutil.virtual_memory().percent}%")

df = spark.read.csv('/Users/wizrdm/Desktop/UEL/Machine Learning on Big Data/'
                    'Diabetes (4).csv', header=True, inferSchema=True)

if show_outputs:
    df.show(10)
    print(f"Total Records: {df.count()}")
    print(f"Total Partitions: {df.rdd.getNumPartitions()}")
    df.printSchema()
    df.describe().show()
    print(f"CPU Usage after opening the csv file: {psutil.cpu_percent()}%")
    print(f"Memory Usage after csv file: {psutil.virtual_memory().percent}%")

# check missing / null values without error (due to isnan)
numeric_cols = [f.name for f in df.schema.fields if isinstance(f.dataType, NumericType)]
non_numeric_cols = [c for c in df.columns if c not in numeric_cols]
exprs = ([count(when(col(c).isNull() | isnan(col(c)), c)).alias(c) for c in numeric_cols]
    + [count(when(col(c).isNull(), c)).alias(c) for c in non_numeric_cols])
if show_outputs: df.select(exprs).show(truncate=False)

# convert categorical column into numbers
indexer = StringIndexer(inputCol = 'sex', outputCol = 'gender')

assembler = VectorAssembler(inputCols = ["age", "gender", "bmi","bp","tc","ldl","hdl","tch",
                                         "ltg","glu"],
                           outputCol = "Features")

scaler = StandardScaler(inputCol = "Features", outputCol = "scaled_Features")

regressor = LinearRegression(labelCol = 'progression', featuresCol = 'scaled_Features')

pipeline  = Pipeline(stages = [indexer,assembler,scaler,regressor])
#--Saving the Pipeline
pipeline.write().overwrite().save("pipeline_LRsaved_model")

pipelineModel = Pipeline.load('./pipeline_LRsaved_model')

data_train , data_test = df.randomSplit([0.7,0.3], seed = 123)

Model = pipeline.fit(data_train)
# --- Debug: see what each stage in the fitted pipeline actually is ---
for i, s in enumerate(Model.stages):
    print(i, type(s).__name__)

# --- Get the trained Linear Regression model (last stage) ---
lr_model = Model.stages[-1]   # safest: last stage is your regressor model

print("total LR coefficients", len(lr_model.coefficients))
print("coefficients", lr_model.coefficients)
print("intercept", lr_model.intercept)

pred = Model.transform(data_test)
pred.select('prediction', 'progression').show(10, truncate = False)

#create linear regression model.
Lasoregressor = LinearRegression(
    labelCol="progression",
    featuresCol="scaled_Features",
    elasticNetParam=1.0,
    regParam=0.1
)

Lasaopipeline = Pipeline(stages=[indexer, assembler, scaler, Lasoregressor])
LassoModel = Lasaopipeline.fit(data_train)

#n the prediction phase, we test our model on some unseen data.
lassopred = LassoModel.transform(data_test)
lassopred.select('prediction', 'progression').show(10, truncate = False)

#create linear regression model.
Ridgeregressor = LinearRegression(
    labelCol="progression",
    featuresCol="scaled_Features",
    elasticNetParam=0.0,
    regParam=0.1
)

Ridgepipeline = Pipeline(stages=[indexer, assembler, scaler, Ridgeregressor])
RidgeModel = Ridgepipeline.fit(data_train)

#n the prediction phase, we test our model on some unseen data.
Ridgepred = RidgeModel.transform(data_test)
Ridgepred.select('prediction', 'progression').show(10, truncate = False)

#Model Evaluation Spark Provides evaluation metrics
#for regression and classification tasks.
from pyspark.ml.evaluation import RegressionEvaluator
evaluator_mse = RegressionEvaluator(labelCol =
                                    'progression',
                                    predictionCol =
                                    'prediction',
                                    metricName =
                                    'mse')
# calculate MSE
mse1 = evaluator_mse.evaluate(pred)
mselasso = evaluator_mse.evaluate(lassopred)
mseridge = evaluator_mse.evaluate(Ridgepred)

evaluator_rmse = RegressionEvaluator(labelCol =
                                     'progression',
                                     predictionCol =
                                     'prediction',
                                     metricName =
                                     'rmse')
# calculate RMSE
rmse1 = evaluator_rmse.evaluate(pred)
rmse2_lasso = evaluator_rmse.evaluate(lassopred)
rmse3Ridge = evaluator_rmse.evaluate(Ridgepred)

evaluator_r2 = RegressionEvaluator(labelCol = 'progression',
                                   predictionCol = 'prediction',
                                   metricName = 'r2')
# calculate R_squared
r2_score1 = evaluator_r2.evaluate(pred)
r2_lasso = evaluator_r2.evaluate(lassopred)
r2_ridge = evaluator_r2.evaluate(Ridgepred)
# print the evaluation metrics
print('Regression - MSE: ', mse1, ', RMSE: ', rmse1, ', R^2: ', r2_score1)
print('Lasso - MSE: ', mselasso, ', RMSE: ', rmse2_lasso, ', R^2: ', r2_lasso)
print('Ridge - MSE: ', mseridge, ', RMSE: ', rmse3Ridge, ', R^2: ', r2_ridge)

# plot
import matplotlib.pyplot as plt
import numpy as np

mse = [mse1, mselasso, mseridge]
rmse = [rmse1, rmse2_lasso, rmse3Ridge]
r2_score = [r2_score1, r2_lasso, r2_ridge]

positions = np.arange(len(mse))
bar_width = 0.2

plt.bar(positions - bar_width, mse, width = bar_width, label = 'MSE')
plt.bar(positions, rmse, width = bar_width, label = 'RMSE')
plt.bar(positions + bar_width, r2_score, width = bar_width, label = 'R2_Score')

# adding labels and title
plt.xlabel('Model')
plt.ylabel('Scores')
plt.title('Comparison of Regression Metrics')

# adding the legend
plt.legend()
plt.xticks(positions, ['Regression', 'Lasso', 'Ridge'])
plt.show()

