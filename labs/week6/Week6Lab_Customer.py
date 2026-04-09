from pyspark.sql import SparkSession
from pyspark.sql.types import NumericType
from pyspark.sql.functions import col, when, count
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.stat import Correlation
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from src.mlonbigdata import show_df, title_xy_labels, show_tight_layout

output = 0 # 1 = show outputs, 0 = hide outputs
show_outputs = (output == 1)
graph = 0 # 1 = show plots, 0 = hide plots
show_plots = (graph == 1)

spark = SparkSession.builder\
    .master("local")\
    .appName("ClusteringApp")\
    .enableHiveSupport()\
    .config("spark.ui.port", "4050")\
    .getOrCreate()

df = spark.read.csv('/Users/wizrdm/Desktop/UEL/Machine '
            'Learning on Big Data/Customer.csv', inferSchema=True, header=True)

if show_outputs:
    show_df(df, rows=5, print_str="Showing data sample:")
    print("Schema of df:")
    df.printSchema()
    print(f"row count of data: {df.count()}")

null_counts = [
    count(when(col(c).isNull(), c)).alias(c)
    for c in df.columns]
if show_outputs:
    print("Number of null values in each column:")
    df.select(null_counts).show()

# counting how many times zero has appeared in a column
numeric_cols = [f.name for f in df.schema.fields if isinstance(f.dataType,
                                                               NumericType)]
if show_outputs: print("Number of zeroes in each column:")
for c in numeric_cols:
    if show_outputs: print(c,df.filter(col(c)==0).count())

pddf = df.toPandas()

plt.figure(figsize=(15, 8))
sns.heatmap(round(pddf[numeric_cols].corr(method='spearman'), 2),
            annot=True, mask=None, cmap='GnBu')
plt.title("Numeric Correlation Matrix")
show_tight_layout(show=show_plots, save=False, folder="plots")

# Calculate Correlation Using Using MLlib
# Assemble feature vector
# Define the feature and label columns & Assemble the feature vector
vector_assembler = VectorAssembler(inputCols=numeric_cols[1:12],
                                   outputCol="features")
data_vector = vector_assembler.transform(df).select("features")

# Calculate correlation
correlation_matrix = Correlation.corr(data_vector, "features").head()[0]

if show_outputs: print("Correlation features[1:12] ", correlation_matrix[0, 2])
df2 = spark.createDataFrame(
    correlation_matrix.toArray().tolist(),
    numeric_cols[1:12])

plt.figure(figsize=(15, 8))
sns.heatmap(df2.toPandas(),annot=True, mask=None, cmap='GnBu')
show_tight_layout(show=show_plots, save=False, folder="plots")

ClusteringColumns = ['PURCHASES_FREQUENCY', 'ONEOFF_PURCHASES_FREQUENCY',
                     'PURCHASES_INSTALLMENTS_FREQUENCY']

assembler = VectorAssembler(inputCols=ClusteringColumns, outputCol="features")
featureDf = assembler.transform(df)
if show_outputs:
    print("Schema of featureDf:")
    featureDf.printSchema()
    show_df(featureDf, rows=5)

if show_outputs: print(f"row count of featureDf: {featureDf.count()}")
trainingData, testData = featureDf.randomSplit([0.7, 0.3], seed = 42)
if show_outputs:
    print(f"Training Dataset Count: {trainingData.count()}")
    print(f"Test Dataset Count: {testData.count()}")
    trainingData.show(10)
    print(testData.columns)

cluster_count = [3, 5, 10, 20, 30, 40]
silhouette_values =[]

best_k = None
best_sil = -1
best_predictDf = None
best_model = None

for i in cluster_count:
    kmeans = (KMeans()
              .setK(i)
              .setSeed(1)
              .setFeaturesCol("features")
              .setPredictionCol("prediction"))
    kmeansModel = kmeans.fit(trainingData)

    # test the model with test data set
    predictDf = kmeansModel.transform(testData)

    evaluator = ClusteringEvaluator()
    silhouette = evaluator.evaluate(predictDf)

    silhouette_values.append(silhouette)

    if show_outputs: print(f"clusters = {i}, "
        f"Silhouette with squared euclidean distance = {silhouette:.5f}")
    if silhouette > best_sil:
        best_sil = silhouette
        best_k = i
        best_predictDf = predictDf
        best_model = kmeansModel
        if show_outputs:
            print(f"New best k = {best_k} with silhouette = {best_sil}")

# Plotting WSSSE values
plt.figure(figsize=(8,5))
plt.plot(cluster_count, silhouette_values, marker='o')
title_xy_labels('Silhouette for Optimal K','Number of Clusters (K)',
                'Silhouette Score')
plt.grid()
show_tight_layout(show=show_plots, save=False, folder="plots")

# test the model with test data set
if show_outputs:
    show_df(best_predictDf, rows=5,
            print_str=f"Showing predictions for {best_k} clusters:")
    print(f"Silhouette with squared euclidean distance = {best_sil}")

if show_outputs: show_df(best_predictDf.groupBy("prediction").count(),
                         print_str=f"Showing count of each cluster:")

clustered_data_pd = best_predictDf.toPandas()

# Visualizing the results
plt.scatter(clustered_data_pd["PURCHASES_FREQUENCY"],
            clustered_data_pd["ONEOFF_PURCHASES_FREQUENCY"],
            c=clustered_data_pd["prediction"], cmap='viridis')
title_xy_labels('K-means Clustering with PySpark MLlib',
                "Purchase Frequency","One-off Purchase Frequency")
plt.colorbar().set_label("Cluster")
show_tight_layout(show=show_plots, save=False, folder="plots")

fig = px.scatter_3d(clustered_data_pd, x='PURCHASES_FREQUENCY',
                    y='ONEOFF_PURCHASES_FREQUENCY',
                    z='PURCHASES_INSTALLMENTS_FREQUENCY',
                    color='prediction', symbol='BALANCE',
                    template='ggplot2', hover_name='CASH_ADVANCE_FREQUENCY')
if show_plots: fig.show()

