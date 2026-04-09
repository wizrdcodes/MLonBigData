from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from src.mlonbigdata import show_df, title_xy_labels

spark = SparkSession.builder\
    .master("local")\
    .appName("ClusteringApp")\
    .enableHiveSupport()\
    .config("spark.ui.port", "4050")\
    .getOrCreate()

df = spark.read.csv('/Users/wizrdm/Desktop/UEL/Machine '
            'Learning on Big Data/Grocery.csv', inferSchema=True, header=True)

show_df(df, rows=5)
df.count()

from pyspark.sql.functions import col, isnan, when, count

null_counts = [
    count(when(col(c).isNull(), c)).alias(c)
    for c in df.columns]
df.select(null_counts).show()

data_df = df.toPandas()

# initialize figure with 4 subplots in a row
fig, ax = plt.subplots(1, 4, figsize=(15, 6))

plt.subplots_adjust(wspace=0.5) # padding between subplots

# draw boxplot for age in the 1st subplot
sns.boxplot(data=data_df['Fresh'], ax=ax[0], color='brown',)
ax[0].set_xlabel('Fresh')

# draw boxplot for station_distance in the 2nd subplot
sns.boxplot(data=data_df['Milk'], ax=ax[1], color='g')
ax[1].set_xlabel('Milk')

# draw boxplot for stores_count in the 3rd subplot
sns.boxplot(data=data_df['Grocery'], ax=ax[2], color='y')
ax[2].set_xlabel('Grocery')

# finally draw boxplot for unit_price in the 4th subplot
sns.boxplot(data=data_df['Frozen'], ax=ax[3])
ax[3].set_xlabel('Frozen')

# by default, you'll see x-tick label set to 0 in each subplot
# remove it by setting it to empty list
for subplot in ax:
    subplot.set_xticklabels([])

plt.show()

numerical_features = ['Fresh', 'Milk', 'Grocery',
       'Frozen', 'Detergents_Paper', 'Delicassen']

plt.figure(figsize=(15, 8))
sns.heatmap(round(data_df[numerical_features].corr(method='spearman'), 2),
            annot=True, mask=None, cmap='GnBu')
plt.show()

ClusteringColumns = ['Fresh', 'Milk', 'Grocery']

assembler=VectorAssembler(inputCols=ClusteringColumns, outputCol="features")
featureDf = assembler.transform(df)
print("Schema of featureDf:")
featureDf.printSchema()
show_df(featureDf, rows=5, print_str="Showing data sample after assembler:")

#preparing data for clustering
featureDf.count()
trainingData, testData = featureDf.randomSplit([0.7, 0.3], seed = 5043)
# print(type(trainingData))
print(f"Training Dataset Count: {trainingData.count()}")
print(f"Test Dataset Count: {testData.count()}")
trainingData.show(10)
print(testData.columns)

cluster_count = [3, 5, 10, 20, 30, 40]
wssse_values = []

# save best model
best_k = None # cluster count
best_sil = -1 # silhouette
best_predictDf = None # predictDf
best_model = None # kmeansModel

for i in cluster_count:
    kmeans = (KMeans().setK(i).setSeed(1).setFeaturesCol("features")
              .setPredictionCol("prediction"))
    kmeansModel = kmeans.fit(trainingData)

    # test the model with test data set
    predictDf = kmeansModel.transform(testData)

    evaluator = ClusteringEvaluator()
    silhouette = evaluator.evaluate(predictDf)

    wssse_values.append(silhouette)
    print(f"clusters = {i}, Silhouette with squared euclidean distance: "
          f"{silhouette:.5f}")
    # for clusters in kmeansModel.clusterCenters():
    #    print("cluster centres",clusters)
    if silhouette > best_sil:
        best_sil = silhouette
        best_k = i
        best_predictDf = predictDf
        best_model = kmeansModel
        print(f"New best k = {best_k} with silhouette = {best_sil}")

# Plotting WSSSE values
plt.plot(range(1, 7), wssse_values)
title_xy_labels('Silhouette for Optimal K','Number of Clusters (K)',
                'Within Set Sum of Squared Errors (WSSSE)')
plt.grid()
plt.show()

# test the model with test data set
best_predictDf.show(10)
evaluator = ClusteringEvaluator()
print(f"Silhouette with squared euclidean distance = {best_sil}")

best_predictDf.groupBy("prediction").count().show()

# Converting to Pandas DataFrame
clustered_data_pd = best_predictDf.toPandas()
# Visualizing the results
plt.scatter(clustered_data_pd["Fresh"], clustered_data_pd["Milk"],
            c=clustered_data_pd["prediction"], cmap='viridis')
title_xy_labels('K-means Clustering with PySpark MLlib',
                'Fresh','Milk')
plt.colorbar().set_label("Cluster")
plt.show()

fig = px.scatter_3d(clustered_data_pd, x='Fresh', y='Milk', z='Grocery',
                    color='prediction')
fig.show()