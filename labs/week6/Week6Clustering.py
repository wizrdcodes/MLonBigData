from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans, KMeansModel
from pyspark.ml.evaluation import ClusteringEvaluator
import matplotlib.pyplot as plt

from src.mlonbigdata import show_df

spark = SparkSession.builder\
    .master("local")\
    .appName("ClusteringApp")\
    .enableHiveSupport()\
    .config("spark.ui.port", "4050")\
    .getOrCreate()

df = spark.read.csv('/Users/wizrdm/Desktop/UEL/Machine '
                'Learning on Big Data/uber.csv',inferSchema=True, header =True)

show_df(df, rows=5, print_str="Showing data sample:")
print(f"row count of data: {df.count()}")

cols = ["Lat", "Lon"]

assembler = VectorAssembler(inputCols=cols, outputCol="features")
featureDf = assembler.transform(df)
print("Schema of featureDf:")
featureDf.printSchema()
show_df(featureDf, rows=5, print_str="Showing data sample after assembler:")

# splitting data (no preprocessing or checks)
trainingData, testData = featureDf.randomSplit([0.7, 0.3], seed = 5043)
print(f"Training Dataset Count: {trainingData.count()}")
print(f"Test Dataset Count: {testData.count()}")

cluster_count = [10, 30, 50, 70, 90]

# retain best model
best_k = None
best_sil = -1
best_predictDf = None
best_model = None

for i in cluster_count:
    kmeans = KMeans().setK(i).setSeed(1).setFeaturesCol("features")
    kmeansModel = kmeans.fit(trainingData)
    print(f"Cluster Centers ({i}):")
    for clusters in kmeansModel.clusterCenters():
        print(clusters)
    # test the model with test data set
    predictDf = kmeansModel.transform(testData)
    show_df(predictDf, rows=5, print_str=f"Showing predictions for {i} clusters:")
    evaluator = ClusteringEvaluator()
    silhouette = evaluator.evaluate(predictDf)
    print(f"Silhouette with squared euclidean distance: {silhouette}")

    if silhouette > best_sil:
        best_sil = silhouette
        best_k = i
        best_predictDf = predictDf
        best_model = kmeansModel
        print(f"New best k = {best_k} with silhouette = {best_sil}")

print(f"Showing count of each cluster:")
best_predictDf.groupBy("prediction").count().show()

pddf_pred = best_predictDf.toPandas()
print(pddf_pred.head())
fig = plt.figure(figsize=(16,12))
KmVis= fig.add_subplot(111)
KmVis.scatter(pddf_pred.Lat, pddf_pred.Lon, c=pddf_pred.prediction)
KmVis.set_xlabel('x')
KmVis.set_ylabel('y')

plt.show()

#save model
best_model.write().overwrite().save("uber-model")
kmeansModelLoaded = KMeansModel.load("uber-model")

df1 = spark.sparkContext.parallelize([
    ("5/1/2014 0:02:00", 40.7521, -73.9914, "B02512"),
    ("5/1/2014 0:06:00", 40.6965, -73.9715, "B02512"),
    ("5/1/2014 0:15:00", 40.7464, -73.9838, "B02512"),
    ("5/1/2014 0:17:00", 40.7463, -74.0011, "B02512"),
    ("5/1/2014 0:17:00", 40.7594, -73.9734, "B02512")]
).toDF(["time", "Lat", "Lon", "base"])
df1.show()

df2 = assembler.transform(df1)
df2.show()

# prediction of sample data set with loaded model
df3 = kmeansModelLoaded.transform(df2)
df3.show()

