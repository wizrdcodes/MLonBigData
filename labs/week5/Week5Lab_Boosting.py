from pyspark.sql import SparkSession
from pyspark.ml.classification import GBTClassifier, OneVsRest
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


# Start Spark session
spark = SparkSession.builder.appName("GBTClassifier-Multiclass").getOrCreate()

# Load dataset (replace with your path)
iris = spark.read.csv('/Users/wizrdm/Desktop/UEL/Machine '
            'Learning on Big Data/iris.csv', header=True, inferSchema=True)

# Prepare features
feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
iris = assembler.transform(iris)

# Index labels
indexer = StringIndexer(inputCol="species", outputCol="label")
iris = indexer.fit(iris).transform(iris)

# Train-test split
train, test = iris.randomSplit([0.7, 0.3], seed=42)

# Initialize GBTClassifier
gbt = GBTClassifier(maxIter=20, maxDepth=3, stepSize=0.1)

# Wrap in OneVsRest for multiclass
ovr = OneVsRest(classifier=gbt)

# Train
model = ovr.fit(train)

# Predict
predictions = model.transform(test)

# Evaluate
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"Test Accuracy: {accuracy:.4f}")

spark.stop()

# Test Accuracy: 0.9583 (0.8, 0.2)
# Test Accuracy: 0.9783 (0.7, 0.3)