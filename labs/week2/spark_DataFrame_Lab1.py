import pyspark
from pyspark.sql import SparkSession
import numpy as np
import pandas as pd

# ----------------------------
# Tutorial 1 - Working with PySpark DataFrame
# ----------------------------

# Initialize PySpark
# A SparkSession represents a Spark application that can manage various
# contexts (such as SQL, streaming, or a SparkR session).
# It can be used to create a DataFrame and register it as tables, execute SQL
# over tables, and read "parquet" files, etc.

# Use the following builder pattern to create a SparkSession:
spark = SparkSession.builder.appName("Tutorial1_CN7030") \
                            .master("local[*]") \
                            .config("spark.some.config.option", "some-value") \
                            .getOrCreate()

# ----------------------------
# Part 2 - Create PySpark DataFrame (with an explicit schema)
# ----------------------------

# PySpark DataFrame with Explicit Schema
df = spark.createDataFrame([
    (1, 87.0, 'Mike', 'math', 'passed'),
    (2, 60.5, 'Mike', 'computing', 'failed'),
    (3, 20.8, 'Mina', 'networking', 'passed'),
    (4, 41.0, 'Emmy', 'math', 'failed'),
    (5, 39.0, 'Alex', 'computing', 'failed'),
    (6, 55.8, 'Alex', 'AI', 'passed'),
    (7, 74.0, 'Emmy', 'AI', 'passed'),
    ], schema = 'ID int, Mark double, Name string, Lesson string, Status string')

# df.printSchema() # show schema

data = [[295, "South Bend", "Indiana",  101190, 112.9]]
columns = ["rank", "city", "state",  "population", "price"]

df1 = spark.createDataFrame(data, schema="rank LONG, city STRING, state STRING, "
                                         "population LONG, price DOUBLE")

# df1.printSchema() # show schema

# ----------------------------
# Part 3 - Create PySpark DataFrame from Pandas DataFrame
# ----------------------------

df_pandas = pd.DataFrame(np.random.randint(0,200,size=(100000,5)),
                         columns=list('ABCDE'))

df2 = spark.createDataFrame(df_pandas)

# df2.printSchema() # show schema
# print(df2.count()) # shows 100,000

dfs={"df":df, "df1":df1, "df2":df2}
show_dataframes = True
if show_dataframes:
    for name, dataframe in dfs.items():
        # n = dataframe.count()
        print(f"{name}: {dataframe.count()} rows")
        dataframe.show()

# print(f"df: {df.count()} rows")
# df.show() # show table

# print(f"df1: {df1.count()} rows")
# df1.show() # show table

# print(f"df2: {df2.count()} rows")
# df2.show() # show table

def failed_union():
    df3 = df1.union(df)

    print(f"df3: {df3.count()} rows")
    df3.show() # show table
    df3.tail(7)

total_partitions=df.rdd.getNumPartitions()
print(f"total partitions: {total_partitions}")
newpartitiondf = df.repartition(4)

print(f"new partitions: {newpartitiondf.rdd.getNumPartitions()}")

# Repartition by column name into 4 partitions,
# df1 column "city" we want to be in four partitions
df4 = df1.repartition(4, "city")
print(f"get num partitions: {df4.rdd.getNumPartitions()}")
# Repartition by multiple columns
# df5 = df4.repartition("ColumnName1","ColumnName2")

# ----------------------------
# Part 4 - PySpark DataFrame from CSV
# ----------------------------

myfile="/Users/wizrdm/Desktop/UEL/Machine Learning on Big Data/PlayersDataset.csv"
df_fifa = spark.read.format("csv").load(myfile, inferSchema=True,
                                        header=True)

df_fifa.show(truncate=True)
df_fifa.printSchema()

print(f"df_fifa count: {df_fifa.count()}")
print(f"number of columns: {len(df_fifa.columns)}")
print(f"number of rows: {df_fifa.rdd.count()}")

df_fifa = df_fifa.repartition(4) # Change the number of partitions
df_fifa.rdd.getNumPartitions()

subset_Age = df_fifa.filter(df_fifa["Age"] > 30) # showing Age more than 20
subset_Age.show(10)

df_fifa.select("Name","Club").distinct().show(10)

