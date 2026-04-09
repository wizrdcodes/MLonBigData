from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, mean, when, isnull
import matplotlib.pyplot as plt
import pandas as pd

# connected to google drive

# ----------------------------
# Creating SparkSession
# ----------------------------

spark = SparkSession.builder \
    .appName("Tutorial_DF") \
    .master("local[*]") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "2g") \
    .config("spark.executor.cores", "2") \
    .config("spark.sql.inMemoryColumnarStorage.compressed", "true") \
    .getOrCreate()

# ----------------------------
# How to Load a CSV File in Spark DataFrame
# ----------------------------

# Load CSV into DataFrame
file_path = "/Users/wizrdm/Desktop/UEL/Machine Learning on Big Data/PlayersDataset.csv"
df = spark.read.csv(file_path, header=True, inferSchema=True)

number = 1
if number == 1:
    print_data = False
else:
    print_data = True
if print_data:
    print(f"first 5 rows: {df.show(5)}") # show first 5 rows
    print(f"Schema of DataFrame: {df.printSchema(1)}")

# ----------------------------
# Setting Partitions
# ----------------------------

# find number of partitions
if print_data:
    print(f"Number of partitions: {df.rdd.getNumPartitions()}") # prints 2

# repartition
df = df.repartition(10)
if print_data:
    print(f"Number of partitions: {df.rdd.getNumPartitions()}") # prints 10

# find total number of columns and rows
if print_data:
    print(f"Total number of columns: {len(df.columns)}")
    print(f"Total number of rows: {df.count()}")

# df.describe().show() # describes statistics of dataset
if print_data:
    df.summary().show()

# check for null values in each column
if print_data:
    print("Checking for columns with null values:")
    null_counts = df.select([(count(when(isnull(c), c)).alias(c))\
                             for c in df.columns])
    null_counts.show() # shows num of null per column

# Fill missing values with appropriate default values
# For numeric columns, fill with 0; for string columns, fill with 'Unknown'
if print_data:
    print("Filling missing values:")
fill_values = {col_name: 0 if dtype.startswith("int") \
               or dtype.startswith("double") \
               else "Unknown" \
               for col_name, dtype in df.dtypes}
df_filled = df.fillna(fill_values)
if print_data:
    df_filled.show(5)

# Rename columns (example renaming)
new_column_names = {"Name": "Player_Name", "Age": "Player_Age",
                    "Nationality": "Country"}
for old_name, new_name in new_column_names.items():
    df_filled = df_filled.withColumnRenamed(old_name, new_name)

# Show updated column names
if print_data:
    print(f"Updated column names: {df_filled.show(5)}")

# Select specific columns
if print_data:
    df_filled.select("Player_Name", "Player_Age").show(5)
    df_filled.select("Player_Name").show(5)

# Show distinct nationalities (no copies)
if print_data:
    df_filled.select("Country").distinct().show(5)

# Filtering rows where "Overall" > 90
if print_data:
    print("Filtering players with Overall > 90:")
    df_filled.filter(col("Overall") > 90).select("Player_Name", "Overall").show(5)

# Filtering rows where Player_Age < 25
if print_data:
    print("Filtering players younger than 25:")
    df_filled.filter(col("Player_Age")<25).select("Player_Name", "Player_Age").show(5)

# Group by country and count players
if print_data:
    print("Grouping by Country and counting players:")
    df_filled.groupBy("Country").count().show(5)

# Group by club and calculate average player overall rating
if print_data:
    print("Group by Club and calculate average Overall:")
    df_filled.groupBy("Club").agg(mean("Overall").alias("Average Overall")).show(5)

# Average player rating by country
average_rating_by_country = df_filled.groupBy("Country").agg(mean("Overall").alias("Average Rating"))\
    .orderBy(col("Average Rating").desc())

# ----------------------------
# Data Visualisation
# ----------------------------

pandas_df = df_filled.select("Player_Age", "Overall", "Country").toPandas()

second_number = 0
if second_number == 1:
    show_plots = False
else:
    show_plots = True

# Plot Age Distribution
plt.figure(figsize=(10,6))
pandas_df["Player_Age"].hist(bins=30, color="skyblue")
plt.title("Age Distribution of Players")
plt.xlabel("Age")
plt.ylabel("Frequency")
if show_plots:
    plt.show()

# Plot Overall Rating Distribution
plt.figure(figsize=(10, 6))
pandas_df["Overall"].hist(bins=30, color="orange")
plt.title("Player Overall Rating Distribution")
plt.xlabel("Overall Rating")
plt.ylabel("Frequency")
if show_plots:
    plt.show()

# Bar Plot of Top 10 Countries by Player Count
country_counts = pandas_df["Country"].value_counts().head(10)
plt.figure(figsize=(12, 6))
country_counts.plot(kind="bar", color="green")
plt.title("Top 10 Countries by Player Count")
plt.xlabel("Country")
plt.ylabel("Player Count")
plt.xticks(rotation=45)
if show_plots:
    plt.show()

# Bar Plot of Top 10 Average Player Rating by Country
average_rating_pd_df = average_rating_by_country.toPandas()
average_rating_pd_df = average_rating_pd_df.head(20)
plt.figure(figsize=(12, 6))
plt.bar(average_rating_pd_df["Country"], average_rating_pd_df["Average Rating"], color="purple")
plt.title("Average Player Rating by Country")
plt.xlabel("Country")
plt.ylabel("Average Rating")
plt.xticks(rotation=45)
if show_plots:
    plt.show()