from pyspark.sql import SparkSession
from pyspark.sql.functions import count, when, isnull, col, greatest, lit
from pyspark.sql.functions import sum as spark_sum
from functools import reduce
from matplotlib import pyplot as plt
import pandas as pd
import os

output = 1 # 1 = show outputs, 0 = hide outputs
show_outputs = (output == 1)
graph = 1 # 1 = show plots, 0 = hide plots
show_plots = (graph == 1)

file = '/Users/wizrdm/Desktop/UEL/Machine Learning on Big Data/Airline_Delay_Cause.csv'

# Two ways to build spark session
# spark = SparkSession.builder.appName("Week 2 Portfolio") \
#                             .master("local[*]") \
#                             .config("spark.some.config.option", "some-value") \
#                             .getOrCreate()

spark = (
    SparkSession.builder.appName("Week 2 Portfolio")
    .master("local[*]")
    .config("spark.some.config.option", "some-value")
    .getOrCreate()
)

df = spark.read.format("csv").load(file, inferSchema=True,
                                   header=True)
def show_dataframe(df, num_rows=5):
    # print(f"Showing {str(df)}: ")
    df.show(num_rows)

if show_outputs:
    show_dataframe(df)

# year = year
# month = month
# carrier = type of plane ?
# carrier_name = name of airline
# airport = abbreviation of airport
# airport_name = full name of airport
# arr_flights = total num of arriving flights for given airline/airport/time period
# arr_del15 = num of arriving flights delayed 15+ minutes
# carrier_ct = num of delays bc of airline/carrier
# weather_ct = num of delays bc of weather
# nas_ct = num of delays bc of nas
# security_ct = num of delays bc of security
# late_aircraft_ct = num of delays bc aircraft is late
# arr_cancelled = num of canceled arrivals
# arr_diverted = num of diverted arrivals
# arr_delay = sum of delay minutes for all flights delayed 15+ minutes
# carrier_delay = sum of delay mins caused by carrier
# weather_delay = sum of delay mins caused by weather
# nas_delay = sum of delay mins caused by nas
# security_delay = sum of delay mins caused by security
# late_aircraft_delay = sum of delay mins caused by late aircraft

def show_dataframe_stats(df):
    print(f"Number of columns: {len(df.columns)}")
    print(f"Number of rows: {df.count()}")
    print(f"Number of partitions: {df.rdd.getNumPartitions()}") # show partitions
    print(f"Showing summary stats: ")
    df.summary().show()

if show_outputs: show_dataframe_stats(df)

# ----------------------------
# Null checks
# ----------------------------

def show_null_counts(df, num_rows=5):
    null_counts = df.select(
        [count(
            when(
                isnull(c), c
            )).alias(c) for c in df.columns])
    print(f"Showing number of null values per column: ")
    null_counts.show(num_rows)

if show_outputs: show_null_counts(df)

# Two ways to show rows with null values - same thing
# conditions = [col(c).isNull() for c in spark_df.columns]
# null_condition = conditions[0]
# for condition in conditions[1:]:
#     null_condition = null_condition | condition
# print("Showing null rows")
# spark_df.filter(null_condition).show(5)

def show_null_rows(df, num_rows=5):
    conditions = [col(c).isNull() for c in df.columns]
    if conditions:
        null_condition = reduce(lambda a, b: a | b, conditions)
        print(f"Showing null rows: ")
        df.filter(null_condition).show(num_rows)
    else:
        print("No columns found; cannot check for nulls.")

if show_outputs: show_null_rows(df)

# ----------------------------
# Data cleaning
# ----------------------------

# Drop rows with null arr_flights (which are unhelpful)
df = df.filter(df.arr_flights.isNotNull()) # drops 657 rows (out of 398,233)

# ----------------------------
# Observe relationship between delays and causes of delays
# ----------------------------

# Filter out rows with null count values in arr_del15, carrier_ct, weather_ct,
# nas_ct, security_ct, late_aircraft_ct
df_filtered1 = df.filter(df.arr_del15.isNotNull() & df.carrier_ct.isNotNull() &
                         df.weather_ct.isNotNull() & df.nas_ct.isNotNull() &
                         df.security_ct.isNotNull() & df.late_aircraft_ct.isNotNull())

# Make temporary column with sum of alleged causes of delays
df_filtered1 = df_filtered1.withColumn("sum_ct", col("carrier_ct") +
                                       col("weather_ct") + col("nas_ct") +
                                       col("security_ct") + col("late_aircraft_ct"))

# Make column with difference between arr_del15 and sum of delays
df_filtered1 = df_filtered1.withColumn("diff_del15_minus_sum",
                                       col("arr_del15") - col("sum_ct"))

if show_outputs:
    print(f"Showing relation between delays >= 15mins and sum of delays: ")
    df_filtered1.select("arr_del15", "sum_ct", "diff_del15_minus_sum").show(15)
    show_dataframe(df_filtered1.select("diff_del15_minus_sum"), 10)

# ----------------------------
# Null checks for imputable values in arr_del15
# ----------------------------

# Filter rows with null values in arr_del15
df_del15_null = df.filter(col("arr_del15").isNull())

# Keep rows where no flights arrived: arr_flights - arr_cancelled = 0
df_del15_null_no_arrivals = df_del15_null.filter(col("arr_flights") -
                                                 col("arr_cancelled") == 0)

if show_outputs:
    print(f"arr_del15 null rows: {df_del15_null.count()}")
    print(f"arr_del15 null rows with no arrivals: {df_del15_null_no_arrivals.count()}")
    show_dataframe(df_del15_null_no_arrivals)

# ----------------------------
# Fill imputable null values in arr_del15 (where no flights arrived)
# ----------------------------

df = df.withColumn("arr_del15", when(col("arr_del15").isNull() &
                    ((col("arr_flights") - col("arr_cancelled")) == 0),
                        0).otherwise(col("arr_del15")))

# ----------------------------
# Null checks
# ----------------------------

if show_outputs:
    print(f"num of arr_del15 null rows with no arrivals: "
          f"{df.filter(col("arr_del15").isNull()).count()}")

if show_outputs: show_dataframe_stats(df) # shows new df stats
if show_outputs: show_null_counts(df) # shows only 36 null remain in arr_del15
if show_outputs: show_null_rows(df, 36) # shows remaining 36 rows with nulls

# ----------------------------
# Fill remaining imputable null values in arr_del15 (consider redirected flights)
# ----------------------------

df = df.withColumn("arr_del15", when(col("arr_del15").isNull() &
                    ((col("arr_flights") - col("arr_cancelled") -
                      col("arr_diverted")) == 0), 0).otherwise(
                    col("arr_del15")))

if show_outputs:
    print(f"arr_del15 null rows after filling: {df.filter(
                                            col('arr_del15').isNull()).count()}")
if show_outputs: show_dataframe(df, 10)

# ----------------------------
# Q1 - Line chart of arr_del15 / arr_flights by year and month
# ----------------------------

def agg_delay_stats(df, cols):
    return df.groupBy(*cols).agg(spark_sum("arr_flights").alias("total_flights"),
                                spark_sum("arr_del15").alias("total_delays_15"))

df_y_m = agg_delay_stats(df, ["year", "month"]).withColumn("delay_rate",
                                    col("total_delays_15") / col("total_flights"))

df_y_m_ordered = df_y_m.orderBy("year", "month")
if show_outputs: show_dataframe(df_y_m_ordered, 10)

pdf = df_y_m_ordered.toPandas()

# Define date column
pdf["date"] = pd.to_datetime(pdf["year"].astype(str) + "-" +
                             pdf["month"].astype(str) + "-01")

def title_xy_labels(title, xlabel, ylabel):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

def show_tight_layout(show):
    if show:
        plt.tight_layout()

        plt.savefig(next_plot_path(), dpi=200, bbox_inches="tight")
        plt.show()
    else:
        plt.close()

# Save plots + avoid overwriting
def next_plot_path(base="plot", folder="plots", ext="png"):
    os.makedirs(folder, exist_ok=True)
    i = 1
    while True:
        name = f"{base}_{i:03d}.{ext}"   # plot_001.png, plot_002.png, ...
        path = os.path.join(folder, name)
        if not os.path.exists(path):
            return path
        i += 1

plt.figure(figsize=(12, 5))
plt.plot(pdf["date"], pdf["delay_rate"])
title_xy_labels("Overall Delay Rate (Arrivals Delayed 15+ mins / "
                "Total Arrivals)", "Date", "Delay Rate")
show_tight_layout(show_plots)

# ----------------------------
# Q2 - Bar chart of average delay rate per carrier_name
# ----------------------------

df_carrier = agg_delay_stats(df, ["carrier_name"]).withColumn("delay_rate",
                                    col("total_delays_15") / col("total_flights"))

pdf_carrier = df_carrier.toPandas()
top10carrier = pdf_carrier.sort_values("delay_rate", ascending=False).head(10)

plt.figure(figsize=(12, 6))
plt.bar(top10carrier["carrier_name"], top10carrier["delay_rate"])
title_xy_labels("Top 10 Carriers with Highest Delay Rate",
                "Airline", "Delay Rate")
plt.xticks(rotation=45, ha="right")
show_tight_layout(show_plots)

# ----------------------------
# Q3 - Airlines with most carrier-caused delays
# ----------------------------

df_carrier_ct = (df.groupBy("carrier_name").agg(spark_sum("carrier_ct").alias(
    "carrier_delay_count"), spark_sum("arr_flights").alias("total_flights"))
    .withColumn("carrier_delay_rate",
                col("carrier_delay_count") / col("total_flights"))
    .orderBy(col("carrier_delay_rate").desc()))

pdf_carrier_ct = df_carrier_ct.toPandas()
top10carrier_ct = pdf_carrier_ct.head(10)

plt.figure(figsize=(12, 6))
plt.barh(top10carrier_ct["carrier_name"], top10carrier_ct["carrier_delay_rate"])
plt.gca().invert_yaxis()
title_xy_labels("Top 10 Airlines by Carrier-Caused Delay Rate",
                "Carrier-Caused Delay Rate", "Airline")
show_tight_layout(show_plots)

# ----------------------------
# Q4 - Scatter or bar chart of delay rate per airport, sorted by highest rates
# ----------------------------

df_airport = (agg_delay_stats(df, ["airport_name"])
              .withColumn("delay_rate", col("total_delays_15") / col("total_flights")))
              # .orderBy(col("delay_rate").desc()))

pdf_airport = df_airport.toPandas()
pdf_airport = pdf_airport.sort_values("delay_rate", ascending=False).head(10)

plt.figure(figsize=(12, 6))
plt.barh(pdf_airport["airport_name"], pdf_airport["delay_rate"])
plt.gca().invert_yaxis()
title_xy_labels("Top 10 Airports with Highest Delay Rate",
                "Delay Rate", "Airport")
show_tight_layout(show_plots)

# ----------------------------
# Q5 - Stacked bar chart: dominant causes of delays across all flights per year
# ----------------------------

df_causes = (df.groupBy("year").agg(spark_sum("carrier_ct").alias("carrier"),
                            spark_sum("weather_ct").alias("weather"),
                            spark_sum("nas_ct").alias("nas"),
                            spark_sum("security_ct").alias("security"),
                            spark_sum("late_aircraft_ct").alias("late_aircraft"))
     .withColumn("total_causes", col("carrier")+col("weather")+
                col("nas")+col("security")+col("late_aircraft"))
     .withColumn("carrier_pct", col("carrier")/col("total_causes"))
     .withColumn("weather_pct", col("weather")/col("total_causes"))
     .withColumn("nas_pct", col("nas")/col("total_causes"))
     .withColumn("security_pct", col("security")/col("total_causes"))
     .withColumn("late_aircraft_pct", col("late_aircraft")/col("total_causes"))
     .orderBy("year"))

pdf_causes = (df_causes.select("year", "carrier_pct", "weather_pct", "nas_pct",
                              "security_pct", "late_aircraft_pct")
              .toPandas().sort_values("year"))

years = pdf_causes["year"]
bottom=0

plt.figure(figsize=(12, 6))
plt.bar(years, pdf_causes["carrier_pct"], label="Carrier Caused", bottom=bottom)
bottom += pdf_causes["carrier_pct"]
plt.bar(years, pdf_causes["weather_pct"], label="Weather Caused", bottom=bottom)
bottom += pdf_causes["weather_pct"]
plt.bar(years, pdf_causes["nas_pct"], label="NAS Caused", bottom=bottom)
bottom += pdf_causes["nas_pct"]
plt.bar(years, pdf_causes["security_pct"], label="Security Caused", bottom=bottom)
bottom += pdf_causes["security_pct"]
plt.bar(years, pdf_causes["late_aircraft_pct"], label="Late Aircraft Caused", bottom=bottom)
title_xy_labels("Causes of Delays Across All Flights by Year",
                "Year", "Proportion of Delay Causes")
plt.ylim(0, 1)
plt.legend()
show_tight_layout(show_plots)

# Extra: record highest cause of delays per year
cause_cols_pct = ["carrier_pct", "weather_pct", "nas_pct", "security_pct",
                  "late_aircraft_pct"]
top_cause_col = pdf_causes[cause_cols_pct].idxmax(axis=1)
top_pct = pdf_causes[cause_cols_pct].max(axis=1)

# Rename causes
name_map = {"carrier_pct": "Carrier", "weather_pct": "Weather",
            "nas_pct": "NAS", "security_pct": "Security",
            "late_aircraft_pct": "Late Aircraft"}

pdf_top_causes = pdf_causes[["year"]].copy()
pdf_top_causes["top_cause"] = top_cause_col.map(name_map)
pdf_top_causes["top_pct"] = top_pct

plt.figure(figsize=(12, 6))
plt.plot(pdf_top_causes["year"], pdf_top_causes["top_pct"], marker="o")
prev = None
for x, y, label in zip(pdf_top_causes["year"], pdf_top_causes["top_pct"],
                       pdf_top_causes["top_cause"]):
    if label != prev:
        plt.annotate(label, xy=(x, y), xytext=(6, 8), textcoords="offset points",
                     fontsize=9)
    prev = label
title_xy_labels("Highest Cause of Delays per Year",
                "Year", "Top Cause Proportion")
plt.ylim(0, 1)
show_tight_layout(show_plots)

# ----------------------------
# Q6 - How causes vary by season
# ----------------------------

df_m_causes = (df.groupBy("month").agg(
                            spark_sum("carrier_ct").alias("carrier"),
                            spark_sum("weather_ct").alias("weather"),
                            spark_sum("nas_ct").alias("nas"),
                            spark_sum("security_ct").alias("security"),
                            spark_sum("late_aircraft_ct").alias("late_aircraft"))
               .orderBy("month"))

pdf_monthly = df_m_causes.toPandas()

plt.figure(figsize=(12, 6))
plt.plot(pdf_monthly["month"], pdf_monthly["carrier"],
         marker="o", label="Carrier")
plt.plot(pdf_monthly["month"], pdf_monthly["weather"],
         marker="o", label="Weather")
plt.plot(pdf_monthly["month"], pdf_monthly["nas"],
         marker="o", label="NAS")
plt.plot(pdf_monthly["month"], pdf_monthly["security"],
         marker="o", label="Security")
plt.plot(pdf_monthly["month"], pdf_monthly["late_aircraft"],
         marker="o", label="Late Aircraft")

title_xy_labels("Monthly Causes of Delay (Raw)",
                "Month", "Count of Delays by Cause")
plt.xticks(range(1, 13))
plt.legend()
show_tight_layout(show_plots)

# ----------------------------
# Q7 - Months with highest and lowest delay rate
# ----------------------------

df_m_rate = (df.withColumn("effective_arrivals", greatest(col("arr_flights")-
                                                          col("arr_cancelled")-
                                                          col("arr_diverted"), lit(0)))
             .groupBy("month")
             .agg(spark_sum("arr_del15").alias("total_delays_15"), spark_sum("effective_arrivals")
                  .alias("total_effective_arrivals"))
             .withColumn("delay_rate", when(col("total_effective_arrivals")>0,
                                            col("total_delays_15")/col("total_effective_arrivals"))
             .otherwise(lit(None)))).orderBy("month")

pdf_m_rate = df_m_rate.toPandas()

valid = pdf_m_rate.dropna(subset=["delay_rate"])
max_row = valid.loc[valid["delay_rate"].idxmax()]
min_row = valid.loc[valid["delay_rate"].idxmin()]

plt.figure(figsize=(12, 6))
plt.plot(pdf_m_rate["month"], pdf_m_rate["delay_rate"], marker="o")
plt.xticks(range(1, 13))
title_xy_labels("Overall Delay Rate by Month",
                "Month", "Delay Rate")
plt.scatter([max_row["month"]], [max_row["delay_rate"]], zorder=3)
plt.annotate(f"Max: M{int(max_row['month'])}",
             (max_row["month"], max_row["delay_rate"]),
             textcoords="offset points", xytext=(8, 8))
plt.scatter([min_row["month"]], [min_row["delay_rate"]], zorder=3)
plt.annotate(f"Min: M{int(min_row['month'])}",
             (min_row["month"], min_row["delay_rate"]),
             textcoords="offset points", xytext=(8, -14))
show_tight_layout(show_plots)

spark.stop()

