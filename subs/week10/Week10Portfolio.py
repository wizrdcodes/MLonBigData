#%%

# ---------------------------------------------------------------------------------
# In this assessment lab, you will independently implement a distributed
# time-series forecasting system using PySpark to predict cryptocurrency prices.
# Building on the Guided Lab, you must adapt the same forecasting pipeline to the
# Bitcoin Historical Data dataset, which contains real-world financial time-series
# data. The task requires you to load the dataset into a Spark DataFrame,
# preprocess and organise the data chronologically, generate temporal features
# using window functions, and transform the time-series into a supervised learning
# format for forecasting. You must implement the complete pipeline, including
# feature engineering using lag and rolling statistical features, creation of a
# target variable representing future price values, and a time-based train–test
# split to avoid data leakage. Distributed regression models such as Linear
# Regression and Gradient Boosted Trees should be trained and used to generate
# predictions. The models must be evaluated using appropriate regression metrics
# such as RMSE and MAE, and the results should be analysed to assess forecasting
# performance. Your submission should clearly demonstrate the full workflow by
# providing evidence of data loading, preprocessing, feature generation, model
# training, prediction outputs, and evaluation results. The objective of this
# assessment is to evaluate your ability to apply distributed machine learning
# techniques in PySpark and adapt a forecasting pipeline to a new financial
# dataset in a real-world fintech scenario.
#
# Dataset Description
# The Bitcoin Historical Data dataset contains one-minute cryptocurrency price
# records with Timestamp, Open, High, Low, Close, and Volume columns. The Close
# price is used as the main target variable for forecasting future values. This
# is a time-series forecasting problem where historical cryptocurrency prices are
# used to predict future price movements. The dataset is chronologically ordered
# and requires time-aware processing to ensure correct modelling and avoid data
# leakage.
# ---------------------------------------------------------------------------------

# Core Python libraries
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Helps avoid Spark's common local hostname warning
os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")

class WarnLineFilter:
    """File-like stream wrapper that skips console lines containing WARN.
    Spark often writes useful results and noisy WARN lines to the console. This
    wrapper keeps the output easier to paste into an HTML/report by filtering
    lines that contain WARN while still allowing ERROR messages through.
    """

    def __init__(self, stream, skip_terms=("WARN",)):
        self.stream = stream
        self.skip_terms = skip_terms
        self._buffer = ""

    def write(self, text):
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if not any(term in line for term in self.skip_terms):
                self.stream.write(line + "\n")

    def flush(self):
        if self._buffer:
            if not any(term in self._buffer for term in self.skip_terms):
                self.stream.write(self._buffer)
            self._buffer = ""
        self.stream.flush()

    def __getattr__(self, attr):
        return getattr(self.stream, attr)

FILTER_WARN_MESSAGES = True
if FILTER_WARN_MESSAGES:
    sys.stdout = WarnLineFilter(sys.stdout)
    sys.stderr = WarnLineFilter(sys.stderr)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import (
    avg,
    col,
    dayofweek,
    from_unixtime,
    lag,
    lead,
    max as spark_max,
    max_by,
    min as spark_min,
    min_by,
    month,
    row_number,
    signum,
    stddev,
    sum as spark_sum,
    to_date,
    when,
)
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.ml.regression import GBTRegressor, LinearRegression

# Plotting aesthetics
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.figsize"] = (14, 6)

print("✅ Setup complete. Libraries imported.")

#%%
default_path = "/Users/wizrdm/Desktop/UEL/Machine Learning on Big Data/btcusd_1-min_data.csv"

if os.path.exists(default_path):
    data_file = default_path
    print("✅ Bitcoin CSV file found locally.")
else:
    raise FileNotFoundError(
        "Bitcoin CSV was not found at default_path. "
        "Check that btcusd_1-min_data.csv is stored at the path above."
    )

try:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
except NameError:
    PROJECT_ROOT = Path.cwd()

PLOTS_DIR = PROJECT_ROOT / "outputs" / "week10" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

REQUIRED_COLUMNS = ["Timestamp", "Open", "High", "Low", "Close", "Volume"]
FORECAST_HORIZON = 1  # Predict the next daily close price.


def save_current_plot(filename: str) -> None:
    """Save the active matplotlib figure to the Week 10 plots folder."""
    plot_path = PLOTS_DIR / filename
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot: {plot_path}")
    plt.show()


print(f"📁 Plots will be saved to: {PLOTS_DIR}")

#%%
# Load a daily Pandas summary for readable EDA plots.
# The raw dataset is one-minute data with millions of rows, so plotting every row
# directly would be slow and visually cluttered. Chunked daily aggregation keeps
# the EDA memory-safe while still representing the full time range.
def load_daily_btc_summary(csv_path: str, chunk_size: int = 1_000_000) -> pd.DataFrame:
    daily_frames = []

    for chunk in pd.read_csv(csv_path, usecols=REQUIRED_COLUMNS, chunksize=chunk_size):
        chunk["DateTime"] = pd.to_datetime(chunk["Timestamp"], unit="s")
        chunk["Date"] = chunk["DateTime"].dt.date

        daily_chunk = (
            chunk.groupby("Date")
            .agg(
                Open=("Open", "first"),
                High=("High", "max"),
                Low=("Low", "min"),
                Close=("Close", "last"),
                Volume=("Volume", "sum"),
            )
            .reset_index()
        )
        daily_frames.append(daily_chunk)

    daily_pdf = pd.concat(daily_frames, ignore_index=True).sort_values("Date")

    # Regroup because a single calendar day may be split across two chunks.
    daily_pdf = (
        daily_pdf.groupby("Date")
        .agg(
            Open=("Open", "first"),
            High=("High", "max"),
            Low=("Low", "min"),
            Close=("Close", "last"),
            Volume=("Volume", "sum"),
        )
        .reset_index()
    )

    daily_pdf["Date"] = pd.to_datetime(daily_pdf["Date"])
    daily_pdf["Daily_Return"] = daily_pdf["Close"].pct_change() * 100
    daily_pdf["Price_Range"] = daily_pdf["High"] - daily_pdf["Low"]
    daily_pdf["Price_Range_Pct"] = np.where(
        daily_pdf["Open"] != 0,
        (daily_pdf["High"] - daily_pdf["Low"]) / daily_pdf["Open"],
        0,
    )
    return daily_pdf


pdf = load_daily_btc_summary(data_file)

print(
    f"✅ Daily BTC summary loaded: {pdf.shape[0]:,} days. "
    f"Range: {pdf['Date'].min().date()} to {pdf['Date'].max().date()}"
)
print(pdf.head())

#%%
# 2.1 Dual-Axis Plot: Bitcoin Daily Close Price & Trading Volume
fig, ax1 = plt.subplots(figsize=(16, 7))

color = "tab:blue"
ax1.set_xlabel("Date", fontweight="bold")
ax1.set_ylabel("BTC Close Price (USD)", color=color, fontweight="bold")
ax1.plot(pdf["Date"], pdf["Close"], color=color, linewidth=1.5)
ax1.tick_params(axis="y", labelcolor=color)

ax2 = ax1.twinx()
color = "tab:grey"
ax2.set_ylabel("Daily Trading Volume", color=color, fontweight="bold")
ax2.fill_between(pdf["Date"], pdf["Volume"], color=color, alpha=0.3)
ax2.tick_params(axis="y", labelcolor=color)

plt.title("Bitcoin Price History vs. Trading Volume", fontsize=16, fontweight="bold")
save_current_plot("btc_price_volume.png")

#%%
# 2.2 Distribution of Daily Returns
plt.figure(figsize=(14, 6))
sns.histplot(pdf["Daily_Return"].replace([np.inf, -np.inf], np.nan).dropna(), bins=100, kde=True)
plt.title("Distribution of Bitcoin Daily Returns (%)", fontsize=16, fontweight="bold")
plt.xlabel("Daily Return (%)")
plt.ylabel("Frequency")
plt.axvline(0, color="black", linestyle="--")

std_dev = pdf["Daily_Return"].std()
plt.axvline(std_dev * 3, color="red", linestyle=":", label="3 Std Dev (Extreme Gain)")
plt.axvline(-std_dev * 3, color="red", linestyle=":", label="3 Std Dev (Extreme Loss)")
plt.legend()
save_current_plot("btc_returns_distribution.png")

#%%
# 2.3 Feature Correlation Heatmap
plt.figure(figsize=(10, 8))
corr_cols = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Daily_Return",
    "Price_Range",
    "Price_Range_Pct",
]
corr_matrix = pdf[corr_cols].corr()
sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="vlag",
    center=0,
    fmt=".2f",
    linewidths=0.5,
    cbar_kws={"shrink": 0.8},
)
plt.title("Bitcoin Feature Correlation Heatmap", fontsize=16, fontweight="bold")
save_current_plot("btc_correlation.png")

#%%
# Spark Data Loading and Preprocessing
spark = (
    SparkSession.builder.appName("Week10_Bitcoin_Forecasting")
    .config("spark.ui.showConsoleProgress", "false")
    .config("spark.sql.shuffle.partitions", "8")
    .config("spark.default.parallelism", "8")
    .config("spark.sql.adaptive.enabled", "true")
    .config("spark.sql.debug.maxToStringFields", "200")
    .config("spark.driver.memory", "8g")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")
spark_df = spark.read.csv(data_file, header=True, inferSchema=True)

print("✅ Raw Spark schema:")
spark_df.printSchema()

parsed_df = spark_df.select(
    from_unixtime(col("Timestamp").cast("long")).cast("timestamp").alias("DateTime"),
    col("Open").cast("double"),
    col("High").cast("double"),
    col("Low").cast("double"),
    col("Close").cast("double"),
    col("Volume").cast("double"),
).dropna(subset=["DateTime", "Open", "High", "Low", "Close", "Volume"])

# The raw CSV contains one-minute data. Running global lag/rolling windows across
# all 7.5+ million minute rows can force Spark into a single huge window partition
# and cause an out-of-memory error. The fix is to aggregate to daily OHLCV rows
# before applying the forecasting windows. This still uses the full CSV, but the
# ML stage runs over a manageable daily time series.
df = (
    parsed_df.withColumn("Date", to_date(col("DateTime")))
    .groupBy("Date")
    .agg(
        min_by(col("Open"), col("DateTime")).alias("Open"),
        spark_max("High").alias("High"),
        spark_min("Low").alias("Low"),
        max_by(col("Close"), col("DateTime")).alias("Close"),
        spark_sum("Volume").alias("Volume"),
    )
    .withColumn("DateTime", col("Date").cast("timestamp"))
    .orderBy("DateTime")
    .cache()
)

print("✅ Cleaned daily Spark DataFrame preview:")
df.show(5, truncate=False)
print(f"✅ Daily Spark rows used for modelling: {df.count():,}")

#%%
# Distributed Window Functions for Lag, Rolling Statistics, and Target Creation
time_window = Window.orderBy("DateTime")
rolling_5 = time_window.rowsBetween(-4, 0)
rolling_20 = time_window.rowsBetween(-19, 0)
rolling_60 = time_window.rowsBetween(-59, 0)

features_df = (
    df.withColumn("Lag_1_Close", lag("Close", 1).over(time_window))
    .withColumn("Lag_2_Close", lag("Close", 2).over(time_window))
    .withColumn("Lag_5_Close", lag("Close", 5).over(time_window))
    .withColumn("Lag_10_Close", lag("Close", 10).over(time_window))
    .withColumn(
        "Daily_Return",
        when(col("Lag_1_Close") != 0, (col("Close") - col("Lag_1_Close")) / col("Lag_1_Close")).otherwise(0.0),
    )
    .withColumn("Lag_1_Return", lag("Daily_Return", 1).over(time_window))
    .withColumn("Lag_2_Return", lag("Daily_Return", 2).over(time_window))
    .withColumn("RollingAvg_5", avg("Close").over(rolling_5))
    .withColumn("RollingAvg_20", avg("Close").over(rolling_20))
    .withColumn("RollingAvg_60", avg("Close").over(rolling_60))
    .withColumn("RollingStd_20", stddev("Close").over(rolling_20))
    .withColumn("RollingVolume_20", avg("Volume").over(rolling_20))
    .withColumn("Price_Range", col("High") - col("Low"))
    .withColumn(
        "Price_Range_Pct",
        when(col("Open") != 0, (col("High") - col("Low")) / col("Open")).otherwise(0.0),
    )
    .withColumn("Month", month("DateTime"))
    .withColumn("DayOfWeek", dayofweek("DateTime"))
    # ML TARGET: predict the next daily closing price.
    .withColumn("Target_Next_Close", lead("Close", FORECAST_HORIZON).over(time_window))
    .withColumn(
        "Target_Next_Return",
        when(col("Close") != 0, (col("Target_Next_Close") - col("Close")) / col("Close")).otherwise(0.0),
    )
    .dropna()
    .cache()
)

print("✅ Bitcoin daily time-series feature engineering complete. Preview:")
features_df.select(
    "DateTime",
    "Close",
    "Daily_Return",
    "Lag_1_Return",
    "RollingAvg_20",
    "RollingStd_20",
    "Target_Next_Close",
).show(5, truncate=False)

#%%
# Time-based Train/Test Split
# The earliest 80% of rows are used for training; the latest 20% are held out for
# testing. This avoids data leakage from the future into the past.
features_df = features_df.withColumn("row_num", row_number().over(time_window)).cache()
total_rows = features_df.count()
split_index = int(total_rows * 0.8)

train_df = features_df.filter(col("row_num") <= split_index).cache()
test_df = features_df.filter(col("row_num") > split_index).cache()

# Avoid repeated count actions over the same windowed pipeline. These counts are
# known from the row-number split.
print(f"🎓 Training Data: {split_index:,} rows (earlier BTC history)")
print(f"🧪 Testing Data: {total_rows - split_index:,} rows (later BTC history)")

#%%
# Build Machine Learning Pipelines
feature_cols = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Price_Range",
    "Price_Range_Pct",
    "Lag_1_Close",
    "Lag_2_Close",
    "Lag_5_Close",
    "Lag_10_Close",
    "Daily_Return",
    "Lag_1_Return",
    "Lag_2_Return",
    "RollingAvg_5",
    "RollingAvg_20",
    "RollingAvg_60",
    "RollingStd_20",
    "RollingVolume_20",
    "Month",
    "DayOfWeek",
]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="raw_features")
scaler = StandardScaler(inputCol="raw_features", outputCol="scaled_features", withStd=True, withMean=True)

lr = LinearRegression(
    featuresCol="scaled_features",
    labelCol="Target_Next_Close",
    predictionCol="lr_prediction",
    maxIter=20,
    regParam=0.01,
)

gbt = GBTRegressor(
    featuresCol="raw_features",
    labelCol="Target_Next_Close",
    predictionCol="gbt_prediction",
    maxIter=20,
    maxDepth=5,
    seed=42,
)

lr_pipeline = Pipeline(stages=[assembler, scaler, lr])
gbt_pipeline = Pipeline(stages=[assembler, gbt])

print("✅ Bitcoin forecasting ML pipelines constructed.")

#%%
# Train Models
print("🏋️ Training Linear Regression Model...")
lr_model = lr_pipeline.fit(train_df)
lr_predictions = lr_model.transform(test_df).cache()

print("🌲 Training Gradient-Boosted Trees Model...")
gbt_model = gbt_pipeline.fit(train_df)
gbt_predictions = gbt_model.transform(test_df).cache()
print("✅ Distributed training complete.")

#%%
# Evaluate Models
def evaluate_financial_model(predictions_df, prediction_col: str, model_name: str) -> dict:
    rmse_eval = RegressionEvaluator(
        labelCol="Target_Next_Close", predictionCol=prediction_col, metricName="rmse"
    )
    mae_eval = RegressionEvaluator(
        labelCol="Target_Next_Close", predictionCol=prediction_col, metricName="mae"
    )
    r2_eval = RegressionEvaluator(
        labelCol="Target_Next_Close", predictionCol=prediction_col, metricName="r2"
    )

    rmse = rmse_eval.evaluate(predictions_df)
    mae = mae_eval.evaluate(predictions_df)
    r2 = r2_eval.evaluate(predictions_df)

    # Directional accuracy checks whether the model predicted the correct price
    # movement direction compared with the current close price.
    dir_df = predictions_df.withColumn(
        "Actual_Sign", signum(col("Target_Next_Close") - col("Close"))
    ).withColumn("Pred_Sign", signum(col(prediction_col) - col("Close")))

    correct = dir_df.filter(col("Actual_Sign") == col("Pred_Sign")).count()
    total = dir_df.count()
    dir_accuracy = (correct / total) * 100 if total else 0

    return {
        "Model": model_name,
        "RMSE_USD": round(float(rmse), 4),
        "MAE_USD": round(float(mae), 4),
        "R2": round(float(r2), 4),
        "Directional Accuracy (%)": round(float(dir_accuracy), 2),
    }


results = pd.DataFrame(
    [
        evaluate_financial_model(lr_predictions, "lr_prediction", "Linear Regression (Scaled)"),
        evaluate_financial_model(gbt_predictions, "gbt_prediction", "Gradient-Boosted Trees"),
    ]
)

print("\n📊 Model Evaluation Results:")
print(results.to_string(index=False))

#%%
# Visualise Predictions on the Last 120 Test Days
plot_rows = (
    gbt_predictions.select("DateTime", "Close", "Target_Next_Close", "gbt_prediction")
    .orderBy("DateTime")
    .tail(120)
)
plot_pdf = pd.DataFrame(plot_rows, columns=["DateTime", "Close", "Actual_Next_Close", "Predicted_Next_Close"])
plot_pdf["DateTime"] = pd.to_datetime(plot_pdf["DateTime"])

plt.figure(figsize=(16, 6))
plt.plot(plot_pdf["DateTime"], plot_pdf["Actual_Next_Close"], label="Actual Next Close", alpha=0.7)
plt.plot(plot_pdf["DateTime"], plot_pdf["Predicted_Next_Close"], label="GBT Predicted Next Close", linewidth=2)
plt.title("GBT Model: Actual vs. Predicted BTC Next-Day Close", fontsize=16, fontweight="bold")
plt.xlabel("Date")
plt.ylabel("BTC Price (USD)")
plt.legend(loc="upper left")
save_current_plot("btc_predictions.png")

#%%
# CELL 7.1 — Convert Recent Spark Predictions to Pandas for Dashboard-style Visualisation
dashboard_rows = (
    gbt_predictions.select(
        "DateTime",
        "Close",
        "Target_Next_Close",
        "Target_Next_Return",
        "gbt_prediction",
        "Volume",
        "Price_Range",
        "RollingStd_20",
    )
    .orderBy(col("DateTime").desc())
    .limit(365)
)

dashboard_df = dashboard_rows.toPandas()
dashboard_df["DateTime"] = pd.to_datetime(dashboard_df["DateTime"])
dashboard_df = dashboard_df.sort_values("DateTime").reset_index(drop=True)

print("Dashboard Data Shape:", dashboard_df.shape)
print(dashboard_df.head())

#%%
# CELL 7.2 — Dashboard KPIs
total_dashboard_rows = len(dashboard_df)
avg_actual_return = dashboard_df["Target_Next_Return"].mean()
avg_predicted_close = dashboard_df["gbt_prediction"].mean()
avg_volume = dashboard_df["Volume"].mean()
latest_close = dashboard_df["Close"].iloc[-1]
latest_prediction = dashboard_df["gbt_prediction"].iloc[-1]

print("\n📌 Dashboard KPIs")
print(f"- Dashboard records analysed: {total_dashboard_rows:,}")
print(f"- Latest BTC closing price: ${latest_close:,.2f}")
print(f"- Latest predicted next-day close: ${latest_prediction:,.2f}")
print(f"- Average actual next-day return: {avg_actual_return:.6f}")
print(f"- Average predicted close: ${avg_predicted_close:,.2f}")
print(f"- Average daily trading volume: {avg_volume:,.4f}")

#%%
# CELL 7.3 — Dashboard Time-series: Actual vs Predicted Next Close
plt.figure(figsize=(16, 6))
plt.plot(dashboard_df["DateTime"], dashboard_df["Target_Next_Close"], label="Actual Next Close")
plt.plot(dashboard_df["DateTime"], dashboard_df["gbt_prediction"], label="Predicted Next Close")
plt.title("Dashboard View: BTC Actual vs Predicted Next-Day Close")
plt.xlabel("Date")
plt.ylabel("BTC Price (USD)")
plt.legend()
plt.grid(True)
save_current_plot("btc_dashboard_actual_vs_predicted.png")

#%%
# CELL 7.4 — Rolling Behaviour Dashboard
dashboard_df["Rolling_Actual_30"] = dashboard_df["Target_Next_Close"].rolling(window=30).mean()
dashboard_df["Rolling_Pred_30"] = dashboard_df["gbt_prediction"].rolling(window=30).mean()

plt.figure(figsize=(16, 6))
plt.plot(dashboard_df["DateTime"], dashboard_df["Rolling_Actual_30"], label="30-Day Rolling Actual Close")
plt.plot(dashboard_df["DateTime"], dashboard_df["Rolling_Pred_30"], label="30-Day Rolling Predicted Close")
plt.title("Dashboard View: Rolling BTC Forecast Signals")
plt.xlabel("Date")
plt.ylabel("BTC Price (USD)")
plt.legend()
plt.grid(True)
save_current_plot("btc_dashboard_rolling_signals.png")

#%%
# CELL 7.5 — Simple Anomaly Dashboard Using Z-score Threshold on Returns
returns_mean = dashboard_df["Target_Next_Return"].mean()
returns_std = dashboard_df["Target_Next_Return"].std()

dashboard_df["Return_ZScore"] = (dashboard_df["Target_Next_Return"] - returns_mean) / returns_std
anomalies_df = dashboard_df[dashboard_df["Return_ZScore"].abs() > 3].copy()

print(f"Number of anomalous recent daily return rows detected: {len(anomalies_df)}")
print(anomalies_df[["DateTime", "Target_Next_Return", "Return_ZScore"]].head(10))

#%%
# CELL 7.6 — Visual Anomaly Dashboard
plt.figure(figsize=(16, 6))
plt.plot(dashboard_df["DateTime"], dashboard_df["Target_Next_Return"], label="Actual Next-Day Return")
plt.scatter(anomalies_df["DateTime"], anomalies_df["Target_Next_Return"], label="Anomaly")
plt.title("Dashboard View: Bitcoin Return Anomalies")
plt.xlabel("Date")
plt.ylabel("Return")
plt.legend()
plt.grid(True)
save_current_plot("btc_dashboard_return_anomalies.png")

#%%
# CELL 7.7 — Optional Interactive Dashboard Chart (Plotly)
try:
    import plotly.express as px

    fig = px.line(
        dashboard_df,
        x="DateTime",
        y=["Target_Next_Close", "gbt_prediction"],
        title="Interactive Dashboard: BTC Actual vs Predicted Next-Day Close",
    )
    fig.show()
except ImportError:
    print("Plotly is not installed, so the optional interactive dashboard was skipped.")

#%%
# CELL 8.1 — Stop Spark (always run last)
spark.stop()
print("✅ Spark session stopped. Lab complete!")
print("\n📁 Saved output files available in directory:")
for f in [
    "btc_price_volume.png",
    "btc_returns_distribution.png",
    "btc_correlation.png",
    "btc_predictions.png",
    "btc_dashboard_actual_vs_predicted.png",
    "btc_dashboard_rolling_signals.png",
    "btc_dashboard_return_anomalies.png",
]:
    plot_path = PLOTS_DIR / f
    if plot_path.exists():
        print(f" - {plot_path}")
