from functools import reduce
from pyspark.sql import DataFrame, functions as F
from pyspark.sql.functions import col, count, when, isnan, lit, sum as spark_sum
from pyspark.sql.types import NumericType
from pyspark.ml.functions import vector_to_array

def _is_numeric(df, c: str) -> bool:
    """Return True if column c is numeric (so isnan() is valid)."""
    return isinstance(df.schema[c].dataType, NumericType)

def show_df(df, *cols, rows=5, print_str=False, truncate=True, distinct=False, max_cols=None):
    """Show a DataFrame, optionally with a header.
    Set print_str=True to print the header.
    Pass .select() for select columns."""
    if print_str:
        if print_str is True:
            print(f"Showing dataframe: ")
        else:
            print(print_str)
    if cols:
        out = df.select(*cols)
    elif max_cols is not None:
        out = df.select(*df.columns[:max_cols])
    else:
        out = df
    if distinct:
        out = out.distinct()
    out.show(rows, truncate=truncate)

def describe_df(df, *cols, num_rows=5, print_str=False, truncate=True):
    """Show a description of a DataFrame, optionally with a header.
    Pass .select() for select columns.
    Output includes summary statistics for numeric columns."""
    if print_str:
        print(f"Describing dataframe: ")
    if cols:
        df.describe(*cols).show(num_rows, truncate=truncate)
    else:
        df.describe().show(num_rows, truncate=truncate)

def show_df_stats(df, print_str=False):
    if print_str:
        print(f"Showing dataframe stats: ")
    print(f"Number of columns: {len(df.columns)}, Number of rows: {df.count()}")
    print(f"Number of partitions: {df.rdd.getNumPartitions()}")
    print(f"Summary stats:")
    df.summary().show()

def show_x_decimals(
    df: DataFrame,
    decimals: int = 2,
    cols: list[str] | None = None,
    n: int = 20,
    truncate: bool | int = True,
    vertical: bool = False,
):
    """
    Display a Spark DataFrame with numeric columns rounded to `decimals`.
    Safety:
    - Leaves non-numeric columns (string/boolean/date/timestamp/struct/array/etc.) unchanged.
    - Leaves nulls intact.
    - Leaves NaNs intact (Spark round(NaN) -> NaN).
    - Does not mutate the original df.
    - If `cols` is provided, rounds only those columns *that are numeric*; non-numeric cols in
      `cols` are left unchanged (no error).
    """
    if decimals < 0:
        raise ValueError("decimals must be >= 0")

    # Map column -> dataType for quick checks
    dtype_map = {f.name: f.dataType for f in df.schema.fields}

    if cols is None:
        cols_to_round = [c for c, dt in dtype_map.items() if isinstance(dt, NumericType)]
    else:
        missing = [c for c in cols if c not in dtype_map]
        if missing:
            raise ValueError(f"Columns not in DataFrame: {missing}")
        # Only round the numeric ones; ignore non-numeric safely
        cols_to_round = [c for c in cols if isinstance(dtype_map[c], NumericType)]

    exprs = []
    for c in df.columns:
        if c in cols_to_round:
            # round() safely preserves nulls/NaNs. Non-numeric never reaches here.
            exprs.append(F.round(F.col(c), decimals).alias(c))
        else:
            exprs.append(F.col(c))

    df.select(*exprs).show(n=n, truncate=truncate, vertical=vertical)

def show_vector_x_decimals(
    df: DataFrame,
    vector_cols: str | list[str],
    decimals: int = 2,
    n: int = 20,
    truncate: bool | int = True,
):
    """
    Display one or more vector columns as rounded arrays for readability.

    - vector_cols can be a single column name or a list of column names.
    - Leaves the original df unchanged.
    - null vectors stay null; NaNs inside vectors remain NaN.
    """
    if isinstance(vector_cols, str):
        vector_cols = [vector_cols]

    missing = [c for c in vector_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Vector columns not in DataFrame: {missing}")

    exprs = []
    for c in vector_cols:
        arr_col = vector_to_array(F.col(c))
        rounded_arr = F.transform(arr_col, lambda x: F.round(x, decimals))
        exprs.append(rounded_arr.alias(f"{c}_rounded"))

    df.select(*exprs).show(n=n, truncate=truncate)

def show_null_counts(df, num_rows=5):
    # Count missing values per column:
    # - for numeric cols: null OR NaN
    # - for non-numeric cols: null only
    exprs = []
    for c in df.columns:
        if _is_numeric(df, c):
            missing = col(c).isNull() | isnan(col(c))
        else:
            missing = col(c).isNull()
        exprs.append(count(when(missing, lit(1))).alias(c))

    df.select(exprs).show(num_rows, truncate=False)

def show_null_rows(df, num_rows=5):
    # Show rows where ANY column is missing:
    # - numeric cols: null OR NaN
    # - non-numeric cols: null only
    conditions = []
    for c in df.columns:
        if _is_numeric(df, c):
            conditions.append(col(c).isNull() | isnan(col(c)))
        else:
            conditions.append(col(c).isNull())

    if not conditions:
        print("No columns found; cannot check for missing values.")
        return

    missing_any_col = reduce(lambda a, b: a | b, conditions)
    df.filter(missing_any_col).show(num_rows, truncate=False)

def agg_sum(df, group_cols, sums: dict):
    """
        sums: {"source_col_name": "alias_name", ...}
        """
    exprs = [spark_sum(src).alias(alias) for src, alias in sums.items()]
    return df.groupBy(*group_cols).agg(*exprs)
    # older, specific version:
    # return df.groupBy(*cols).agg(
    #     spark_sum("arr_flights").alias("total_flights"),
    #     spark_sum("arr_del15").alias("total_delays_15"),
    # )

