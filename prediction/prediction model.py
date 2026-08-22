# Databricks notebook source
# MAGIC %pip install catboost

# COMMAND ----------

from pyspark.sql import functions as F
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# COMMAND ----------

# load training and test data into current environment
train = spark.table("cell_catalog.celldata.train")
test = spark.table("cell_catalog.celldata.test")

# COMMAND ----------

# select gene columns 
genes = train.columns[1:201]
# select relevant metadata columns
train_metadata = [
    "Datasets",
    "volume",
    "center_x",
    "center_y",
    "Excitatory_vs_Inhibitory",
    "Segment",
    "Gender",
    "Mouse_ID",
    "AP_position",
    "Section_ID",
    "MERFISH_cell_type_annotation"
]

train_features = genes + train_metadata

test_metadata = [
    "Datasets",
    "volume",
    "center_x",
    "center_y",
    "Excitatory_vs_Inhibitory",
    "Segment",
    "Gender",
    "Mouse_ID",
    "AP_position",
    "Section_ID"
]

test_features = genes + test_metadata

# COMMAND ----------

# convert to pandas
train_pd  = train.toPandas()[train_features]
test_pd = test.toPandas()[test_features]

train_pd.head(5)
test_pd.head(5)

# COMMAND ----------

# impute categorical columns with "unknown"
categorical = [
    "Datasets",
    "Excitatory_vs_Inhibitory",
    "Segment",
    "Gender",
    "Mouse_ID",
    "AP_position",
    "Section_ID"
]

for col in categorical:
    train_pd[col] = train_pd[col].fillna("unknown").astype(str)
    test_pd[col] = test_pd[col].fillna("unknown").astype(str)


# COMMAND ----------

# impute the numerical columns with the median
numerical = [
   "volume",
    "center_x",
    "center_y", 
]

for col in numerical:
    train_pd[col] = train_pd[col].fillna(train_pd[col].median())
    test_pd[col] = test_pd[col].fillna(train_pd[col].median())

# COMMAND ----------

X = train_pd.drop(columns=["MERFISH_cell_type_annotation"])
y = train_pd["MERFISH_cell_type_annotation"]

X_test = test_pd.copy()

# COMMAND ----------

# split the training data into training and validation sets
X_train, X_val, y_train, y_Val = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# COMMAND ----------

# get the indices of the categorical columns
cat_features = [
    X_train.columns.get_loc(col)
    for col in categorical
]

# COMMAND ----------

# train the CatBoost model
model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=5,
    loss_function="MultiClass",
    eval_metric="Accuracy",
    random_seed=42,
    verbose=50,
    thread_count=1
)

model.fit(
    X_train,
    y_train,
    eval_set=(X_val, y_Val),
    cat_features=cat_features,
    early_stopping_rounds=50
)

# COMMAND ----------


y_pred = model.predict(X_val).flatten()

accuracy = accuracy_score(y_Val, y_pred)
kappa = cohen_kappa_score(y_Val, y_pred)

print(f"Validation accuracy:{accuracy:.2f}")
print(f"Validation kappa:{kappa:.2f}")

# COMMAND ----------

# final prediction 
model = CatBoostClassifier(
    iterations=350,
    learning_rate=0.05,
    depth=5,
    loss_function="MultiClass",
    random_seed=42,
    verbose=1,
    thread_count=1
)

model.fit(
    X,
    y,
    cat_features=cat_features
)

# COMMAND ----------

y_pred =  model.predict(X_test)
y_pred = y_pred.ravel()

# COMMAND ----------

cell = test.select("Cell_ID").toPandas()["Cell_ID"]
final_prediction = pd.DataFrame({
    "Cell_ID": cell,
    "MERFISH_cell_type_annotation": y_pred
})

print(final_prediction.shape)
print(final_prediction.head(5))

# COMMAND ----------

submission_path = (
    "/Volumes/cell_catalog/celldata/workspace/"
    "prediction_day3.csv"
)

final_prediction.to_csv(
    submission_path,
    index=False
)

print(f"Final prediction saved to: {submission_path}")

# COMMAND ----------

