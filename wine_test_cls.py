# ==========================================================
# WINE QUALITY PREDICTION (VALIDATION TEST) - OWENS
# ==========================================================
# goal: load the saved model from training, run it on
# ValidationDataset.csv, and print the F1 + sample predictions.
# ==========================================================

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# imports i actually use
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import sys

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# check if the user passed a test dataset path
# (we can pass it as: spark-submit wine_test_cls.py ValidationDataset.csv)
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
if len(sys.argv) < 2:
    print("Usage: spark-submit wine_test_cls.py <path_to_validation_csv>")
    sys.exit(1)

test_path = sys.argv[1]

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# start spark session
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
spark = SparkSession.builder.appName("WineQualityValidation").getOrCreate()

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# load the validation data
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
df_val = spark.read.csv(test_path, header=True, inferSchema=True, sep=';')

print("==========================================================")
print("Before cleaning validation dataset column names:")
df_val.printSchema()
print("==========================================================")

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# clean weird header quotes again just like in training
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
for name in df_val.columns:
    clean = name.replace('"', '').strip()
    if clean != name:
        df_val = df_val.withColumnRenamed(name, clean)

print("==========================================================")
print("After cleaning validation dataset column names:")
df_val.printSchema()
print("==========================================================")

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# feature preparation
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
feature_cols = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol"
]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df_val = assembler.transform(df_val)

df_val = df_val.withColumn("label", col("quality").cast("double"))

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# load the trained classifier model
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
model = LogisticRegressionModel.load("wine_quality_cls_model")

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# make predictions
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
predictions = model.transform(df_val)

print("==========================================================")
print("Showing a few sample predictions from validation set:")
predictions.select("quality", "prediction", "probability").show(10)
print("==========================================================")

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# evaluate performance (Accuracy + F1)
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
evaluator_f1 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
evaluator_acc = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")

f1 = evaluator_f1.evaluate(predictions)
acc = evaluator_acc.evaluate(predictions)

print("==========================================================")
print("VALIDATION RESULTS")
print(f"Accuracy: {acc:.4f}")
print(f"F1 Score: {f1:.4f}")
print("==========================================================")

spark.stop()
# end

