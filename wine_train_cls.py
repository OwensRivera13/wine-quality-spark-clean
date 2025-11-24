# ==========================================================
# WINE QUALITY TRAINING (CLASSIFIER VERSION) - OWENS
# ==========================================================
# goal: train a classification model (not regression) so we can
# report F1 score later. we’ll use multinomial logistic regression.
# also cleaning up the weird quoted headers like before.
# ==========================================================

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# imports i actually use here
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# start spark
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
spark = SparkSession.builder.appName("WineQualityTrainingClassifier").getOrCreate()

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# read training data (semicolon-separated + quoted headers)
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
df = spark.read.csv("TrainingDataset.csv", header=True, inferSchema=True, sep=';')

print("==========================================================")
print("Before cleaning column names:")
df.printSchema()
print("==========================================================")

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# clean column names (strip extra quotes/spaces)
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
for name in df.columns:
    clean = name.replace('"', '').strip()
    if clean != name:
        df = df.withColumnRenamed(name, clean)

print("==========================================================")
print("After cleaning column names:")
df.printSchema()
print("==========================================================")
df.show(5)

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# feature vector (same set we used before)
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
df = assembler.transform(df)

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# label prep: quality is already an integer 1..10.
# logistic regression wants a double label; just cast to double.
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
df = df.withColumn("label", col("quality").cast("double"))

# quick split just to sanity-check model quality here
train_df, holdout_df = df.randomSplit([0.8, 0.2], seed=42)

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# multinomial logistic regression (multi-class)
# you can tweak maxIter / regParam later if you want to tune
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
lr = LogisticRegression(
    featuresCol="features",
    labelCol="label",
    maxIter=200,           # a bit higher to help convergence
    regParam=0.0,          # can try 0.01 or 0.1 if you want regularization
    elasticNetParam=0.0,   # pure L2 right now
    family="multinomial"   # multi-class softmax
)

model = lr.fit(train_df)

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# quick evaluation on the small holdout split (not the official validation)
# just to see if it’s sane
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
preds = model.transform(holdout_df)

evaluator_f1 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
evaluator_acc = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")

f1 = evaluator_f1.evaluate(preds)
acc = evaluator_acc.evaluate(preds)

print("==========================================================")
print("CLASSIFIER TRAINED (quick holdout metrics below)")
print(f"Accuracy: {acc:.4f}")
print(f"F1 Score: {f1:.4f}")
print("==========================================================")

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# save the classifier model for the real prediction app
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
model.save("wine_quality_cls_model")

spark.stop()
# end (classifier training)

