import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import Row

# Tạo SparkSession
spark = SparkSession.builder.appName("ParallelDecisionTrees").getOrCreate()
sc = spark.sparkContext

# Đọc dữ liệu từ URL
url = "https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv"
pdf = pd.read_csv(url)

# Chuyển về RDD
rdd = sc.parallelize(pdf.values.tolist())

# Biến RDD thành Row RDD (schema rõ ràng)
rdd_rows = rdd.map(lambda x: Row(
    sepal_length=float(x[0]),
    sepal_width=float(x[1]),
    petal_length=float(x[2]),
    petal_width=float(x[3]),
    species=str(x[4])
))

# SRS chia thành 3 phần bằng tỉ lệ
rdd1, rdd2, rdd3 = rdd_rows.randomSplit([0.33, 0.33, 0.34], seed=42)

from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def train_on_partition(rdd_part):
    df = spark.createDataFrame(rdd_part)

    # Encode label
    indexer = StringIndexer(inputCol="species", outputCol="label")
    df = indexer.fit(df).transform(df)

    # Vectorize features
    assembler = VectorAssembler(
        inputCols=["sepal_length", "sepal_width", "petal_length", "petal_width"],
        outputCol="features"
    )
    df = assembler.transform(df)

    # Train-test split
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=1)

    # Train model
    dt = DecisionTreeClassifier(featuresCol="features", labelCol="label")
    model = dt.fit(train_df)
    preds = model.transform(test_df)

    # Evaluate
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    acc = evaluator.evaluate(preds)
    return acc, model

acc1, model1 = train_on_partition(rdd1)
acc2, model2 = train_on_partition(rdd2)
acc3, model3 = train_on_partition(rdd3)

print(f"Accuracy RDD1: {acc1:.2f}")
print(f"Accuracy RDD2: {acc2:.2f}")
print(f"Accuracy RDD3: {acc3:.2f}")
