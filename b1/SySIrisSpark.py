import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import Row

# Tạo SparkSession
spark = SparkSession.builder.appName("SysSamplingDecisionTree").getOrCreate()
sc = spark.sparkContext

# Đọc từ URL bằng pandas
url = "https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv"
pdf = pd.read_csv(url)

# Chuyển sang RDD
rdd = sc.parallelize(pdf.values.tolist())

# Tổng số phần tử và số phần cần chia
total_count = rdd.count()
n_parts = 3
step = total_count // n_parts  # k = floor(N / n)

# Đánh chỉ số index cho mỗi dòng
rdd_indexed = rdd.zipWithIndex().map(lambda x: (x[1], x[0]))

# Hàm chia theo SyS
def systematic_sample(rdd_idxed, start, step):
    return rdd_idxed.filter(lambda x: (x[0] - start) % step == 0).map(lambda x: x[1])

# Lấy 3 mẫu hệ thống SyS bắt đầu tại index 0,1,2
rdd1 = systematic_sample(rdd_indexed, 0, n_parts)
rdd2 = systematic_sample(rdd_indexed, 1, n_parts)
rdd3 = systematic_sample(rdd_indexed, 2, n_parts)

from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def train_on_sys_sample(rdd_sample):
    rdd_row = rdd_sample.map(lambda x: Row(
        sepal_length=float(x[0]),
        sepal_width=float(x[1]),
        petal_length=float(x[2]),
        petal_width=float(x[3]),
        species=str(x[4])
    ))
    df = spark.createDataFrame(rdd_row)

    indexer = StringIndexer(inputCol="species", outputCol="label")
    df = indexer.fit(df).transform(df)

    assembler = VectorAssembler(
        inputCols=["sepal_length", "sepal_width", "petal_length", "petal_width"],
        outputCol="features"
    )
    df = assembler.transform(df)

    train_df, test_df = df.randomSplit([0.8, 0.2], seed=1)

    dt = DecisionTreeClassifier(featuresCol="features", labelCol="label")
    model = dt.fit(train_df)
    preds = model.transform(test_df)

    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    acc = evaluator.evaluate(preds)
    return acc

acc1 = train_on_sys_sample(rdd1)
acc2 = train_on_sys_sample(rdd2)
acc3 = train_on_sys_sample(rdd3)

print(f"Accuracy SyS Sample 1: {acc1:.2f}")
print(f"Accuracy SyS Sample 2: {acc2:.2f}")
print(f"Accuracy SyS Sample 3: {acc3:.2f}")

# In ra nội dung của từng RDD
print("===== Sample 1 =====")
for row in rdd1.collect():
    print(row)

print("\n===== Sample 2 =====")
for row in rdd2.collect():
    print(row)

print("\n===== Sample 3 =====")
for row in rdd3.collect():
    print(row)

