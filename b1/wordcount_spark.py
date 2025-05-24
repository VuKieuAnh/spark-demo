from pyspark.sql import SparkSession

# Tạo SparkSession
spark = SparkSession.builder \
    .appName("Word Count with PySpark") \
    .master("local[*]") \
    .getOrCreate()

# Đọc file input.txt
rdd = spark.sparkContext.textFile("input.txt")

# Xử lý và đếm từ
# rdd1 = rdd.collect()
rdd2 = rdd.flatMap(lambda line: line.split())
rdd3= rdd2.map(lambda  w: (w.lower(), 1))
rdd4= rdd3.reduceByKey(lambda a, b: a + b)

for word, count in rdd4.collect():
    print(f"{word}: {count}")

# word_counts = (
#     rdd.flatMap(lambda line: line.split())       # Tách dòng thành từ
#        .map(lambda word: (word.lower(), 1))      # Đưa về dạng (từ, 1)
#        .reduceByKey(lambda a, b: a + b)          # Cộng số lần xuất hiện
# )

# In kết quả
# for word, count in word_counts.collect():
#     print(f"{word}: {count}")

# Dừng Spark
spark.stop()
