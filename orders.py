from pyspark.sql import SparkSession
import os

# Thêm cấu hình để tắt kiểm tra bảo mật
# Đặt biến môi trường PYSPARK_SUBMIT_ARGS
os.environ['PYSPARK_SUBMIT_ARGS'] = '--conf spark.driver.extraJavaOptions="-Djava.security.manager=allow" --conf spark.executor.extraJavaOptions="-Djava.security.manager=allow" pyspark-shell'

# Tạo SparkSession
spark = SparkSession.builder \
    .appName("Word Count with PySpark") \
    .master("local[*]") \
    .config("spark.driver.extraJavaOptions", "-Djava.security.manager=allow") \
    .config("spark.executor.extraJavaOptions", "-Djava.security.manager=allow") \
    .getOrCreate()

order_rdd = spark.sparkContext.textFile("orders_67000.txt")
mapped_rdd = order_rdd.map(lambda x: (x.split(",")[3], 1))
reduce_rdd = mapped_rdd.reduceByKey(lambda x,y: x+y)
print(reduce_rdd.collect())

# Dừng Spark
spark.stop()