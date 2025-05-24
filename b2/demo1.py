from pyspark.sql import SparkSession
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number
import random

# 1. Khởi tạo SparkSession
spark = SparkSession.builder \
    .appName("SystematicSampling") \
    .config("spark.memory.offHeap.enabled", "true") \
    .config("spark.memory.offHeap.size", "2g") \
    .getOrCreate()

# 2. Tải dữ liệu
# Vui lòng cập nhật đường dẫn đến file CSV của bạn
csv_file_path = "bank-additional-full.csv"  # Hoặc đường dẫn tuyệt đối của bạn

try:
    df = spark.read.option("header", "true").option("inferSchema", "true").option("delimiter", ";").csv(csv_file_path)

    print(f"Tổng số hàng trong DataFrame gốc: {df.count()}")

# 3. Áp dụng Systematic Sampling
# Gán chỉ mục duy nhất và liên tục cho mỗi hàng
    window_spec = Window.orderBy(monotonically_increasing_id())
    df_indexed = df.withColumn("row_index", row_number().over(window_spec))

# Định nghĩa bước nhảy (sampling interval)
# Ví dụ: lấy 1/5 dữ liệu, tức là mỗi 5 hàng lấy 1 hàng
    sampling_interval = 5

# Chọn điểm bắt đầu ngẫu nhiên (offset)
# random.seed(42) # Có thể đặt seed để đảm bảo kết quả có thể tái lập
    start_offset = random.randint(0, sampling_interval - 1)

# Lấy mẫu có hệ thống: chọn các hàng mà chỉ mục của chúng có phần dư là start_offset
    sampled_df = df_indexed.filter((df_indexed["row_index"] % sampling_interval) == start_offset)

    print(
    f"\nSố lượng hàng sau khi lấy mẫu có hệ thống (sampling_interval={sampling_interval}, start_offset={start_offset}): {sampled_df.count()}")

    print("\n5 hàng đầu tiên của tập dữ liệu đã lấy mẫu:")
    sampled_df.select("row_index", *df.columns).show(5, truncate=False)

except Exception as e:
    print(f"Đã xảy ra lỗi: {e}")
    print("Vui lòng kiểm tra lại đường dẫn file CSV và đảm bảo file có dấu chấm phẩy (;) làm delimiter.")

finally:
# 4. Dừng SparkSession
    spark.stop()