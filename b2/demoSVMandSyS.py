import pandas as pd
from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number
import random

# Import các thư viện cần thiết cho ML
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler # Đã thêm StandardScaler
# Thay đổi từ DecisionTreeClassifier sang LinearSVC
from pyspark.ml.classification import LinearSVC
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# 1. Khởi tạo SparkSession
spark = SparkSession.builder \
    .appName("SysSamplingSVM_BankData") \
    .config("spark.memory.offHeap.enabled", "true") \
    .config("spark.memory.offHeap.size", "2g") \
    .getOrCreate()
sc = spark.sparkContext  # Lấy SparkContext để làm việc với RDD

print("SparkSession đã được khởi tạo.")

# 2. Tải dữ liệu từ file CSV
csv_file_path = "bank-additional-full.csv"  # Đảm bảo file này nằm trong cùng thư mục hoặc cung cấp đường dẫn tuyệt đối

try:
    # Đọc dữ liệu vào DataFrame trước để xử lý tên cột dễ hơn
    df_raw = spark.read \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .option("delimiter", ";") \
        .csv(csv_file_path)

    print(f"Tổng số hàng trong DataFrame gốc: {df_raw.count()}")

    # --- ĐỔI TÊN CÁC CỘT CÓ KÝ TỰ ĐẶC BIỆT ---
    rename_mapping = {
        "`emp.var.rate`": "emp_var_rate",
        "`cons.price.idx`": "cons_price_idx",
        "`cons.conf.idx`": "cons_conf_idx",
        "`nr.employed`": "nr_employed"
    }

    df_cleaned = df_raw
    for old_name_quoted, new_name in rename_mapping.items():
        df_cleaned = df_cleaned.withColumnRenamed(old_name_quoted, new_name)

    print("Đã đổi tên các cột có ký tự đặc biệt.")

    # 3. Chuyển DataFrame đã làm sạch sang RDD để thực hiện Systematic Sampling dựa trên index
    df_rdd = df_cleaned.rdd

    # Tổng số phần tử
    total_count = df_rdd.count()

    # Đánh chỉ số index cho mỗi dòng (index, Row)
    rdd_indexed = df_rdd.zipWithIndex().map(lambda x: (x[1], x[0]))
    print(f"RDD đã được đánh chỉ số với tổng {rdd_indexed.count()} phần tử.")


    # 4. Hàm chia theo Systematic Sampling
    def systematic_sample(rdd_idxed_data, start_index, step_size):
        """
        Thực hiện lấy mẫu có hệ thống từ RDD đã được đánh chỉ số.

        Args:
            rdd_idxed_data (RDD): RDD với định dạng (index, Row_object).
            start_index (int): Chỉ mục bắt đầu cho việc lấy mẫu.
            step_size (int): Bước nhảy giữa các mẫu.

        Returns:
            RDD: RDD của các Row_object đã được lấy mẫu.
        """
        return rdd_idxed_data.filter(lambda x: (x[0] - start_index) % step_size == 0).map(lambda x: x[1])


    # ------------------------------------------------------------------
    # HÀM HUẤN LUYỆN VÀ ĐÁNH GIÁ MÔ HÌNH SUPPORT VECTOR MACHINE TRÊN MỘT MẪU DỮ LIỆU
    # ------------------------------------------------------------------
    def train_and_evaluate_svm(rdd_sample, experiment_id,
                                max_iter=100, reg_param=0.1,
                                train_test_seed=None):
        """
        Chuyển RDD mẫu thành DataFrame, tiền xử lý, huấn luyện Linear SVM và đánh giá.

        Args:
            rdd_sample (RDD): RDD chứa các đối tượng Row của mẫu dữ liệu.
            experiment_id (int): ID của lần chạy thử nghiệm.
            max_iter (int): Số lần lặp tối đa cho thuật toán tối ưu hóa.
            reg_param (float): Tham số chuẩn hóa (regularization parameter).
            train_test_seed (int, optional): Seed cho randomSplit và SVM.

        Returns:
            dict: Chứa ID thử nghiệm, độ chính xác và số mẫu đã sử dụng.
        """
        num_rows_in_sample = rdd_sample.count()
        if num_rows_in_sample == 0:
            print(f"Thử nghiệm {experiment_id}: RDD mẫu rỗng. Bỏ qua.")
            return {"experiment_id": experiment_id, "accuracy": None, "num_samples": 0}

        print(f"\n--- Bắt đầu thử nghiệm {experiment_id} (Linear SVM) ---")
        print(f"Thử nghiệm {experiment_id}: Kích thước RDD mẫu: {num_rows_in_sample} hàng")

        df_sample = spark.createDataFrame(rdd_sample)

        # Tiền xử lý dữ liệu
        categorical_features = ['job', 'marital', 'education', 'default', 'housing', 'loan',
                                'contact', 'month', 'day_of_week', 'poutcome']
        numerical_features = ['age', 'duration', 'campaign', 'pdays', 'previous',
                              'emp_var_rate', 'cons_price_idx', 'cons_conf_idx',
                              'euribor3m', 'nr_employed']

        stages = []
        label_indexer = StringIndexer(inputCol="y", outputCol="indexedLabel")
        stages.append(label_indexer)

        for col in categorical_features:
            indexer = StringIndexer(inputCol=col, outputCol=col + "_indexed")
            stages.append(indexer)

        indexed_categorical_cols = [col + "_indexed" for col in categorical_features]
        all_features = numerical_features + indexed_categorical_cols

        assembler = VectorAssembler(inputCols=all_features, outputCol="features")
        stages.append(assembler)

        # *** BƯỚC MỚI VÀ QUAN TRỌNG CHO SVM: CHUẨN HÓA THUỘC TÍNH (FEATURE SCALING) ***
        # SVM rất nhạy cảm với thang đo của các thuộc tính. Cần chuẩn hóa dữ liệu.
        # StandardScaler sẽ chuẩn hóa các thuộc tính để chúng có giá trị trung bình = 0 và độ lệch chuẩn = 1.
        scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures",
                                withStd=True, withMean=False) # withMean=False nếu muốn chỉ scale bằng std dev
        stages.append(scaler)

        # Xây dựng mô hình Support Vector Machine (LinearSVC)
        # LinearSVC chỉ hỗ trợ phân loại nhị phân. Nếu có nhiều hơn 2 lớp, nó sẽ sử dụng chiến lược One-Vs-Rest.
        svm_model = LinearSVC(labelCol="indexedLabel", featuresCol="scaledFeatures",
                              maxIter=max_iter, regParam=reg_param,
                              seed=train_test_seed)
        stages.append(svm_model)

        pipeline = Pipeline(stages=stages)

        # Chia tập dữ liệu đã lấy mẫu thành tập huấn luyện và tập kiểm tra
        train_df, test_df = df_sample.randomSplit([0.8, 0.2], seed=train_test_seed)

        if train_df.count() == 0 or test_df.count() == 0:
            print(f"Thử nghiệm {experiment_id}: Tập huấn luyện hoặc kiểm tra rỗng sau split. Bỏ qua.")
            return {"experiment_id": experiment_id, "accuracy": None, "num_samples": num_rows_in_sample}

        print(
            f"Thử nghiệm {experiment_id}: Kích thước tập huấn luyện: {train_df.count()}, tập kiểm tra: {test_df.count()}")

        # Huấn luyện mô hình
        model = pipeline.fit(train_df)

        # Đánh giá mô hình
        preds = model.transform(test_df)

        evaluator = MulticlassClassificationEvaluator(
            labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")

        accuracy = evaluator.evaluate(preds)
        print(f"Thử nghiệm {experiment_id}: Độ chính xác (Accuracy): {accuracy:.4f}")

        return {"experiment_id": experiment_id, "accuracy": accuracy, "num_samples": num_rows_in_sample}


    # -------------------------------------------------------------
    # CHẠY SONG SONG VÀ TỔNG HỢP KẾT QUẢ
    # -------------------------------------------------------------

    num_experiments = 5  # Số lượng thử nghiệm bạn muốn chạy
    sampling_step_size = 5  # Mỗi thử nghiệm sẽ lấy mẫu với bước nhảy này (ví dụ: 1/5 dữ liệu gốc)

    all_results = []

    print(f"\nBắt đầu chạy {num_experiments} thử nghiệm song song (lấy mẫu SyS + Linear SVM)...")

    for i in range(num_experiments):
        current_start_offset = random.randint(0, sampling_step_size - 1)

        # Lấy mẫu có hệ thống từ RDD đã đánh chỉ số
        current_sys_sample_rdd = systematic_sample(rdd_indexed, current_start_offset, sampling_step_size)

        # Chạy hàm huấn luyện và đánh giá mô hình Linear SVM trên mẫu SyS này
        result = train_and_evaluate_svm(
            current_sys_sample_rdd,
            experiment_id=i + 1,
            max_iter=100,      # Số lần lặp tối đa
            reg_param=0.1,     # Tham số chuẩn hóa
            train_test_seed=42 + i # Seed cho tính tái lập
        )
        all_results.append(result)

    # 5. Tổng hợp kết quả
    print("\n--- Tổng hợp kết quả từ các lần chạy song song ---")

    successful_runs = [r for r in all_results if r['accuracy'] is not None]

    if successful_runs:
        accuracies = [r['accuracy'] for r in successful_runs]
        avg_accuracy = sum(accuracies) / len(accuracies)
        min_accuracy = min(accuracies)
        max_accuracy = max(accuracies)

        print(f"Tổng số thử nghiệm hoàn tất: {len(successful_runs)}/{num_experiments}")
        print(f"Độ chính xác trung bình: {avg_accuracy:.4f}")
        print(f"Độ chính xác thấp nhất: {min_accuracy:.4f}")
        print(f"Độ chính xác cao nhất: {max_accuracy:.4f}")
        print("\nChi tiết kết quả từng thử nghiệm:")
        for res in successful_runs:
            print(
                f"  Thử nghiệm {res['experiment_id']}: Accuracy = {res['accuracy']:.4f}, Số mẫu = {res['num_samples']}")
    else:
        print("Không có thử nghiệm nào thành công để tổng hợp kết quả.")

except Exception as e:
    print(f"Đã xảy ra lỗi: {e}")
    print("Vui lòng kiểm tra lại đường dẫn file CSV và đảm bảo file có dấu chấm phẩy (;) làm delimiter.")

finally:
    # 6. Dừng SparkSession
    spark.stop()
    print("SparkSession đã dừng.")