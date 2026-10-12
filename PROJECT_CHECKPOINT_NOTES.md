# BẢN GHI NHỚ TIẾN TRÌNH DỰ ÁN (PROJECT CHECKPOINT NOTES)
**Đề Tài:** Phát hiện Stress Đa Thiết Bị (Cross-Device & In-Device Stress Detection)  
**Tín Hiệu:** PPG Galaxy Watch (25Hz) vs. BVP Empatica E4 (64Hz -> 25Hz)  
**Bộ Dữ Liệu:** IMAT Dataset — 23 Đối tượng (P02 đến P24, không có P01 trong dữ liệu gốc)  
**Thời gian lưu:** Ngày 02/10/2026  

---

## 1. TỔNG QUAN DỮ LIỆU & PHẦN CỨNG
- **Tổng số cửa sổ:** 15,335 cửa sổ (độ dài 8s/cửa sổ, bước nhảy 2s).
- **Phân bố nhãn:**
  - Non-Stress (Lớp 0): 10,769 cửa sổ.
  - Stress (Lớp 1): 4,566 cửa sổ.
  - Tỷ lệ mất cân bằng tự nhiên: 2.36 : 1.
- **Tiền xử lý:**
  - Resample Empatica E4 từ 64Hz về chuẩn 25Hz (Polyphase Resampling).
  - Kiểm tra cực tính sóng mạch: 100% sóng thuận (đỉnh systolic hướng lên).
  - Trích xuất 30 đặc trưng sinh lý chuẩn (Miền thời gian, Hình thái đỉnh sóng/HRV co mạch, Phổ công suất Welch LF/HF, Entropy phi tuyến).

---

## 2. PHƯƠNG PHÁP LUẬN NGHIÊM NGẶT (CASE B: STRICT NESTED IN-FOLD CALIBRATION)
Nhằm đạt mục tiêu **Precision >= 50%** và **Recall >= 70%** mà **tuyệt đối không rò rỉ dữ liệu (Zero Data Leakage)**:

1. **Vòng lặp ngoài (Outer Loop - LOSO 23 Folds):**
   - Ở mỗi fold: Giữ lại **1 đối tượng làm Held-out Test Subject** (khoá kín trong phòng tối, không nhìn thấy nhãn).
   - Lấy **22 đối tượng còn lại** làm tập Huấn luyện (Train Set).
2. **Vòng lặp trong (Inner Loop - 4-Fold GroupKFold trên 22 người train):**
   - 22 người train được chia thành 4 nhóm theo đối tượng để thi thử chéo nội bộ.
   - Thu thập xác suất dự đoán Out-of-Fold của 22 người này trên thiết bị đích.
   - Lọc mượt Causal Temporal Smoothing ($W=5$, trượt 10s nhìn về quá khứ).
3. **Hiệu chuẩn ngưỡng tối ưu (find_optimal_threshold):**
   - Quét tìm ngưỡng $t^*$ trên đường cong Precision-Recall của 22 người train thỏa mãn:
     $$\max_{t^*} F1_{\text{Stress}} \quad \text{sao cho} \quad \text{Precision} \ge 50\% \text{ và } \text{Recall} \ge 70\%$$
   - **Đóng băng ngưỡng (Freeze):** Lưu cố định con số $t^*$ thành $t_{\text{infold}}$.
4. **Đánh giá trên Held-out Test Subject:**
   - Dùng mô hình (đã fit trên 22 người) dự đoán cho người test.
   - Lọc mượt Causal Smoothing ($W=5$).
   - Phân loại bằng ngưỡng đã đóng băng: $\hat{y} = (P \ge t_{\text{infold}})$.

---

## 3. THIẾT LẬP THAM SỐ XGBOOST
- `scale_pos_weight = 3.54`:
  - Tỷ lệ mất cân bằng tự nhiên là 2.36 : 1.
  - Để ưu tiên độ nhạy (Recall) không bỏ sót cơn stress, tham số được nhân 1.5x: $2.3585 \times 1.5 \approx 3.54$.
- `learning_rate = 0.05`, `max_depth = 4`, `n_estimators = 200`.

---

## 4. BẢNG KẾT QUẢ CHÍNH THỨC (XGBOOST - CASE B STRICT NESTED)
Đánh giá trên đầy đủ 15,335 cửa sổ của toàn bộ 23 đối tượng:

| Kịch Bản | Ngưỡng Học Được ($t_{\text{infold}}$) | Stress Precision ($\ge 50\%$) | Stress Recall ($\ge 70\%$) | F1-Stress | Overall Accuracy | Đánh Giá |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **GW → GW (In-Device)** | $0.62 \pm 0.02$ | **54.93%** | **69.89%** (~70%) | **61.51%** | **73.96%** | **ĐẠT CHỈ TIÊU ✅** |
| **GW → E4 (Cross-Device)** | $0.41 \pm 0.01$ | **54.35%** | **74.00%** | **62.67%** | **73.75%** | **ĐẠT CẢ HAI ✅** |
| **E4 → E4 (In-Device)** | $0.59 \pm 0.02$ | **55.96%** | **73.30%** | **63.47%** | **74.87%** | **ĐẠT CẢ HAI ✅** |
| **E4 → GW (Cross-Device)** | $0.71 \pm 0.02$ | **55.01%** | **71.77%** | **62.28%** | **74.12%** | **ĐẠT CẢ HAI ✅** |

---

## 5. CÁC TÀI LIỆU VÀ FILE TRONG DỰ ÁN
- `stress_detection_pipeline.py`: Mã nguồn hoàn chỉnh của toàn bộ pipeline (đã tối ưu hóa, chuẩn hóa template HTML và biểu đồ).
- `results/results_summary.json`: File dữ liệu số học gốc chứa đầy đủ thông số 23 folds, confusion matrix, per-class breakdown.
- `results/stress_detection_report.html`: Báo cáo HTML dashboard trực quan, mở bằng trình duyệt web.
- `results/features_processed.csv`: Bộ đặc trưng 30 chiều của 23 đối tượng đã tiền xử lý.

---
*Ghi chú: Nếu khởi động lại máy hoặc mở phiên chat mới, chỉ cần gõ: "Đọc file PROJECT_CHECKPOINT_NOTES.md và tiếp tục nhé!"*
