# Tích hợp DINOv3 Encoder vào mô hình U-TAE dự báo cháy rừng

## 1. Mục tiêu cập nhật
Dự án được phát triển dựa trên mã nguồn U-TAE có sẵn. Thay đổi chính là thay thế CNN Encoder dùng để trích xuất đặc trưng không gian bằng **DINOv3 ViT-S/16 pretrained**, đồng thời giữ nguyên thiết kế của L-TAE (Lightweight Temporal Attention Encoder) và Decoder.

Mục tiêu là tích hợp bộ trích xuất đặc trưng thị giác pretrained vào pipeline dự báo cháy rừng theo chuỗi thời gian, thay vì sử dụng hoàn toàn CNN Encoder ban đầu.

## 2. Các file đã thêm và chỉnh sửa

| File | Trạng thái | Nội dung cập nhật |
| :--- | :--- | :--- |
| `src/models/utae_paps_models/dinov3_encoder.py` | Thêm mới | Triển khai DINOv3 Encoder, chuyển đổi số kênh đầu vào, trích xuất đặc trưng và tạo các bản đồ đặc trưng nhiều tỉ lệ. |
| `src/models/utae_paps_models/utae.py` | Chỉnh sửa | Bổ sung lựa chọn `spatial_backbone`, tích hợp DINOv3 vào U-TAE và kết nối đầu ra Encoder với các thành phần tiếp theo. |
| `src/dataloader/FireSpreadDataset.py` | Chỉnh sửa | Điều chỉnh import để tương thích với môi trường Python/PyTorch đang sử dụng. |
| `test_dinov3_utae.py` | Thêm mới | Kiểm tra khởi tạo mô hình, lan truyền xuôi và lan truyền ngược. |
| `test_dinov3_real_batch.py` | Thêm mới | Kiểm tra mô hình với một mẫu dữ liệu cháy rừng pilot thực tế. |
| `train_dinov3_pilot.py` | Thêm mới | Huấn luyện thử mô hình trên bộ dữ liệu pilot và lưu checkpoint. |
| `train_eval_dinov3_split.py` | Thêm mới | Huấn luyện trên tập train và đánh giá thử trên các mẫu tách theo thứ tự thời gian. |
| `demo/app.py` | Thêm mới | Xây dựng giao diện demo cho phép chọn ngày, xem dữ liệu và gọi mô hình dự báo. |

## 3. Chi tiết vị trí thay đổi Encoder

### 3.1. Xây dựng Encoder mới
**File:** `src/models/utae_paps_models/dinov3_encoder.py`

File này triển khai bộ trích xuất đặc trưng dựa trên DINOv3 ViT-S/16. Các chức năng chính:
* Nạp kiến trúc DINOv3 từ mã nguồn cục bộ và sử dụng pretrained weights.
* Nhận chuỗi đầu vào nhiều thời điểm có dạng `[B, T, C, H, W]`, trong đó B là batch size, T là số thời điểm, C là số kênh dữ liệu.
* Sử dụng lớp tích chập $1 \times 1$ để chuyển đầu vào 40 kênh thành 3 kênh phù hợp với đầu vào của backbone DINOv3.
* Đưa các ảnh tại từng thời điểm qua cùng backbone để trích xuất đặc trưng không gian.
* Lấy đặc trưng trung gian của DINOv3 và dùng các lớp chiếu để điều chỉnh số kênh, độ phân giải đặc trưng theo yêu cầu của U-TAE.
* Trả về danh sách các bản đồ đặc trưng nhiều tỉ lệ, phục vụ quá trình kết hợp thông tin thời gian và giải mã.

> **Lưu ý:** Do DINOv3 được thiết kế cho đầu vào ảnh RGB, lớp chuyển đổi 40 kênh sang 3 kênh đóng vai trò cầu nối giữa dữ liệu vệ tinh đa kênh và backbone pretrained.

### 3.2. Thay thế đường đi của CNN Encoder
**File:** `src/models/utae_paps_models/utae.py`

Đây là file chỉnh sửa trực tiếp kiến trúc U-TAE. 
* **Thay đổi thứ nhất (Bổ sung tham số lựa chọn Encoder):** Thêm các tham số cấu hình:
  * `spatial_backbone`: lựa chọn backbone không gian.
  * `dinov3_repo_dir`: đường dẫn tới mã nguồn DINOv3.
  * `dinov3_weights_path`: đường dẫn tới pretrained weights.
  *(Nhờ đó, mô hình có thể lựa chọn đường Encoder CNN ban đầu hoặc đường Encoder DINOv3)*
* **Thay đổi thứ hai (Khởi tạo DINOv3 Encoder):** 
  * Khi `spatial_backbone="dinov3"`, mô hình khởi tạo `DinoV3Encoder` thay cho đường xử lý CNN Encoder ban đầu. 
  * Khi lựa chọn CNN, nhánh `in_conv` và `down_blocks` ban đầu vẫn được giữ lại.
* **Thay đổi thứ ba (Thay đổi đường trích xuất đặc trưng trong `forward()`):** 
  * Với cấu hình DINOv3, dữ liệu được đưa qua `dinov3_encoder` để tạo danh sách đặc trưng. Đặc trưng ở mức cuối được chuyển sang phần xử lý tiếp theo của U-TAE. 
  * Với cấu hình CNN, mô hình vẫn sử dụng `in_conv` và `down_blocks` như ban đầu.

Như vậy, thay đổi nằm ở đường trích xuất đặc trưng không gian; phần kết hợp đặc trưng thời gian và giải mã tiếp tục sử dụng cấu trúc của U-TAE hiện có.

## 4. Kiến trúc sau khi cập nhật

**Luồng xử lý mới:**
`Chuỗi dữ liệu vệ tinh đa kênh` $\rightarrow$ `Channel Adapter` $\rightarrow$ `DINOv3 ViT-S/16 Encoder` $\rightarrow$ `Feature Projection` $\rightarrow$ `L-TAE` $\rightarrow$ `Decoder` $\rightarrow$ `Bản đồ dự báo cháy ngày tiếp theo`

**Chi tiết luồng hoạt động:**
1. **Đầu vào:** chuỗi 5 thời điểm; trong thí nghiệm pilot, mỗi thời điểm có 40 kênh dữ liệu.
2. **Channel Adapter:** chuyển đổi đầu vào đa kênh thành đầu vào 3 kênh cho DINOv3.
3. **DINOv3 Encoder:** trích xuất các đặc trưng không gian sử dụng pretrained weights.
4. **Feature Projection:** điều chỉnh đặc trưng trung gian để tương thích với các tầng xử lý của U-TAE.
5. **L-TAE:** tổng hợp thông tin từ nhiều thời điểm bằng cơ chế chú ý thời gian.
6. **Decoder:** tái tạo bản đồ dự báo theo không gian và tạo đầu ra dự báo cháy.

**Phạm vi thay đổi:**
* Thay thế đường CNN Encoder bằng DINOv3 trong cấu hình được lựa chọn.
* Giữ nguyên thiết kế của L-TAE.
* Giữ nguyên thiết kế của Decoder và đầu dự báo.
* Bổ sung các bước điều chỉnh đặc trưng để kết nối đầu ra DINOv3 với kiến trúc U-TAE hiện có.

## 5. Dữ liệu và checkpoint

Thí nghiệm pilot sử dụng dữ liệu của sự kiện cháy `25295804` trong năm 2021.
* **Số file GeoTIFF:** 25
* **Số mẫu chuỗi thời gian:** 20
* **Độ dài chuỗi đầu vào:** 5 thời điểm
* **Số kênh đầu vào mỗi thời điểm:** 40
* **Kích thước đầu vào đơn mẫu:** `(5, 40, 288, 192)`
* **Kích thước đầu ra đơn mẫu:** `(1, 288, 192)`

**Các checkpoint đã tạo trong quá trình thử nghiệm:**
* `weights/dinov3_utae_pilot.pt`: checkpoint sau 2 epoch trên bộ pilot.
* `weights/dinov3_utae_eval.pt`: checkpoint trong thí nghiệm huấn luyện và đánh giá theo thứ tự thời gian.

> **Quan trọng:** Pretrained weights của DINOv3 được lưu riêng với mã nguồn mô hình. Khi chạy trong môi trường khác, cần thiết lập đúng đường dẫn tới mã nguồn DINOv3 và checkpoint pretrained.

## 6. Kiểm thử mô hình

Mô hình đã được kiểm tra với dữ liệu pilot thực tế:
* **Đầu vào:** `(1, 5, 40, 288, 192)`
* **Đầu ra:** `(1, 1, 288, 192)`
* Đầu ra có giá trị hữu hạn.
* Mô hình thực hiện được quá trình suy luận và tính loss.

Các bước kiểm thử này xác nhận rằng đường xử lý DINOv3 đã được kết nối và có thể thực thi với dữ liệu pilot.

## 7. Kết quả đánh giá pilot

Trong thí nghiệm đánh giá theo thứ tự thời gian:
* **Tập huấn luyện:** 16 mẫu
* **Tập kiểm tra:** 4 mẫu
* **Số epoch huấn luyện:** 5
* **Test BCE loss:** 0.718110
* **Pixel Accuracy:** 0.0000
* **Precision:** 0.0000
* **Recall:** 1.0000
* **F1-score:** ~0.0001
* **IoU:** 0.0000

*Nhận xét:* Tại ngưỡng phân loại 0.5, mô hình dự đoán gần như toàn bộ pixel là vùng cháy. Vì vậy, mặc dù quá trình tích hợp và suy luận đã chạy được, kết quả hiện tại chưa cho thấy mô hình có khả năng dự báo cháy hữu ích. Bộ dữ liệu kiểm tra chỉ gồm 4 mẫu thuộc cùng một sự kiện cháy, nên kết quả chỉ được xem là đánh giá chẩn đoán ban đầu, không đại diện cho khả năng tổng quát hóa sang những sự kiện khác.

## 8. Hạn chế và hướng phát triển

* Mở rộng dataset với nhiều sự kiện cháy độc lập.
* Kiểm tra chất lượng và sự nhất quán của nhãn cháy ngày kế tiếp.
* Điều chỉnh quá trình huấn luyện và xử lý mất cân bằng giữa pixel cháy và không cháy.
* Đánh giá trên nhiều sự kiện riêng biệt bằng Precision, Recall, F1-score và IoU.
* Kiểm tra khả năng chạy mô hình trong môi trường triển khai trực tuyến.
* Chỉ sử dụng mô hình cho cảnh báo thực tế khi đã có đánh giá phù hợp và kiểm chứng đầy đủ.

## 9. Kết luận

Thay đổi trọng tâm của dự án là tích hợp DINOv3 ViT-S/16 làm bộ trích xuất đặc trưng không gian trong U-TAE, đồng thời duy trì thiết kế của L-TAE và Decoder.

Kết quả hiện tại xác nhận mô hình có thể khởi tạo, xử lý chuỗi dữ liệu pilot và tạo bản đồ đầu ra. Tuy nhiên, chất lượng dự báo vẫn chưa đạt yêu cầu và cần được cải thiện thông qua mở rộng dữ liệu, huấn luyện và đánh giá trên các sự kiện cháy độc lập.