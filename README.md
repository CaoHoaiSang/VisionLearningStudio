# Vision Lab Studio

Ứng dụng desktop Windows để học trọn quy trình Vision 2D cho bài gắp–đặt hai mảnh vải: lấy ảnh Basler, tách vật, học mẫu, định vị, đặt TCP, đổi pixel sang tọa độ Robot và giao tiếp với DeltaX Studio qua TCP.

Ứng dụng này là chương trình độc lập phục vụ học tập. Nó không sửa file thực thi, không bỏ qua cơ chế kích hoạt và không thay thế bản quyền HOXCO.

## 1. Chạy chương trình

Yêu cầu: Windows 10/11, Python 3.10 trở lên.

1. Cài **Basler pylon Camera Software Suite** phù hợp với camera và Windows.
2. Mở PowerShell trong thư mục này.
3. Cài thư viện:

```powershell
python -m pip install -r requirements.txt
```

4. Chạy bằng `run.bat`, hoặc:

```powershell
python modern_app.py
```

Nếu chỉ học bằng ảnh tĩnh, chương trình vẫn chạy khi chưa có camera Basler. Tab Camera sẽ báo chưa có `pypylon` hoặc chưa tìm thấy camera.

## 2. Quy trình học và vận hành

### B1 — Tách và lấy hai mẫu

1. Chọn sản phẩm `radxa-live-study` để dùng mẫu Radxa đã chuẩn bị, hoặc bấm **Thêm**.
2. Mở ảnh có cả hai mảnh.
3. Chọn **Mảnh 1 · DƯỚI** hoặc **Mảnh 2 · TRÊN (gắp)**.
4. Chọn riêng một phương pháp tách cho từng mảnh:
   - **Trừ ẢNH NỀN**: ổn định khi camera, bàn và ánh sáng cố định.
   - **Theo ĐỘ SÁNG**: phù hợp khi vật sáng/tối tách biệt rõ với nền.
   - **Theo MÀU vải**: bấm **Bấm ảnh để lấy màu**, sau đó bấm đúng vùng vải.
5. Với trừ nền, dùng **Nền từ camera** hoặc **Mở ảnh nền…**. Hai chức năng này lưu nền riêng cho sản phẩm.
6. Có thể bật **Tự động chọn ngưỡng sáng** để dùng Otsu cho chế độ độ sáng hoặc trừ nền.
7. Chỉnh mask bằng các thanh Làm mượt, ngưỡng, dung sai LAB, khử nhiễu, diện tích tối thiểu và ngưỡng hoa văn.
8. Nếu một mảnh bị chia thành nhiều vùng do che khuất, bật **Gộp vùng cùng màu khi bị che khuất**.
9. Bấm **Chạy tách**. Chỉ giữ vùng đúng của mảnh đang chọn.
10. Bấm **Lấy mẫu vào mảnh đang chọn**.
11. Làm lại với mảnh còn lại.

Panel tham số B1 có thanh cuộn riêng; lăn chuột hoặc kéo thanh cuộn bên phải để xuống phần Lấy mẫu.

Mask tốt có vật màu trắng, nền đen, ít lỗ và không dính mép khung hình.

### B2 — Quan hệ thiết kế

Kéo hai mảnh đến vị trí mong muốn sau khi đặt. B2 dùng canvas native nên chỉ di chuyển đối tượng đang kéo, không dựng lại toàn ảnh theo từng pixel. Chọn từng mảnh và chỉnh góc. Chương trình lưu `Δx`, `Δy`, `Δθ`; tùy chọn ghost trong Vận hành dùng chính quan hệ này để xem trước vị trí đặt.

### B3 — TCP

Bấm trên từng ảnh mảnh tại điểm đầu hút Robot thực sự chạm vải. B3 chỉ hiển thị ROI đã áp mask và tự cắt sát vật, vì vậy nền băng tải không còn đi theo ROI. Nút **Làm sạch ROI** loại vùng nhiễu, giữ thành phần lớn nhất và tự bù TCP khi crop. Nút **TCP về tâm vật** tính trọng tâm hình học của mask sạch cho riêng từng mảnh; sau khi chọn vẫn cần bấm **Lưu mẫu thiết kế**.

### Học Vision — phòng thực hành tương tác

Chọn một trong 15 bài rồi kéo thanh trượt để xem ảnh thay đổi ngay. Các bài có mô phỏng riêng cho kênh BGR/Gray/LAB, histogram và ánh sáng, threshold/Otsu, morphology, contour/minAreaRect, TCP, trừ nền, template matching, ORB, che khuất, pixel→mm, quan hệ PICK/PLACE, Exposure/Gain, tuổi frame và an toàn Robot. Chọn **DƯỚI** hoặc **TRÊN (gắp)** để học trên đúng template của từng mảnh.

Các thông số trong trang học là mô phỏng độc lập, không ghi đè cấu hình B1 và không gửi lệnh Robot.

### Camera Basler

1. Cắm camera và mở pylon Viewer để xác nhận camera hoạt động.
2. Đóng chức năng grab của pylon Viewer để tránh hai chương trình giữ camera cùng lúc.
3. Trong tab **CAMERA BASLER**, bấm **Quét Basler**, chọn camera và **Kết nối**.
4. Đặt Exposure/Gain, ROI, giới hạn FPS và Trigger rồi **Áp dụng**. Chương trình tắt auto trước khi ghi giá trị. Với Trigger `Software`, dùng nút **PHÁT SOFTWARE TRIGGER**; `Line1` dành cho cảm biến/PLC ngoài.
5. Bấm **LIVE / DỪNG**. Luồng dùng `GrabStrategy_LatestImageOnly`, nên frame cũ bị bỏ thay vì xếp hàng gây trễ.
6. Dùng **Chụp → CÀI ĐẶT** để học mẫu hoặc **Chụp → VẬN HÀNH** để kiểm tra.

Nếu preview trễ, giảm Exposure, giảm độ phân giải/ROI trong cấu hình camera, dùng GigE có dây, kiểm tra packet loss và không mở đồng thời nhiều viewer.

### Vận hành và cầu nối DeltaX

1. Chọn nguồn **Ảnh tĩnh** để mở file, dùng ảnh mẫu hoặc lấy frame Basler mới nhất. Chọn **Robot · DeltaX** để mở cầu nối và nhận ảnh trực tiếp từ Studio.
2. Trong **CẤU HÌNH TRẠM**, đặt cổng `6001`, cho phép LAN, dấu trục W, bù góc, `mm/pixel`, gốc U/V và giới hạn tuổi kết quả.
3. Chọn thuật toán:
   - **Tự động**: ưu tiên ORB khi mẫu đủ đặc trưng, tự rơi về contour.
   - **Đặc trưng vải — xoay 360°, che khuất**: ORB + homography.
   - **So khớp mẫu — ảnh xám**: edge template matching theo góc/scale.
   - **So khớp mẫu — biến dạng**: tách vật + so hình contour.
   - **Chỉ tách nền**: ép dùng ảnh nền cho cả hai mảnh.
4. Bấm **MỞ CẦU NỐI ROBOT** trước khi chạy Blockly.
5. Firewall Windows phải cho phép ứng dụng Python nhận TCP cổng 6001 trên mạng
   Private. Luật dành riêng cho `HoxcoVision.exe` không áp dụng cho Vision Learning
   Studio. Trên máy hiện tại đã có luật `Vision Learning Studio TCP 6001 from Radxa`
   chỉ cho phép địa chỉ `192.168.1.29`.
6. Trên Radxa kiểm tra:

```bash
ping -c 4 <IP_PC>
python3 -c "import socket; s=socket.create_connection(('<IP_PC>',6001),2); print('KET NOI OK'); s.close()"
```

DeltaX gửi mỗi frame theo giao thức hiện tại: `uint32 độ dài JPEG` + JPEG. Vision Lab trả `uint32 số vật`, rồi mỗi vật là `<ifffff`: `type, tcp_u, tcp_v, width, height, angle`.

Phần xem trước ĐẶT dùng đúng phép biến đổi đã thiết lập trong B2: góc gốc của
mask hai mảnh + góc chỉnh bằng thanh trượt + pose thực của mảnh DƯỚI. Vì vậy góc
ghost giữ đúng quan hệ trực quan trong B2 kể cả khi hai ROI được học ở góc khác nhau.

Ứng dụng chỉ hiển thị frame mới nhất. Nếu thời gian xử lý vượt **Tuổi kết quả tối đa**, tọa độ bị từ chối thay vì gửi kết quả quá chậm cho Robot.

## 3. Nhập mẫu HOXCO đang có

Bấm **Nhập HOXCO**, chọn thư mục sản phẩm chứa `template.json`, `p1.png`, `p2.png` và tùy chọn `bg.png`. Chương trình tạo một bản sao trong `data/products`; thư mục HOXCO gốc không bị sửa.

## 4. Cấu trúc dữ liệu

Mỗi sản phẩm nằm tại:

```text
data/products/<tên-sản-phẩm>/
├── product.json
├── design_source.jpg
├── background.png
├── bottom_template.png
├── bottom_mask.png
├── top_template.png
├── top_mask.png
└── captures/
```

`product.json` chứa tham số tách riêng từng mảnh, quan hệ thiết kế, TCP và hằng số trạm.

## 5. Kiểm thử

Chạy `run_tests.bat`, hoặc:

```powershell
python -m unittest discover -s tests -v
```

Bộ test kiểm tra lưu/nhân bản sản phẩm, nhập HOXCO, học–detect hai mảnh và giao thức TCP.

Bộ kiểm thử hiện còn kiểm tra thêm ngưỡng Otsu tự động, trừ nền tự động, gộp fragment, làm sạch ROI, cuộn B1, chuyển nguồn ảnh và hiệu năng kéo B2.

## 6. An toàn

Trước khi cho Robot chạy tự động:

1. Kiểm tra TCP và dấu góc W bằng một mảnh có góc biết trước.
2. Kiểm tra `mm/pixel` hoặc hiệu chuẩn mặt phẳng bằng thước chuẩn.
3. Chạy với Z an toàn, tốc độ thấp, đầu hút tắt và nút dừng khẩn cấp sẵn sàng.
4. Chỉ cho phép pick/place khi đủ đúng hai type: `0 = DƯỚI`, `1 = TRÊN`.
5. Không dùng một kết quả cũ sau khi vật hoặc camera đã di chuyển.
