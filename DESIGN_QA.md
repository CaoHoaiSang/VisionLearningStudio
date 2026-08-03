# Design QA

final result: passed

## TCP tâm vật và Học Vision tương tác — 2026-07-30

- B3 có nút `TCP về tâm vật` riêng cho mảnh TRÊN và DƯỚI; tâm được tính bằng moment của mask sạch, không giả định tâm ảnh crop là tâm vật.
- Trang Học Vision có 15 bài thực hành với phản hồi hình ảnh theo thanh trượt; thông số mô phỏng không ghi đè B1 và không gửi Robot.
- Đã kiểm tra trực quan các bài Threshold, ORB, PICK/PLACE và B3 tại 1680×960.
- Automated tests: **15/15 đạt**, gồm test tâm TCP và render đủ 15 bài.

## Sửa lỗi kết nối và góc ghost — 2026-07-30

- Radxa `192.168.1.29` đã kết nối `ESTABLISHED` tới
  `192.168.1.90:6001`.
- Dữ liệu hai chiều đã được xác nhận bằng bộ đếm TCP tăng liên tục:
  Radxa gửi JPEG và nhận payload kết quả.
- Nguyên nhân lỗi kết nối là luật Firewall cũ chỉ gắn với
  `HoxcoVision.exe`; đã thêm luật `Vision Learning Studio TCP 6001 from
  Radxa` cho TCP 6001, mạng Private và riêng IP `192.168.1.29`.
- Ghost hiện dùng cả góc trục gốc của hai mask, góc chỉnh B2 và pose mảnh
  DƯỚI ngoài hiện trường. Dấu góc OpenCV đã được đổi đúng.
- Kiểm chứng mẫu hiện tại: B2 `Δθ = +0.39°`; ghost sau biến đổi cũng
  `Δθ = +0.39°`.
- Unit test: **13/13 đạt**, gồm test hồi quy riêng cho góc ghost.
- Ảnh kiểm tra: `qa/ghost-angle-fixed.png`.

## Nguồn đối chiếu

Các ảnh chụp HOXCO Vision v1.7.0 do người dùng cung cấp: B1 tách hai mảnh, B2 kéo thả, B3 TCP, Camera, Vận hành, cấu hình trạm và hai nguồn Ảnh tĩnh/Robot.

## Kết quả vòng cuối

- Giao diện tối, mềm và cân đối hơn bằng card/button/tab bo góc; không sao chép logo HOXCO.
- B1 có vùng cuộn độc lập: viewport 692 px, nội dung 1329 px, cuộn được đến 100%.
- B1 có `Nền từ camera`, `Mở ảnh nền…`, ngưỡng Otsu tự động và gộp vùng khi che khuất.
- Độ nhạy trừ nền và ngưỡng sáng là hai tham số độc lập.
- B2 dùng canvas native; 1.000 bước drag đo được khoảng 63 ms trên máy kiểm thử.
- B2 không ghi dữ liệu drag giả từ test vào sản phẩm thật.
- B3 hiển thị ROI theo mask sạch, cắt sát vật và giữ đúng offset TCP.
- Vận hành có hai trạng thái nguồn riêng: `Ảnh tĩnh` và `Robot · DeltaX`.
- Ảnh Radxa thật phát hiện đủ type 0/1; overlay, tọa độ, góc và lệnh Robot hiển thị đúng.
- Camera Basler có Exposure/Gain/ROI/FPS/Trigger và `LatestImageOnly`.
- 12/12 automated tests passed.

## Ảnh kiểm tra

- `qa/ui-b1.png`
- `qa/ui-b1-bottom.png`
- `qa/ui-b2.png`
- `qa/ui-b3.png`
- `qa/ui-operation.png`
- `qa/ui-operation-robot.png`
- `qa/ui-camera.png`

## Ghi chú

Camera Basler vật lý chưa có trên máy kiểm thử, nên phần kết nối thiết bị thật vẫn cần chạy tại trạm có camera và pylon Runtime. Các trạng thái không-camera đã được chụp và đối chiếu ở kích thước cửa sổ 1680×960.
