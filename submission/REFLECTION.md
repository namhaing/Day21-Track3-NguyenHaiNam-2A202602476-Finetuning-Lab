# Reflection — Lab 21

*Ngắn gọn, thành thật. Phần này chấm theo độ cụ thể, không theo độ dài.*

**1. Điều gì làm bạn ngạc nhiên nhất?**

Kết quả đổi ngược chỉ vì số mẫu. Lần đầu tôi chạy thử với `EVAL_LIMIT=8` thì được PASSED, điểm kiến thức chung còn tăng. Chạy lại đủ 50 + 15 mẫu thì ra FAILED, điểm kiến thức chung giảm 0.269. Cùng adapter, cùng code, chỉ khác số câu để chấm. Tôi cũng ngạc nhiên vì model học được gần hết bài phân loại (0.970) nhưng vẫn sai đúng một cụm "Khi nào tiện" ở cả 6 ticket, dù cụm này có 30 lần trong dữ liệu train và lần nào cũng là `thap`.

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**

Không phải. Tôi tưởng sẽ mất nhiều thời gian ở phần train. Thực tế train trên T4 chỉ khoảng 7 phút mỗi run. Thời gian mất nhiều nhất là ở môi trường:
- Lần chạy Colab đầu tiên tôi không để ý runtime không có GPU (`GPU : NONE`). Model bị đẩy xuống ổ đĩa, một batch 4 câu chạy hơn một tiếng mà không báo lỗi gì.
- Trên máy cá nhân, tải model bị treo giữa chừng ở 939 MB mà không có thông báo.
- File zip kết quả thì không tải về được bằng `files.download` khi chạy Colab qua VS Code, phải đưa qua HuggingFace.

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**

Tôi từng nghĩ loss thấp hơn là model tốt hơn, và tăng rank là cách để model học tốt hơn. Trong lab:
- `attn_only` có loss thấp nhất nhưng chỉ hoà với `correct` trên bài thật, dù rank đã tăng từ 16 lên 283.
- Run sai learning rate có loss chỉ cao hơn khoảng 2.5 lần nhưng điểm bài thật là 0, không trả nổi JSON.

Tôi cũng từng nghĩ fine-tune xong mà điểm bài chính cao là thành công. Giờ tôi thấy phải đo thêm phần kiến thức chung: model của tôi làm tốt bài phân loại nhưng lại trả JSON cho cả câu hỏi kiến thức thông thường.

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**

Tôi dùng Claude Code để:
- đọc đề, rubric và code, rồi lập checklist các bước;
- chạy thử cả pipeline với model 0.8B trên laptop;
- đọc output Colab mà VS Code lưu lại để kiểm tra từng bước;
- soạn nháp report từ các file trong `results/`.

Nó giúp nhiều nhất ở chỗ phát hiện lỗi im lặng: nhận ra Colab đang chạy không có GPU, và nhận ra Git trên Windows đổi ký tự xuống dòng làm `verify.py` báo tập eval bị sửa.

Những chỗ nó sai hoặc đoán chưa đúng:
- Checklist ghi `supervised_fraction` thường khoảng 0.2–0.35, thực tế là 0.41.
- Nó dựa theo ghi chú F-30 của lab và nói template đóng sẵn khối `<think>` trong prompt. Với bản `unsloth` tôi dùng thì không đúng, `</think>` vẫn nằm trong phần được tính loss.
- Nó ước tính NB2 mất 17–23 phút, thực tế chỉ khoảng 5 phút.
- Nó bảo tải file bằng `files.download`, nhưng cách này không chạy khi dùng Colab qua VS Code.
- Lỗi hết bộ nhớ ở phần hoán đổi adapter của NB6, nó chỉ phát hiện ra sau khi chạy.

Tôi phải tự kiểm tra lại các con số trong report với file JSON chứ không tin hẳn bản nháp.

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**

Tôi sẽ viết thật kỹ prompt tốt nhất có thể và đo nó trên một bộ eval cố định, chưa train gì cả. Trong lab này chỉ riêng prompt tốt đã đạt 0.765 và nhanh hơn prompt đơn giản 3 lần. Nếu prompt đã đủ dùng thì không cần fine-tune. Cùng lúc đó tôi sẽ chuẩn bị luôn một bộ câu hỏi kiến thức chung để đo xem fine-tune có làm model "quên" không. Đó chính là chỗ model của tôi bị trượt lần này.
