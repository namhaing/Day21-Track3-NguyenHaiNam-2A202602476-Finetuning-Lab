# Lab 21 — Evaluation Report

**Họ tên**: Nguyễn Hải Nam  **MSSV**: 2A202602476  **Ngày**: 2026-10-07
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: Tesla T4 (Colab Free, 14.6 GB khả dụng, sm_75 → **fp16**)

> Mọi con số trong report lấy từ `results/` (nguồn ghi ở từng bảng). Chạy đầy đủ, `EVAL_LIMIT` để trống
> (`baselines_frozen.json`: `smoke_mode: false`, 50 target + 15 regression). `scripts/verify.py`: mọi kiểm
> tra liêm chính đều PASS (mask, ngân sách tham số, cùng số step, checksum eval, SHA prompt (b)).

---

## 1. Setup — lựa chọn và lý do

| | |
|---|---|
| Base model | `unsloth/Qwen3.5-4B` — model mặc định của tier T4 |
| Dataset | corpus mặc định: 250 ticket CSKH tiếng Việt → JSON triage 4 trường (`intent`, `urgency`, `product`, `sentiment`) |
| Train / val | 225 / 25 (seed 42) |
| Eval | 50 ticket target + 15 câu kiến thức phổ thông (regression), đóng băng ở NB2 |
| `max_length` | 1024 — p95 đo được là **98** token, gợi ý 256 *(results/token_stats.json)* |
| `MASK_MODE` | `assistant-only` |
| Huấn luyện | 2 epoch → **30** optimizer step · batch 1 × grad_accum 16 = effective batch **16** (< 32) · LR **1e-4** (= 10 × LR full-FT 1e-5) · cosine · warmup 3 step (10 %) · r=16, alpha=32 (2r) · fp16 + GradScaler |

**Vì sao chọn model và dataset này.**
* **Qwen3.5-4B** là model lớn nhất chạy được LoRA 16-bit trên GPU tôi có (Colab T4): peak VRAM đo được **8.78 GB / 14.6 GB** *(results/runs.csv)*. Đây cũng là model đã có số đo tham chiếu trong `docs/`, nên tôi đối chiếu được baseline của mình với số đã công bố ((b) = 0.765, khớp đúng).
* **Corpus mặc định** vì mọi nhóm điểm đều có thang khách quan (so trường JSON, keyword recall), không cần LLM judge. Với lần đầu chạy pipeline, tôi muốn kết quả so được với tham chiếu trước khi đổi bất kỳ biến nào.

**`max_length` = 1024 dù p95 = 98.** Tôi giữ giá trị của tier, có chủ đích. Với `per_device_batch=1` và `packing=False` không có padding giữa các mẫu, nên `max_length` chỉ là **trần cắt cụt**, không làm tăng compute hay VRAM. Mẫu dài nhất là 101 token (*token_stats.json*: max 101) nên không mẫu nào bị cắt. Đặt 256 sẽ cho kết quả y hệt; tôi giữ 1024 để không phải sửa cấu hình tier.

**Template có giữ khối `<think>` không?** **Có** — `"verdict": "reasoning preserved — safe to train on traces"` *(results/template_check.json)*. Chi tiết đáng chú ý: generation prompt của bản `unsloth` kết thúc ở `<think>\n\n` (chưa đóng), nên `</think>\n\n` nằm **trong** đoạn được tính loss (xem mục 2). Model học thêm việc tự đóng một khối think rỗng trước khi trả JSON. Corpus không có reasoning trace nên không cần xử lý thêm; lúc sinh, lab dùng `enable_thinking=False`.

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | **0.4149** (39/94 token) |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi KHÔNG nằm trong loss | `true` |

*(results/mask_proof.json)*

Đoạn được tính loss (`supervised_preview`):

```
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

Đoạn bị che (`masked_preview`) là toàn bộ `system` + `user` + `<|im_start|>assistant\n<think>\n\n`. Để đối chiếu, mode `everything` giám sát **94/94 (100 %)** token — cả system prompt lẫn ticket — tức model sẽ học cách viết lại câu hỏi.

**Vì sao không dùng `assistant_only_loss=True` của TRL.** Template Qwen3.5 không có marker `{% generation %}`. Tôi chạy `scripts/check_mask_agreement.py`: mask lấy từ tokenizer theo cách TRL dùng giám sát **0/31 token** (chỉ có warning), còn mask của labkit giám sát 11/31. Vì vậy NB3 huấn luyện trên chính dữ liệu đã pre-tokenize bằng mask được chứng minh ở trên: **9014/20951 token (43.0 %)** của 225 mẫu được giám sát (log NB3).

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.000 | 0.791 | 0.000 | 3228 |
| (b) base + optimized prompt | **0.765** | 0.791 | **1.000** | **1023** |
| (c) LoRA fine-tune | **0.970** | **0.522** | 1.000 | 1359 |

*(a), (b): results/baselines_frozen.json · (c): results/verdict.json — n = 50 target, 15 regression*

**(b) có thật sự mạnh hơn (a) không?** Có, rõ rệt: target 0.000 → 0.765, format 0.000 → 1.000, và **nhanh hơn 3.2 lần** (3228 → 1023 ms). Prompt ngây thơ “Phân loại ticket sau.” không nêu schema nên model trả văn xuôi tới trần 160 token: target và format đều bằng 0. Prompt (b) cho schema, đủ tập nhãn của từng trường và một ví dụ few-shot, nên model trả đúng một object JSON ngắn rồi dừng. Riêng prompt engineering đã mua được cả độ chính xác lẫn latency trước khi train.

**Có sửa `OPTIMIZED_PROMPT` không?** **Không.** `verify.py`: `baseline (b) prompt unmodified` PASS (SHA `719e74d3b6232053`).

Nhìn từng trường, (b) chưa hoàn hảo *(results/qualitative_b_vs_ft.json, sinh lại (b) cho trung bình đúng 0.765)*: `product` 1.00 · `sentiment` 0.88 · `urgency` 0.64 · `intent` 0.54. (b) thiên về `intent=hoan_tien` (26/50 lần) và `urgency=cao` (31/50 lần). Khoảng trống để fine-tune lấp nằm ở `intent` và `urgency`.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | biến thay đổi | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | format | VRAM GB | train s |
|---|---|---|---|---|---|---|---|---|---|---|
| `correct` | — | text-linear (12 module) | 16 | 32,464,896 | 1e-4 | 0.6259 | **0.970** | 1.000 | 8.78 | 400 |
| `attn_only` | **vị trí** | q,v (2 module) | **283** | 32,456,704 | 1e-4 | **0.5373** | **0.970** | 1.000 | 8.79 | 265 |
| `wrong_lr` | **learning rate** | text-linear | 16 | 32,464,896 | **1e-5** | 1.5702 | **0.000** | **0.000** | 8.78 | 401 |
| `qlora` | **độ chính xác base** (4-bit NF4) | text-linear | 16 | 32,464,896 | 1e-4 | 0.7058 | **0.940** | 1.000 | **3.86** | 455 |

*Train loss, trainable, VRAM, thời gian: results/runs.csv · target, format: results/autopsy.json. Cả 4 run `max_steps = 30`. “train loss” là `result.training_loss`, tức loss **trung bình** cả run, không phải loss bước cuối.*

Mỗi run chỉ đổi **một** biến so với `correct` (cột thứ hai). `attn_only` dùng `matched_rank()` → r=283 để giữ nguyên ngân sách: 32,456,704 so với 32,464,896 tham số, lệch **0.025 %** (< 5 %, `verify.py`: `attn_only is a FAIR contrast` PASS). `verify.py` cũng xác nhận cả 4 run chung **30 step**.

**Xếp hạng theo target (NB5 §4):** `correct` = `attn_only` (0.970) > `qlora` (0.940) ≫ `wrong_lr` (0.000).
**Xếp hạng theo train loss (NB4):** `attn_only` (0.537) < `correct` (0.626) < `qlora` (0.706) < `wrong_lr` (1.570).
Hai thứ tự **không trùng nhau** ở vị trí đầu: theo loss, `attn_only` là cấu hình tốt nhất; theo tác vụ, nó chỉ hoà.

**4.1 — `attn_only` thắng, thua hay hoà?**
Trên tập target, `attn_only` **hoà** `correct` (0.970 = 0.970, format đều 1.000), dù cùng ngân sách tham số. Thứ tự theo train loss thì khác: `attn_only` có loss thấp nhất (0.537 so với 0.626). Nếu chấm bằng loss, tôi sẽ kết luận sai rằng gắn adapter chỉ vào q,v là tốt hơn — đúng Lỗi #3 mà lab cảnh báo. Loss thấp hơn ở đây nhiều khả năng là ghi nhớ: rank 283 tập trung vào ít module, 225 mẫu rất đồng dạng. Về *rank* và *vị trí*: trên Qwen3.5 lai (`linear_attention: 24, full_attention: 8`), `q_proj/v_proj` chỉ có ở 8/32 lớp, nên nâng rank 16 → 283 (17.7 lần) không mua thêm được lớp nào. Ở tác vụ hẹp này và với 50 mẫu, tôi **không đo được** lợi thế của vị trí text-linear như deck khẳng định; kết luận trung thực là vị trí và rank *không* tạo khác biệt đo được trên target. Lợi thế thực tế của `attn_only` là chi phí: train nhanh hơn 34 % (265 so với 400 s) và sinh nhanh hơn 35 % (887 so với 1359 ms). Điều tôi rút ra: **rank không phải đòn bẩy**, vì tăng rank 17.7 lần không mang lại gì hơn.

**4.2 — `wrong_lr` chỉ khác đúng một con số.**
LR 1e-5 thay vì 1e-4 làm loss trung bình cao gấp **2.5 lần** (1.570 so với 0.626). Trong log, loss bước cuối của `wrong_lr` vẫn ~1.12, trong khi `correct` đã xuống ~0.03. Trên tác vụ, hậu quả còn nặng hơn loss gợi ý: **target 0.000 và format 0.000**, thấp hơn cả baseline (b) và ngang baseline (a) không train. Latency 5157 ms (gấp 3.8 lần `correct`) cho thấy model vẫn lan man tới trần token thay vì trả JSON. Sau 30 step, model chưa học nổi cả *định dạng* đầu ra. Nếu chỉ nhìn đường loss mà không biết LR, tôi sẽ kết luận “LoRA không học được bài này”, hoặc “cần rank cao hơn / thêm dữ liệu / dữ liệu quá khó”. Cả ba đều sai: nguyên nhân là thang LR. Trong ba nút vặn, **LR là nút phá hoại mạnh nhất** (−0.970 điểm target, so với 0 điểm của vị trí và −0.030 của 4-bit).

**4.3 — `qlora` tiết kiệm bao nhiêu, trả giá bằng gì?**
QLoRA giảm peak VRAM từ **8.78 xuống 3.86 GB (−56 %)**. Đổi lại:
* target giảm **0.030** (0.970 → 0.940);
* train chậm hơn **14 %** (455 so với 400 s);
* sinh chậm hơn **31 %** (1780 so với 1359 ms), do phải dequantize trọng số 4-bit ở mỗi bước.

Số đo **ủng hộ** khuyến nghị “không dùng QLoRA cho dòng Qwen3.5 khi không bắt buộc”. Trên T4, model 4B ở 16-bit đã vừa (8.78 GB), nên trả 3 điểm target và hơn 30 % latency để tiết kiệm VRAM mình không cần là không đáng. QLoRA chỉ có lý khi bộ nhớ là ràng buộc cứng, ví dụ nếu tôi muốn đẩy lên 9B trên cùng T4.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: **FAILED**
`target Δ = +0.205` · `regression Δ = −0.269` · `valid_trace_rate = 0.0` *(results/verdict.json)*

Lý do gate ghi: *“general capability regressed by 0.269 (tolerance 0.020)”*.

**Diễn giải.** Fine-tune thắng baseline (b) rõ ràng trên chính tác vụ được train: target 0.765 → 0.970 (+0.205), format giữ 1.000, và fine-tune không thua (b) ở bất kỳ ticket nào (mục 6). Lý do trượt là **quên thảm hoạ**: điểm kiến thức phổ thông rơi từ 0.791 xuống 0.522, gấp hơn 13 lần ngưỡng 0.020.

Cơ chế, theo tôi hiểu: 225 mẫu train đều cùng một dạng — system ngắn “Phân loại ticket sau.” + một ticket trần → một object JSON — không có mẫu nào khác loại. Hai epoch trên dữ liệu đồng nhất như vậy dạy model rằng *đầu vào nào cũng là ticket cần phân loại*. Khi nhận câu hỏi kiến thức (không có system prompt), model có xu hướng trả JSON triage thay vì trả lời. Tôi thấy đúng hiện tượng này ở mức nặng hơn khi chạy thử cùng pipeline với Qwen3.5-0.8B (mục 7): ở đó gần như mọi câu hỏi đều nhận về JSON.

Thứ tự chẩn đoán của NB5 khớp với số đo:
1. `format` = 1.000 → template và mask không có vấn đề.
2. `target` tăng mạnh → LR và cấu hình đúng.
3. **`regression` tụt** → quên thảm hoạ.

Cách sửa đúng là trộn 1–5 % dữ liệu hỏi–đáp phổ thông (replay, deck §6.3) vào tập train. Không được nới ngưỡng, không làm yếu prompt (b), không đổi tập eval.

**`valid_trace_rate = 0.0`** không phải dấu hiệu “mất khả năng suy luận”. Corpus không có reasoning trace và lúc sinh dùng `enable_thinking=False`, nên số này bằng 0 về mặt cấu trúc với mọi run.

**Kết luận của phán quyết:** bản fine-tune này **không nên deploy** như một model đa dụng. Nếu nó chỉ chạy sau một router chắc chắn rằng mọi input đều là ticket, rủi ro quên kiến thức không lộ ra, và fine-tune là lựa chọn tốt hơn prompt (b) (+0.205 target). Nhưng cổng hồi quy được thiết kế chính xác để chặn trường hợp “thắng việc chính, làm hỏng mọi việc khác”, và nó đã chặn đúng.

---

## 6. Định tính — có cả ca THUA

Nguồn: *results/qualitative_b_vs_ft.json*. Tôi sinh lại output của (b) cho cả 50 ticket bằng greedy decode; trung bình ra đúng 0.765 như NB2, nên đây là output (b) thật. Phân bố chênh lệch điểm FT − (b) theo ticket: **0.00 × 17 · +0.25 × 25 · +0.50 × 8**. Trên tập target, fine-tune **không thua (b) ở ticket nào**. Vì vậy các “ca thua” dưới đây là ca fine-tune **sai so với nhãn**. Nhóm fine-tune thua (b) thật sự là **regression** (0.791 → 0.522, mục 5).

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 6 | “…balo laptop… Đổi size. Hỏi cho biết thôi. Lần cuối mua ở đây.” | doi_tra · thap · tieu_cuc | **hoan_tien · cao** · tieu_cuc — 0.50 | doi_tra · thap · tieu_cuc — **1.00** | ✅ FT thắng: sửa cả intent lẫn urgency |
| 7 | “…máy xay sinh tố… Muốn đổi. Đã 3 ngày rồi. Bực mình.” | doi_tra · trung_binh · tieu_cuc | **van_chuyen · cao** · tieu_cuc — 0.50 | doi_tra · trung_binh · tieu_cuc — **1.00** | ✅ FT thắng |
| 2 | “…đèn bàn LED… Hoàn tiền. Quá hạn rồi. Cảm ơn shop nhiều.” | hoan_tien · cao · tich_cuc | đúng cả 4 — 1.00 | đúng cả 4 — 1.00 | hoà: nhãn trùng thiên kiến sẵn có của (b) |
| 3 | “…bình giữ nhiệt… Chưa thấy tiền. **Khi nào tiện.** Cảm ơn shop nhiều.” | hoan_tien · **thap** · tich_cuc | urgency **trung_binh** — 0.75 | urgency **trung_binh** — 0.75 | ❌ **FT sai** (cùng lỗi với (b)) |
| 39 | “…nồi chiên không dầu… Hoàn tiền. **Khi nào tiện.** Quá tệ.” | hoan_tien · **thap** · tieu_cuc | urgency **cao** — 0.75 | urgency **trung_binh** — 0.75 | ❌ **FT sai** |

*(product đúng ở mọi ô nên không ghi lại.)*

**Mẫu chung ở các ca FT sai.** Toàn bộ **6/50** ticket mà fine-tune không đạt điểm tuyệt đối (#3, 5, 12, 39, 41, 46) đều sai đúng **một** trường, `urgency`, và **cả 6 đều chứa cụm “Khi nào tiện”** (nhãn `thap`). Fine-tune đoán `trung_binh` trong cả 6 ca. Điều đáng chú ý: trong tập train, “Khi nào tiện” xuất hiện **30 lần, 100 % gán `thap`**, vậy mà sau 30 step model vẫn không học được ánh xạ này. Trong khi đó, cụm “Hỏi cho biết thôi” (22 lần, `thap`) thì được học đúng. Giả thuyết của tôi: nghĩa thông thường của “khi nào tiện” (lịch sự, không gấp nhưng vẫn chờ phản hồi) khiến prior của base kéo về mức giữa mạnh hơn tín hiệu nhãn. (b) cũng sai urgency ở cùng nhóm ticket này. Đây là giả thuyết, chưa kiểm chứng: cần train thêm step hoặc thêm ví dụ tương phản mới kết luận được.

---

## 7. Kết luận & điều tôi học được

**Kết luận.** Tôi **không** deploy bản fine-tune này làm model chung, dù nó giải bài triage gần như hoàn hảo (target 0.970 so với 0.765 của prompt tốt nhất). Lý do nằm trong chính dữ liệu: 225 mẫu cùng một dạng “ticket → JSON” dạy model một hành vi quá hẹp, và cái giá là 0.269 điểm kiến thức chung, gấp hơn 13 lần mức chịu được. Đòn bẩy thật sự trong lab này, xếp theo mức ảnh hưởng đo được:
1. **Learning rate.** Sai thang 10 lần là mất toàn bộ tác vụ: target 0.970 → 0.000, kể cả format.
2. **Thành phần dữ liệu.** Thiếu dữ liệu replay là nguyên nhân duy nhất khiến phán quyết FAILED.
3. **Độ chính xác 4-bit.** Mất 0.030 target, đổi lại tiết kiệm 56 % VRAM.
4. **Vị trí và rank** gần như không tạo khác biệt trên tác vụ hẹp này (0.970 = 0.970). Tăng rank 17.7 lần không mua thêm được gì.

Mask và template là điều kiện cần chứ không phải đòn bẩy: chúng đúng (format 1.000, mask được chứng minh), nhưng nếu sai thì mọi con số khác đều vô nghĩa. Vì vậy chúng được kiểm tra đầu tiên. Bước tiếp theo hợp lý nhất không phải đổi cấu hình LoRA mà là sửa dữ liệu: thêm 1–5 % câu hỏi phổ thông (tự viết, không trùng 15 câu eval), giữ nguyên mọi thứ khác, và đo lại cả bốn nhóm. Nếu regression hồi về trong ngưỡng mà target giữ được trên 0.765, fine-tune mới thật sự đáng ship.

**Ba điều tôi học được**:
1. **Số mẫu eval nhỏ có thể đảo ngược kết luận.** Lượt chạy thử với `EVAL_LIMIT=8` (8 target, 8 regression) báo **PASSED**, regression còn *tăng* (+0.125). Lượt đầy đủ 50 + 15 mẫu báo **FAILED** với regression −0.269. Cùng adapter, cùng code, chỉ khác số mẫu. Với 8 câu, mỗi câu nặng 0.125 điểm; một vài câu may mắn đủ để lật phán quyết. Tôi hiểu vì sao `verify.py` từ chối bản smoke.
2. **Train loss thấp nhất không phải cấu hình tốt nhất.** `attn_only` có loss thấp nhất (0.537) nhưng chỉ hoà trên tác vụ; `wrong_lr` có loss “chỉ” gấp 2.5 lần nhưng điểm tác vụ là 0. Độ chênh của loss không tỉ lệ với độ chênh của kết quả. Từ giờ tôi chỉ xếp hạng run bằng metric của tác vụ.
3. **Môi trường chạy là một phần của thí nghiệm.** Lần chạy Colab đầu tiên của tôi nằm trên runtime **không có GPU**: model 4B bị offload xuống đĩa, một batch 4 câu mất 3988 s, và notebook vẫn “đang chạy” bình thường. Trên máy Windows, Git tự đổi LF → CRLF làm `verify.py` báo tập eval “bị sửa” dù không ai đụng vào. Cả hai lỗi đều im lặng; chỉ đo trực tiếp (dòng `GPU :`, SHA từng file) mới phát hiện được.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:** (1) thêm ~10 câu hỏi–đáp phổ thông (≈4 %) vào tập train, chạy lại NB3 + NB4 + NB5 để kiểm chứng replay có kéo regression về ngưỡng không; (2) tăng lên 3 epoch hoặc thêm ví dụ tương phản cho cụm “Khi nào tiện” để kiểm tra giả thuyết ở mục 6; (3) quét rank r ∈ {8, 16, 64} ở vị trí text-linear (B4) để xác nhận rank không phải đòn bẩy trên tác vụ này.

---

## Phụ lục A — thưởng đã làm

- [x] **B1 NB6 merge + hot-swap.** Merge: target **0.970 → 0.970** (Δ 0.000, ngưỡng 0.01, n=50) *(results/merge_check.json)*. Hot-swap: 3 adapter `correct`, `attn_only`, `qlora` nạp trên **một** base, chuyển bằng `set_adapter()`; cả 3 trả JSON hợp lệ cho cùng ticket #0 *(results/hotswap_check.json)*. Ghi chú: §3 của NB6 lỗi `ValueError: We need an offload_dir` vì model đã merge ở §2 chưa được giải phóng (`del merged` không xoá được tham chiếu `model`), nên base mới bị đẩy một phần sang CPU. Tôi chạy §3 trong một tiến trình riêng. Trả lời câu hỏi B1: merge cho overhead suy luận bằng 0 nhưng mất tính mô-đun — một base không còn phục vụ được nhiều adapter, không A/B hay rollback nhanh được. Nên giữ adapter riêng khi phục vụ nhiều khách hàng hoặc tác vụ trên cùng một base.
- [ ] B2 dataset miền riêng
- [ ] B3 reasoning-trace collapse — không làm: corpus mặc định không có trace, nên `masked-think`/`response-only` cho mask giống hệt `assistant-only` (F-30).
- [ ] B4 quét rank có kiểm soát
- [x] **B5 HuggingFace Hub** — adapter `correct` công khai kèm model card (ghi rõ cả phán quyết FAILED và lỗi đã biết): **https://huggingface.co/namhai631/lab21-qwen35-4b-triage-vi-lora**

## Phụ lục B — thí nghiệm phụ (không thuộc `results/`, chỉ để đối chiếu)

Trước khi chạy trên Colab, tôi chạy thử toàn bộ pipeline trên laptop (RTX 3050 Ti 4 GB) với **Qwen3.5-0.8B** (tier `CPU` chạy trên GPU, 58 step, bf16), cùng corpus và cùng tập eval. Hiện tượng quên thảm hoạ xuất hiện ở mức nặng hơn nhiều: target 0.495 → **0.990** nhưng regression 0.611 → **0.067**; fine-tune trả JSON triage cho 14/15 câu kiến thức (“Ai là tác giả Truyện Kiều?” → `{"intent": "hoi_thong_tin", ...}`). Ở model nhỏ, `attn_only` *thua* `correct` (0.945 so với 0.990) còn ở 4B thì hoà. Model càng nhỏ, đòn bẩy vị trí càng rõ và cái giá của quên càng lớn. Số liệu và log của lượt này nằm ngoài `results/` vì là model khác; tôi chỉ dùng chúng để đối chiếu xu hướng, không dùng cho phán quyết.
