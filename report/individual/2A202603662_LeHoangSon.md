# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603662
- **Họ và tên:** Lê Hoàng Sơn 
- **SEED** (in ở ô đầu notebook): 3402

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☑ |

## 2. Kết quả chính

| Cách tiếp cận | F1 (test) | F1 (SUB 40) | p95 (ms/tin) | USD / 1k tin |
| --- | --- | --- | --- | --- |
| 1. BoW + LogReg (tự code) | 0.872 | 0.919 | ≈ 0.08 | 0 |
| 2. BoW + MLP (deep learning) | 0.910 | 0.974 | ≈ 0.3 | 0 |
| 3a. Encoder (MiniLM) + kNN | 0.880 | 0.947 | ≈ 41 | 0 |
| 3b. Encoder (MiniLM) + LogReg | 0.904 | 0.919 | ≈ 35 | 0 |
| 3c. Encoder (MiniLM) + MLP head | 0.932 | 0.947 | ≈ 31 | 0 |
| 4a. LLM nhỏ zero-shot | – | 0.909 | ≈ 1 630 | ≈ 0.004 |
| 4b. LLM lớn zero-shot | – | 0.930 | ≈ 410 | ≈ 0.009 |
| 4c. LLM nhỏ few-shot (4 ví dụ) | – | 0.974 | ≈ 1 890 | ≈ 0.013 |

## 3. Dự đoán và thực tế

**(a) `emb_tricky` — Encoder + LogReg đúng nhiều hơn BoW trên 6 tin "ngụy trang"?**
- Mình đoán: *[predict("emb_tricky", "Encoder + LogReg đúng nhiều hơn BoW (ví dụ ~5/6 so với ~2-3/6). BoW nhìn từ rời rạc nên bị lừa bởi tin ham chứa 'free/won/prize/call' và bỏ sót tin spam né từ khoá; embedding hiểu nghĩa cả câu nên phân biệt được 'free for lunch' (rủ bạn) với 'complimentary holiday, reply YES' (lừa đảo). Có thể vẫn sai 1 tin khó.")]*
- Ra gì: BoW đúng 5/6, Encoder + LogReg cũng đúng 5/6 (hòa). Cả hai cùng sai tin `Last chance! Your account rewards expire tonight, confirm details to keep them`.
- Vì sao lệch: tin này không có từ khoá spam quen thuộc và nghĩa của nó giống các thông báo giao dịch hợp lệ. Mô hình chỉ train trên khoảng 3.6k SMS cũ nên thiếu mẫu kiểu phishing hiện đại. BoW cũng đúng tin `free for lunch` vì các từ `you, are, for, my` có trọng số ham mạnh, lấn át từ `free`. Với chỉ 6 tin, kết quả hòa không đủ để kết luận embedding không tốt hơn.

**(b) `llm_fewshot` — few-shot tăng F1 và chi phí bao nhiêu?**
- Mình đoán: *[predict("llm_fewshot", "F1 tăng nhẹ hoặc giữ nguyên (chênh <0.05, trong ngưỡng nhiễu của 40 tin) vì zero-shot của model nhỏ đã cao ~0.95; 4 ví dụ chỉ giúp model hiểu quy ước nhãn. Chi phí/tin tăng khoảng 2-2.5 lần vì prompt dài thêm ~70-80 từ ví dụ (token input), trong khi output vẫn ~1 token; độ trễ tăng nhẹ.")]*
- Ra gì: model nhỏ zero-shot F1 (SUB) 0.909; thêm 4 ví dụ lên 0.974 (+0.065), chi phí trên 40 tin từ $0.00017 lên $0.00053, gấp khoảng 3.1 lần.
- Vì sao lệch: zero-shot của model nhỏ lỗi toàn là ham bị đoán thành spam (tin nhắc đến "dealing", "money", "sell fast"). Các ví dụ mẫu có thể giúp model hiểu ranh giới của dataset, nên F1 tăng nhiều hơn mình nghĩ; chi phí tăng vì prompt dài hơn (token input). Với 40 tin, 1–2 tin đúng thêm đã đổi F1 vài điểm, nên đây là xu hướng chứ không phải kết luận chắc.

**(c) `tradeoff` — cách nào nằm trên đường Pareto?**
- Mình đoán: *[predict("tradeoff", "Trên đường Pareto: BoW+LogReg (nhanh nhất), một hàng Encoder (nhiều khả năng 3c, ~7 ms, F1 cao) và 4c (LLM few-shot, F1 cao nhất nhưng chậm). Bên phải vạch SLA 50 ms: cả 3 hàng LLM (p95 từ ~400 ms đến ~1.9 s); các hàng còn lại đều đạt SLA. Bị lấn át: 4a (gpt-oss-20b zero-shot) vì 4b vừa nhanh hơn vừa ít lỗi hơn, và có thể cả kNN (3a).")]*
- Ra gì: chỉ BoW + LogReg và BoW + MLP nằm trên đường Pareto. Encoder + MLP head (~31 ms, 1 − F1 ≈ 5.3%) và LLM nhỏ few-shot (~1.9 s, 1 − F1 ≈ 2.5%) đều bị BoW + MLP (~0.27 ms, 1 − F1 ≈ 2.5%) lấn át.
- Vì sao lệch: BoW + MLP có F1 trên SUB (0.974) ngang LLM few-shot nhưng nhanh hơn hàng nghìn lần. Tập SUB nhỏ và cân bằng (20 spam, 20 ham) nên F1 cao hơn F1 test toàn bộ (0.910), và một tin sai đổi 1 − F1 khoảng 2.5 điểm %, nên thứ hạng giữa các chấm gần nhau không đáng tin.

## 4. Phân tích 50 case khó

**Model tụt nhiều nhất:** BoW + LogReg, F1 từ 0.919 (test thường, SUB) xuống 0.333 trên 50 case khó (accuracy 52%); BoW + MLP từ 0.974 xuống 0.421 (accuracy 56%). Các model dùng embedding tụt khoảng 0.22–0.28, còn LLM gần như không tụt (4a: 0.909 → 0.903) và bản có prompt phòng thủ (4d) tăng lên 0.935 (accuracy 92%).
 
**Nhóm case khó nhất với BoW + LogReg:** spam giọng tự nhiên (đúng 10%), thao túng/lừa đảo (12%), né từ khoá (17%). Với BoW + MLP: spam giọng tự nhiên 10%, né từ khoá 0%, thao túng 38%. Ở hai nhóm ham (tưởng spam nhưng không phải, nói về spam) BoW lại đúng 90–100%, nhưng đó chủ yếu là vì mô hình thiên về đoán ham (train trên dữ liệu chỉ ~13% spam), chứ không vì nó hiểu ngữ cảnh.
 
**Hai case cụ thể (model tốt nhất của mình, 4d — LLM nhỏ + prompt phòng thủ, sai 4/50 case, đều là ham bị đoán thành spam):**
 
1. **#8** (nhóm "Tưởng spam nhưng không phải"), thật = ham: `Cash prize for the quiz night goes to the winning team, sign up with Sarah by Friday`. Cơ chế: tin chứa các từ khoá spam mạnh ("cash prize", "sign up") nên LLM bị kéo sang spam; prompt phòng thủ còn định nghĩa spam gồm "prize claims", làm ranh giới này dễ bị đẩy lệch với tin hoạt động cộng đồng.
2. **#47** (nhóm "Nói về spam"), thật = ham: `lol I just got a text saying I won £1000 and to call some 0906 number, people still fall for th…`. Cơ chế: người viết đang kể lại và cảnh báo về một tin lừa đảo, nhưng nội dung chứa đúng các dấu hiệu của spam ("won £1000", "call", số 0906), nên model chú ý vào từ khoá hơn ý định của người viết.

## 5. Thử thách cá nhân

**Đề:** Đổi ngưỡng quyết định 0.5 → 0.2 và → 0.8 cho mô hình Phần 1 (BoW + LogReg tự code). Báo precision/recall/F1 từng ngưỡng. Một ngân hàng sợ chặn nhầm tin OTP thì nên chọn ngưỡng nào? Vì sao?
 
**Cách làm:** giữ nguyên trọng số `w`, `b` đã train, tính `p = predict_proba(X_te, w, b)` một lần, rồi với mỗi ngưỡng `th` đặt nhãn spam nếu `p ≥ th` và tính `metrics(y_te, pred)` cùng số tin ham bị chặn nhầm (FP) và spam bỏ sót (FN) trên tập test.
 
**Kết quả:**
 
| Ngưỡng | Precision | Recall | F1 | FP | FN |
| --- | --- | --- | --- | --- | --- |
| 0.2 |  0.833 | 0.904 | 0.867 | 15 | 8 |
| 0.5 | 0.985 | 0.783 | 0.872 | 1 | 18 |
| 0.8 | 1.000 | 0.530 | 0.693 | 0 | 39 |
 
**Giải thích:** ngưỡng thấp làm mô hình dễ gắn nhãn spam hơn nên recall tăng nhưng precision giảm (chặn nhầm nhiều tin ham hơn); ngưỡng cao ngược lại, precision tăng còn recall giảm.*
 
**Ngân hàng sợ chặn nhầm OTP nên chọn ngưỡng cao (ví dụ 0.8).** OTP là tin ham hợp lệ; chặn nhầm nó (FP) gây hậu quả nặng vì khách không đăng nhập hay giao dịch được. Ngưỡng cao tăng precision, tức giảm số tin ham bị chặn nhầm, đổi lại recall giảm nên lọt thêm một ít spam. Ngân hàng chấp nhận đánh đổi này vì một OTP bị chặn nhầm tốn kém hơn một tin spam lọt qua, và spam lọt có thể xử lý thêm bằng các lớp khác. Lưu ý: tập test chỉ có 83 spam nên chênh lệch vài tin làm số đo dao động; về phương pháp, nên chọn ngưỡng trên tập val rồi báo cáo kết quả cuối trên test.

## 6. Phương án cho sản phẩm thật

**Yêu cầu:** 10 triệu SMS/ngày (≈ 116 tin/giây trung bình), p95 < 50 ms/tin.
 
**Số liệu dẫn chứng:**
- Độ trễ p95: BoW + LogReg ≈ 0.08 ms, BoW + MLP ≈ 0.3 ms, nhóm Encoder (MiniLM) ≈ 31–41 ms (sát vạch SLA hơn); cả ba hàng LLM vượt SLA (≈ 0.4–1.9 s).
- Chất lượng: trên test thường BoW + MLP rất tốt (F1 SUB 0.974), nhưng trên 50 case khó F1 chỉ 0.42 (BoW + MLP) và 0.33 (BoW + LogReg), encoder 0.64–0.72, còn LLM 0.90–0.935.
- Chi phí LLM (theo cột USD/1k tin của bảng, giá Groq): model nhỏ zero-shot ≈ $0.004/1k tin ≈ $42.5/ngày ≈ $1 275/tháng; model lớn ≈ $0.009/1k ≈ $87.5/ngày ≈ $2 625/tháng; model nhỏ few-shot ≈ $0.013/1k ≈ $132.5/ngày ≈ $3 975/tháng (10 triệu tin/ngày × 30 ngày).
- Thông lượng: LLM gọi tuần tự chỉ đạt khoảng 1.3–3.2 tin/giây, trong khi cần ≈ 116 tin/giây, nên phải gọi song song và chịu giới hạn tốc độ của nhà cung cấp.
**Kết luận:** không có cách đơn lẻ nào đạt cả độ trễ lẫn chất lượng: cách nhanh thì yếu trên case khó, cách mạnh trên case khó (LLM) thì chậm và đắt. Mình chọn **thiết kế lai (router)**:
- Tầng 1: model nhanh và rẻ (Encoder + MLP head hoặc BoW + MLP) cho xác suất p(spam). Nếu p < 0.3 thì kết luận ham, nếu p > 0.7 thì kết luận spam.
- Tầng 2: chỉ các tin lưỡng lự (p ∈ [0.3, 0.7]) mới gửi cho LLM với prompt phòng thủ (như `build_defended_prompt`). Thêm một lớp chặn sớm các cụm kiểu "ignore previous instructions" trước khi gọi LLM, và không để LLM là lớp duy nhất.
**Đánh đổi chấp nhận:** (1) một phần nhỏ tin (nhóm lưỡng lự) sẽ có độ trễ cao hơn SLA; (2) chi phí LLM giảm xấp xỉ theo tỉ lệ tin chuyển lên tầng 2, và tỉ lệ này mình chưa đo được trên dữ liệu thật nên cần đo trước khi triển khai; (3) bộ 50 case khó chỉ 50 tin viết tay, nên cần theo dõi thêm trên dữ liệu thật; (4) số p95 đo trên Colab (CPU dùng chung) dao động giữa các lần chạy.

## 7. AI log và điều đã học

| Phần | Mình hỏi AI gì (prompt) | AI trả lời/viết gì | Mình đã sửa / kiểm chứng gì |
| --- | --- | --- | --- |
| 1–3 | Nhờ viết/gợi ý code các TODO (`vectorize`, `sigmoid`, `train_logreg`, `metrics`, `build_mlp`, vòng lặp train, `cosine_similarity`, `top_k_similar`, `knn_predict`) | Code và giải thích từng dòng | *chạy các ô test tự kiểm, đọc lại để hiểu, sửa lỗi nếu có* |
| 4 | Nhờ viết `call_openai`, prompt, `parse_label`, `llm_classify_batch`; xử lý lỗi 400/429/503 với Gemini rồi chuyển sang Groq | Code, cách xử lý lỗi (làm gãy, giãn cách request, đổi provider) | *chạy test giả lập, chạy thật trên 40 tin, đối chiếu kết quả* |
| 5 | Nhờ nháp câu trả lời, đọc biểu đồ/bảng, viết thêm case, nháp báo cáo | Bản nháp | *đối chiếu với số liệu của mình, viết lại bằng lời mình* |
 
**Điều mình hiểu ra hôm nay mà trước đó chưa hiểu:** *điểm cao trên tập test cùng phân phối với train không bảo đảm gì với dữ liệu mới; accuracy dễ đánh lừa khi dữ liệu lệch lớp; chất lượng và độ trễ/chi phí luôn đánh đổi nhau nên cần chọn theo ràng buộc của sản phẩm.*
