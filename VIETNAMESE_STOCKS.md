# Hướng dẫn sử dụng FinMem cho cổ phiếu Việt Nam

Tài liệu này mô tả cách sử dụng FinMem-LLM-StockTrading cho cổ phiếu Việt Nam, cụ thể là FPT.

## Cấu trúc dữ liệu

Dự án đã được mở rộng để hỗ trợ cổ phiếu Việt Nam với cấu trúc dữ liệu sau:
- Price data: `data/01_price/FPT_price.parquet`
- News data: `data/02_news/FPT_news.parquet`
- Summary data: `data/03_summary/FPT_summary.parquet`
- Financial filings: `data/04_financial_filings/FPT_filing_k.pkl` và `FPT_filing_q.pkl`
- Sentiment analysis: `data/06_sentiment_analysis/sentiment_FPT_finbert.pkl`

## Quy trình xử lý dữ liệu

1. **Tải dữ liệu giá và tin tức**: Sử dụng các script trong thư mục `data-pipeline` để tải dữ liệu giá và tin tức cho cổ phiếu FPT.

2. **Tạo tóm tắt tin tức**: Tạo tóm tắt tin tức và lưu vào `data/03_summary/FPT_summary.parquet`.

3. **Phân tích sentiment**: Chạy script `get_sentiment_by_ticker.py` để phân tích sentiment cho tin tức FPT.

4. **Chạy pipeline dữ liệu**: Chạy script `data_pipeline.py` để tạo file `env_data.pkl` chứa dữ liệu cho mô hình.

## Chạy mô hình

1. Cài đặt file cấu hình cho FPT trong `config/fpt_gpt_config.toml`.

2. Chạy mô hình bằng lệnh:
   ```
   run_fpt_gpt.bat
   ```
   
3. Kết quả sẽ được lưu trong `data/09_results/fpt_gpt_results`.

## Phân tích kết quả

1. Trực quan hóa kết quả:
   ```
   python data-pipeline/visualize_fpt_results.py
   ```

2. Tính toán các chỉ số hiệu suất:
   ```
   python data-pipeline/calculate_fpt_metrics.py
   ```

## Lưu ý

- Đảm bảo đã cấu hình đúng API key trong file `.env`.
- Điều chỉnh tham số trong file cấu hình `fpt_gpt_config.toml` để tối ưu hiệu suất mô hình.
- Có thể sử dụng các mô hình khác như TGI hoặc Gemini bằng cách tạo file cấu hình tương ứng.
