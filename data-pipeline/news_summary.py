from model_wrapper import Model_Factory
from pathlib import Path
from tqdm import tqdm
import polars as pl
import time



'''
test_news = r"""
Lợi nhuận sau thuế thuộc về cổ đông công ty mẹ FPT đạt 2.455 tỷ đồng, tăng 21,6% so với cùng kỳ năm trước. EPS tương ứng đạt 1.933 đồng/cổ phiếu. Tập đoàn FPT (mã FPT) vừa công bố kết quả kinh doanh sơ bộ 4 tháng đầu năm 2024 với doanh thu ước đạt 18.989 tỷ đồng và lợi nhuận trước thuế 3.447 tỷ đồng, lần lượt tăng 20,6% và 19,7% so với cùng kỳ 2023. Lợi nhuận sau thuế 4 tháng đầu năm đạt 2.932 tỷ đồng, tăng 19,7% so với cùng kỳ 2023. Trong đó, lợi nhuận sau thuế thuộc về cổ đông công ty mẹ đạt 2.455 tỷ đồng, tăng 21,6% so với cùng kỳ năm trước. EPS tương ứng đạt 1.933 đồng/cổ phiếu.  Mảng Dịch vụ CNTT thị trường nước ngoài tiếp tục đà tăng trưởng ấn tượng, đạt doanh thu 9.450 tỷ đồng, tương đương với mức tăng 29,2%, dẫn dắt bởi sức tăng đến từ cả 4 thị trường. Trong đó, thị trường Nhật Bản và APAC tiếp tục giữ mức tăng trưởng cao, tăng lần lượt 34,3% (tương đương tăng trưởng 44,6% theo Yên Nhật) và 31,6%. Khối lượng đơn hàng ký mới tại thị trường nước ngoài đạt 13.940 tỷ đồng, tăng 12,8%, chủ yếu do Tập đoàn đã đẩy sớm việc ký mới ngay trong tháng 12/2023. Lũy kế 4 tháng đầu năm 2024, FPT thắng thầu 20 dự án lớn với quy mô trên 5 triệu USD/dự án. Tổng giá trị đơn hàng thắng thầu và đang trong giai đoạn xúc tiến ký kết tăng trưởng trên 30% so với cùng kỳ, cho thấy nhu cầu cho Dịch vụ CNTT ngày càng cao trên toàn cầu. Mảng Dịch vụ CNTT trong nước ghi nhận doanh thu đạt 2.005 tỷ đồng, tương đương mức tăng trưởng 8,6%.  Khối Dịch vụ Viễn thông duy trì mức tăng trưởng bền vững với doanh thu đạt 5.365 tỷ đồng và LNTT đạt 1.116 tỷ đồng, lần lượt tăng 6,1% và 13,2% so với cùng kỳ năm trước. Khối Giáo dục, đầu tư và khác cũng tiếp tục mức tăng trưởng doanh thu 41,7% lên 2.169 tỷ đồng trong 4 tháng đầu năm. Lợi nhuận trước thuế cũng tăng tương ứng 16,8% lên 784 tỷ đồng. Năm 2024, FPT đặt ra mục tiêu doanh thu 61.850 tỷ đồng (~2,5 tỷ USD) và lợi nhuận trước thuế 10.875 tỷ đồng, đều tăng khoảng 18% so với kết quả thực hiện năm 2023. Với kết quả đạt được sau 4 tháng đầu năm, tập đoàn đã thực hiện khoảng 31% kế hoạch doanh thu và lợi nhuận đề ra. Mới đây, Tập đoàn FPT cũng thông báo về việc triển khai các phương án phát hành cổ phiếu năm 2024. Theo đó, FPT dự kiến sẽ phát hành thêm 190,5 triệu cổ phiếu nhằm tăng vốn từ nguồn vốn chủ sở hữu với tỷ lệ 20:3 (cổ đông sở hữu 20 cổ phiếu sẽ nhận thêm 3 cổ phiếu mới). Nguồn vốn trích từ lợi nhuận sau thuế chưa phân phối tại ngày 31/12/2023 trên BCTC kiểm toán 2023. Sau phát hành, vốn điều lệ của FPT sẽ tăng từ 12.700 tỷ đồng lên 14.605 tỷ đồng. Thời gian thực hiện không muộn hơn quý 3/2024. Ngày chốt danh sách đăng ký cuối cùng sẽ cùng thời điểm chốt quyền nhận cổ tức còn lại của năm 2023. Theo kế hoạch, FPT sẽ chi khoảng 1.200 tỷ đồng để trả cổ tức bằng tiền mặt cho cổ đông với tỷ lệ 10% trong quý 2/2024.
    """
model = Model_Factory.create_model('ollama',
                           model_name="FinR1:Q8_0")                         
print(model.summarize(test_news))
exit()
'''


REPO_DIR = Path(__file__).parent.parent
DATA_DIR = REPO_DIR / "data"
NEWS_DIR = DATA_DIR / "02_news"
SUMMARY_DIR = DATA_DIR / "03_summary"

SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

#pl.Config.set_fmt_str_lengths(200)


def process_and_summarize_file(file_path, summary_func):
    try:
        symbol = file_path.stem.removesuffix('_news')
        df = pl.read_parquet(file_path)
        news_content = df["content"].to_list()
        
        summaries = []
        for content in tqdm(news_content, desc=f"Summarizing {symbol} news", unit="article"):
            try:
                result = summary_func(content)
                summaries.append(result)
            except Exception as e:
                print(f"\nError summarizing text for {symbol}: {e}")
                print(f"Skipping file {file_path.name}.")
                return 

        df_summarized = df.with_columns([
            pl.Series(name = "summary", values = summaries)
        ])
        
        summary_path = SUMMARY_DIR / f"{symbol}_summary.parquet"
        df_summarized.write_parquet(summary_path)
        print(f"Saved summary to {summary_path}")

    except Exception as e:
        print(f"Error processing file {file_path.name}: {e}")


def main():
    print("Loading model...")
    model = Model_Factory.create_model('ollama', model_name="FinR1:Q8_0")
    print(f"Model: {model.model_name} loaded successfully!")
    summary_func = model.summarize

    news_pattern = "*_news.parquet"
    for file in NEWS_DIR.glob(news_pattern):
        process_and_summarize_file(file, summary_func)

    
if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed = time.time() - start_time
    print(f"\nTotal execution time: {elapsed:.2f} seconds")    


'''
Summarizing FPT news: 100%|█████████████████████████████████████████████████████████████| 225/225 [45:33<00:00, 12.15s/article]
Total execution time: 2734.11 seconds
'''