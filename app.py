import gradio as gr
from post_classification import PostClassificationApp

app = PostClassificationApp()

def create_interface():
    with gr.Blocks(title="Post Classification App", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🤖 Ứng dụng Phân loại bài đăng Facebook               
        Hỗ trợ xử lý post đơn lẻ và theo batch dưới dạng file csv. Ứng dụng hỗ trợ phân loại bài đăng bằng hai mô hình:
        - **BiLSTM+CNN**: Sử dụng pretrained word embeddings phoW2V và kiến trúc kết hợp BiLSTM và CNN
        - **PhoBERT**: Mô hình BERT được fine-tune cho tiếng Việt
        """)
        
        with gr.Tabs():
            # Tab 1: Single text prediction
            with gr.TabItem("📝 Phân loại văn bản đơn"):
                with gr.Row():
                    with gr.Column(scale=2):
                        model_choice_single = gr.Dropdown(
                            choices=["BiLSTM+CNN", "PhoBERT"],
                            label="Chọn mô hình",
                            value="PhoBERT"
                        )
                        
                        text_input = gr.Textbox(
                            label="Nhập văn bản cần phân loại",
                            placeholder="Ví dụ: Công nghệ ngày càng phát triển kéo theo nhiều vấn đề mới...",
                            lines=5
                        )
                        
                        predict_btn = gr.Button("Phân loại", variant="primary")
                        clear_btn = gr.Button("Xóa", variant="secondary")
                    
                    with gr.Column(scale=2):
                        result_output = gr.Markdown(label="Kết quả")

                gr.Examples(
                    examples=[
                        ["Trong bối cảnh nền kinh tế toàn cầu ngày càng biến động, đầu tư thông minh trở thành một trong những cách hiệu quả nhất để gia tăng tài sản. Đây không chỉ là việc dành ra một phần thu nhập để để dành mà còn là cách tối ưu hóa tiền bạc, giúp nó sinh lời một cách bền vững.", "PhoBERT"],
                        ["Hiện nay, các thành phố lớn và khu vực ven đô đang trở thành điểm nóng trên thị trường bất động sản. Sự phát triển hạ tầng giao thông, như đường cao tốc, tàu điện ngầm, và các khu công nghiệp, tạo động lực lớn cho giá trị bất động sản gia tăng. Những khu vực có tiềm năng phát triển cao thường thu hút nhiều nhà đầu tư nhờ khả năng sinh lời vượt trội trong trung và dài hạn.", "BiLSTM+CNN"],
                        ["Bước đầu tiên để tiến tới tự do tài chính là hiểu rõ tình hình tài chính cá nhân. Điều này bao gồm việc theo dõi thu nhập, chi tiêu, và tài sản hiện có. Một kế hoạch ngân sách chặt chẽ sẽ giúp bạn kiểm soát dòng tiền, đồng thời xác định được khoản tiết kiệm cần thiết để đạt được các mục tiêu trong tương lai.", "PhoBERT"],
                        ["Văn hóa ẩm thực của mỗi quốc gia là sự kết hợp độc đáo giữa các nguyên liệu và phương pháp chế biến, tạo nên những món ăn đặc trưng riêng biệt. Trong đó, đồ ăn không chỉ là sự kết hợp giữa các thực phẩm mà còn là một phần quan trọng trong việc duy trì sức khỏe, làm đẹp và thể hiện sự sáng tạo của người nấu. Ở mỗi quốc gia, đồ ăn và đồ uống đều có những đặc trưng riêng biệt", "BiLSTM+CNN"]
                    ],
                    inputs=[text_input, model_choice_single],
                    label="Ví dụ"
                )
            
            # Tab 2: CSV file processing
            with gr.TabItem("📊 Xử lý file CSV"):
                with gr.Row():
                    with gr.Column(scale=1):
                        model_choice_csv = gr.Dropdown(
                            choices=["BiLSTM+CNN", "PhoBERT"],
                            label="Chọn mô hình",
                            value="PhoBERT"
                        )
                        
                        csv_file = gr.File(
                            label="Upload file CSV",
                            file_types=[".csv"],
                            type="filepath"
                        )
                        
                        text_column = gr.Textbox(
                            label="Tên cột chứa văn bản",
                            placeholder="Ví dụ: text, content, description",
                            value="text"
                        )
                        
                        process_btn = gr.Button("Xử lý file", variant="primary")
                    
                    with gr.Column(scale=2):
                        csv_result = gr.Markdown(label="Kết quả xử lý")
                        
                        download_file = gr.File(
                            label="Tải xuống kết quả",
                            visible=False
                        )
                
                gr.Markdown("""
                ### 📋 Hướng dẫn sử dụng:
                1. **Chọn mô hình** bạn muốn sử dụng
                2. **Upload file CSV** chứa dữ liệu văn bản
                3. **Nhập tên cột** chứa văn bản cần phân loại
                4. **Nhấn "Xử lý file"** để bắt đầu phân loại
                5. **Tải xuống kết quả** sau khi xử lý xong
                
                ⚠️ **Lưu ý:** File CSV phải có header và được mã hóa UTF-8
                """)
        
        # Event handlers
        predict_btn.click(
            fn=app.predict_single_text,
            inputs=[text_input, model_choice_single],
            outputs=[result_output, gr.Textbox(visible=False)]
        )
        
        clear_btn.click(
            fn=lambda: ("", ""),
            outputs=[text_input, result_output]
        )
        
        def process_and_show_download(file, model, column):
            result, output_file = app.process_csv_file(file, model, column)
            if output_file:
                return result, gr.File(value=output_file, visible=True)
            else:
                return result, gr.File(visible=False)
        
        process_btn.click(
            fn=process_and_show_download,
            inputs=[csv_file, model_choice_csv, text_column],
            outputs=[csv_result, download_file]
        )

        gr.Markdown("""
        ---
        
        🔧 **Thông tin kỹ thuật:**
        - **BiLSTM+CNN**: Sử dụng pretrained word embeddings phoW2V và kiến trúc kết hợp BiLSTM và CNN
        - **PhoBERT**: Mô hình BERT được fine-tune cho tiếng Việt
        - Hỗ trợ xử lý batch cho file CSV
        - Tương thích với GPU và CPU
        """)
    
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="localhost",
        server_port=8080,
        debug=True
    )