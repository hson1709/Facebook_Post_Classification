import pickle
import torch
import torch.nn.functional as F
import re
import nltk
from transformers import AutoTokenizer
import numpy as np
import pandas as pd
import os
from model.BiLSTMwithCNN import BiLSTMwithCNN
from model.phobert_classifier import PhoBERTClassifier
from model.model_config import MODEL_NAME, NUM_CLASSES, EMBED_LOOKUP, EMBEDING_DIM, VOCAB_SIZE

nltk.download('punkt')
nltk.download('punkt_tab')



class PostClassificationApp:
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.embed_lookup = None
        self.label_mappings = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.load_models()
    

    def load_models(self):
        try:
            self.models['BiLSTM+CNN'] = BiLSTMwithCNN(VOCAB_SIZE, EMBEDING_DIM, NUM_CLASSES, EMBED_LOOKUP)
            self.models['BiLSTM+CNN'].load_state_dict(
                torch.load(r'D:\Post Classification\model\BiLSTMwithCNN.pth', map_location=self.device)
            )
            self.models['BiLSTM+CNN'].to(self.device)
            self.models['BiLSTM+CNN'].eval()
            self.embed_lookup = EMBED_LOOKUP

            self.models['PhoBERT'] = PhoBERTClassifier(MODEL_NAME, NUM_CLASSES)
            checkpoint = torch.load(r'D:\Post Classification\model\phobert_classifier.pth', map_location=self.device)
            self.models['PhoBERT'].load_state_dict(checkpoint['model_state_dict'])
            self.models['PhoBERT'].to(self.device)
            self.models['PhoBERT'].eval()

            self.tokenizers['PhoBERT'] = AutoTokenizer.from_pretrained('vinai/phobert-base')

            with open(r'D:\Post Classification\label\label_mappings.pkl', 'rb') as f:
                self.label_mappings = pickle.load(f)

        except FileNotFoundError as e:
            print(f" Model file not found: {e}")
            self.create_dummy_models()
        except Exception as e:
            print(f" Unexpected error during model loading: {e}")

    
    def create_dummy_models(self):
        label_file = r'D:\Post Classification\label\label_mappings.pkl'

        if os.path.exists(label_file):
            print(f" Đang tải label mappings từ '{label_file}'")
            with open(label_file, 'rb') as f:
                self.label_mappings = pickle.load(f)
        else:
            raise FileNotFoundError(f" Không tìm thấy file '{label_file}'.")

        self.models = {
            'BiLSTM+CNN': None,
            'PhoBERT': None
        }
    
    def preprocess_text(self, text):
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'[^\w\s]|_', '', text)
        text = re.sub(r'\d+', '', text)
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)
        text = text.lower()
        return text
    
    def tokenize_text_bilstm(self, text):
            
        tokens = nltk.word_tokenize(text)
        text_indices = []
        for word in tokens:
            try:
                word_idx = self.embed_lookup.key_to_index[word]
            except (KeyError, AttributeError):
                word_idx = 0
            text_indices.append(word_idx)
        return text_indices
    
    def pad_features(self, reviews_ints, sequence_length=270):
        features = np.zeros((len(reviews_ints), sequence_length), dtype=int)
        for i, row in enumerate(reviews_ints):
            features[i, -len(row):] = np.array(row)[:sequence_length]
        return features
    
    def predict_bilstm_cnn(self, text):
        try:
            if 'BiLSTM+CNN' not in self.models:
                return "Mô hình BiLSTM+CNN chưa được tải", 0.0
            
            model = self.models['BiLSTM+CNN']
            model.eval()
            
            preprocessed_text = self.preprocess_text(text)
            text_indices = self.tokenize_text_bilstm(preprocessed_text)
            features = self.pad_features([text_indices], sequence_length=270)
            
            feature_tensor = torch.from_numpy(features).to(torch.long).to(self.device)
            
            with torch.no_grad():
                output = model(feature_tensor)
                output_prob = F.softmax(output, dim=1)
                _, predicted_class = torch.max(output_prob, dim=1)

            id2label = self.label_mappings.get("id2label", {})
            predicted_label = id2label.get(predicted_class.item(), f"Class_{predicted_class.item()}")
            confidence = output_prob[0][predicted_class.item()].item()
            
            return predicted_label, confidence
            
        except Exception as e:
            return f"Lỗi dự đoán: {str(e)}", 0.0
    
    def predict_phobert(self, text):
        try:
            if 'PhoBERT' not in self.models:
                return "Mô hình PhoBERT chưa được tải", 0.0

            model = self.models['PhoBERT']
            tokenizer = self.tokenizers.get('PhoBERT')

            if tokenizer is None:
                return "Tokenizer PhoBERT chưa được tải", 0.0

            model.eval()
            preprocessed_text = self.preprocess_text(text)

            encoding = tokenizer(
                preprocessed_text,
                truncation=True,
                padding='max_length',
                max_length=256,
                return_tensors='pt'
            )

            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                probabilities = torch.nn.functional.softmax(outputs, dim=-1)
                predicted_class_id = torch.argmax(probabilities, dim=-1).item()
                confidence = probabilities[0][predicted_class_id].item()

            id2label = self.label_mappings.get("id2label", {})
            predicted_label = id2label.get(predicted_class_id, f"Class_{predicted_class_id}")

            return predicted_label, confidence

        except Exception as e:
            return f"Lỗi dự đoán: {str(e)}", 0.0
    
    def predict_single_text(self, text, model_choice):
        if not text.strip():
            return "Vui lòng nhập văn bản cần phân loại", ""
        
        if model_choice == "BiLSTM+CNN":
            label, confidence = self.predict_bilstm_cnn(text)
        elif model_choice == "PhoBERT":
            label, confidence = self.predict_phobert(text)
        else:
            return "Vui lòng chọn mô hình", ""
        
        result = f"**Kết quả phân loại:**\n\n"
        result += f"📝 **Văn bản:** {text[:100]}{'...' if len(text) > 100 else ''}\n\n"
        result += f"🤖 **Mô hình:** {model_choice}\n\n"
        result += f"🏷️ **Nhãn dự đoán:** {label}\n\n"
        result += f"📊 **Độ tin cậy:** {confidence:.4f} ({confidence*100:.2f}%)\n\n"

        if confidence > 0.8:
            result += "✅ **Độ tin cậy:** Rất cao"
        elif confidence > 0.6:
            result += "⚠️ **Độ tin cậy:** Khá tốt"
        else:
            result += "❌ **Độ tin cậy:** Thấp"
        
        return result, ""
    
    def process_csv_file(self, file, model_choice, text_column):
        if file is None:
            return "Vui lòng upload file CSV", None
        
        if not text_column.strip():
            return "Vui lòng nhập tên cột chứa văn bản", None
        
        try:
            df = pd.read_csv(file.name, encoding='utf-8')
            
            na_text = 0
            
            if text_column not in df.columns:
                return f"Không tìm thấy cột '{text_column}' trong file CSV. Các cột có sẵn: {', '.join(df.columns)}", None

            predictions = []
            confidences = []

            label2id = self.label_mappings.get("label2id",{})
            label_count = {label: 0 for label in label2id.keys()}
            
            for text in df[text_column]:
                if pd.isna(text) or not str(text).strip():
                    predictions.append("N/A")
                    confidences.append(0.0)
                    na_text += 1
                    continue

                preprocessed_text = self.preprocess_text(str(text))
                
                if model_choice == "BiLSTM+CNN":
                    label, confidence = self.predict_bilstm_cnn(preprocessed_text)
                    label_count[label] += 1
                elif model_choice == "PhoBERT":
                    label, confidence = self.predict_phobert(preprocessed_text)
                    label_count[label] += 1
                predictions.append(label)
                confidences.append(confidence)

            df['predicted_label'] = predictions
            df['confidence'] = confidences
            df['confidence_percent'] = [f"{c*100:.2f}%" for c in confidences]

            output_file = "classification_results.csv"
            df.to_csv(output_file, index=False, encoding='utf-8')

            total_texts = len(df)
            total_labels = sum(label_count.values())
            valid_label = [label for label in label_count.keys() if label_count[label] > 0]
            label_percentage_text = ""

            for label in valid_label:
                percentage = label_count[label]/total_labels*100
                label_percentage_text += f"- {label}: {percentage:.2f}%\n"

            high_confidence = sum(1 for c in confidences if c > 0.8)
            medium_confidence = sum(1 for c in confidences if 0.6 < c <= 0.8)
            low_confidence = sum(1 for c in confidences if 0 < c <= 0.6)

            
            
            summary = f"""
**Kết quả xử lý file CSV:**

📊 **Tổng quan:**
- Tổng số văn bản: {total_texts}
- Tổng số văn bản trống (lỗi): {na_text}
- Mô hình sử dụng: {model_choice}
- Cột văn bản: {text_column}
- Tổng số nhãn dự đoán : {total_labels}

📈 **Phân bố độ tin cậy:**
- Độ tin cậy cao (>80%): {high_confidence} ({high_confidence/total_texts*100:.2f}%)
- Độ tin cậy trung bình (60-80%): {medium_confidence} ({medium_confidence/total_texts*100:.2f}%)
- Độ tin cậy thấp (<60%): {low_confidence} ({low_confidence/total_texts*100:.2f}%)

📈 **Phân bố loại bài đăng:**
{label_percentage_text}

✅ **File kết quả đã được lưu:** classification_results.csv
            """
            
            return summary, output_file
            
        except Exception as e:
            return f"Lỗi xử lý file: {str(e)}", None