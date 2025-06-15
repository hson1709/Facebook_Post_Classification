# Facebook Post Classification App

This project is a web application for classifying Facebook posts using two different models: **BiLSTM+CNN** (build from scratch with phoW2V embeddings) and **PhoBERT** (fine-tuned for Vietnamese). The app supports both single post classification and batch processing via CSV files.

## Features

- **Single Post Classification:** Classify individual Facebook posts.
- **Batch CSV Processing:** Upload a CSV file and classify posts in bulk.
- **Model Choices:**
  - **BiLSTM+CNN:** Uses pretrained phoW2V word embeddings and a hybrid BiLSTM+CNN architecture.
  - **PhoBERT:** Utilizes the PhoBERT model, fine-tuned for Vietnamese text classification.
- **User Interface:** Built with Gradio for easy interaction.
- **Supports CPU and GPU:** Automatically detects and uses available hardware.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/post-classification.git
cd post-classification
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the application

```bash
python app.py
```

The app will be available at [http://localhost:8080](http://localhost:8080).

## Docker

You can also run the app using Docker:

```bash
docker build -t post-classification-app .
docker run -p 8080:8080 post-classification-app
```


## Usage

- **Single Post:** Enter your Facebook post in the input box and select the model to classify.
- **Batch CSV:** Upload a CSV file, select the text column and model, and download the results.

## Technical Information

- **BiLSTM+CNN:** Uses pretrained phoW2V embeddings and a hybrid BiLSTM+CNN architecture.
- **PhoBERT:** Fine-tuned PhoBERT model for Vietnamese.
- **Batch Processing:** Supports CSV file input for bulk classification.
- **Hardware:** Compatible with both GPU and CPU.

## Acknowledgements

The BiLSTM+CNN model in this project utilizes pre-trained word embeddings from the PhoW2V model. PhoW2V is a Vietnamese word2vec model trained on a large corpus of Vietnamese text. It provides high-quality word embeddings, which help the BiLSTM+CNN architecture effectively capture semantic relationships in text data.

By using PhoW2V embeddings, our model benefits from the semantic richness encoded in these pre-trained vectors, which improves the model's performance on various natural language processing tasks, including text classification, sentiment analysis, and more. The embeddings are used as input features for the BiLSTM and CNN layers, allowing the model to leverage both sequential and spatial features of the text.

For more details on how PhoW2V was trained and how to use it, please refer to the official [PhoW2V repository](https://github.com/datquocnguyen/PhoW2V).
