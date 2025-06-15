from gensim.models import KeyedVectors
from gensim.models.keyedvectors import Vocab
import numpy as np
from tqdm import tqdm

def load_word2vec_with_tqdm(file_path):

    with open(file_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f) - 1  

    with open(file_path, 'r', encoding='utf-8') as f:
        header = f.readline()
        vocab_size, vector_size = map(int, header.strip().split())
        kv = KeyedVectors(vector_size=vector_size)

        vectors = np.zeros((vocab_size, vector_size), dtype=np.float32)
        index2word = []
        vocab = {}

        for i, line in enumerate(tqdm(f, total=total_lines, desc="Converting Word2Vec")):
            tokens = line.rstrip().split(" ")
            word = tokens[0]
            vec = np.array(tokens[1:], dtype=np.float32)

            index2word.append(word)
            vectors[i] = vec
            vocab[word] = Vocab(index=i, count=1)

        kv.add_vectors(index2word, vectors)
        return kv

word2vec_file = r"D:\Post Classification\word2vec_vi_words_300dims\word2vec_vi_words_300dims.txt"

kv_model = load_word2vec_with_tqdm(word2vec_file)

kv_path = word2vec_file + ".kv"
kv_model.save(kv_path)
print(f"\n Saved .kv model to: {kv_path}")
