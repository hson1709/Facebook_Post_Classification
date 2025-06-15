from gensim.models import KeyedVectors

kv_path = r"D:\Post Classification\word2vec_vi_words_300dims\word2vec_vi_words_300dims.txt.kv"
embed_lookup = KeyedVectors.load(kv_path, mmap='r')
pretrained_words = embed_lookup.index_to_key
pretrained_words = pretrained_words[2:]

MODEL_NAME = "vinai/phobert-base"
EMBEDING_DIM = len(embed_lookup[pretrained_words[0]])
VOCAB_SIZE = len(pretrained_words)
EMBED_LOOKUP = embed_lookup
NUM_CLASSES = 23


print(f"Model name        : {MODEL_NAME}")
print(f"Embedding dim     : {EMBEDING_DIM}")
print(f"Vocabulary size   : {VOCAB_SIZE}")
print(f"Number of classes : {NUM_CLASSES}")
print(f"Example word      : {pretrained_words[0]}")
print(f"Vector (first 5)  : {embed_lookup[pretrained_words[0]][:5]}")
