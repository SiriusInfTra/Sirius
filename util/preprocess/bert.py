import numpy as np


vocab_size = 30522

token_len = 64
pesudo_text = np.random.randint(101, vocab_size - 1000, token_len) # in case out of bound
pesudo_text[0] = 101 # [CLS]
pesudo_text[-1] = 102 # [SEP]
pesudo_text[token_len // 2 - 1] = 102 # [SEP]

indexed_tokens = pesudo_text
segments_ids = [0 for _ in range(token_len // 2)] + [1 for _ in range(token_len // 2)]

input_ids = np.array(indexed_tokens).astype('int64')
attention_mask = np.array(segments_ids).astype('int64')

input_ids.tofile('data/bert/input-0.bin')
attention_mask.tofile('data/bert/mask-0.bin')