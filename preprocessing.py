import torch
import numpy as np
from nltk import word_tokenize
from river.utils import dict2numpy

from vocab import Vocab
from unigram_table import UnigramTable


class Preprocessing:

    def __init__(self, 
            vocab_size: int=1_000_000, 
            unigram_table_size: int=100_000_000,
            window_size: int=5, 
            alpha: float=0.75,
            subsampling_threshold:float =1e-3,
            neg_samples_sum: int=5, 
            tokenizer=word_tokenize
        ):
        
        self.vocab_size = vocab_size
        self.vocab = Vocab(vocab_size)
        
        self.total_counts = 0

        self.alpha = alpha

        self.window_size = window_size

        self.subsampling_threshold = subsampling_threshold

        self.unigram_table_size = unigram_table_size
        self.unigram_table = UnigramTable(self.unigram_table_size)

        self.tokenizer = tokenizer

        self.neg_samples_sum = neg_samples_sum

    def reduce_vocab(self):
        self.vocab.counter = self.vocab.counter - 1
        for idx, count in list(self.vocab.counter.items()):
            if count == 0:
                self.vocab.delete(idx)
        self.total_counts = np.sum(dict2numpy(self.vocab.counter))

    def rebuild_unigram_table(self):
        self.unigram_table.build(self.vocab, self.alpha)
    
    def update_unigram_table(self, word: str):
        word_idx = self.vocab.add(word)
        self.total_counts += 1
        #print(f'{word_idx} - {self.vocab.counter[word_idx]}')
        F = np.power(self.vocab.counter[word_idx], self.alpha) - np.power((self.vocab.counter[word_idx] - 1), self.alpha)
        self.unigram_table.update(word_idx, F)

        if self.vocab_size == self.vocab.size:
            #print("wenas")
            self.reduce_vocab()
            self.rebuild_unigram_table()

    def __call__(self, batch):
        examples = []
        labels = []
        for tweet in batch:
            tokens = list(map(lambda s: s.lower(), self.tokenizer(tweet)))
            n = len(tokens)
            for target, token in enumerate(tokens):
                self.update_unigram_table(token)
                target_index = self.vocab[token]
                if target_index == -1:
                    #print("continue1")
                    continue
                random_window_size = np.random.randint(1, self.window_size + 1)
                #print(f'random_window_size = {random_window_size}')
                for offset in range(-random_window_size, random_window_size):
                    if offset == 0 or (target + offset) < 0:
                        #print("continue2")
                        continue
                    if (target + offset) == n:
                        break
                    word_context = tokens[target + offset]
                    context_idx = self.vocab[word_context]
                    #print(target_index, word_context)
                    if context_idx == -1:
                        #print("continue3")
                        continue
                        
                    if 0 < self.vocab.counter[context_idx] and np.sqrt(
                        self.subsampling_threshold * self.total_counts / self.vocab.counter[context_idx]
                    ) < np.random.uniform(0.0, 1.0): 
                        #print("continue4")
                        continue
                    neg_samples = np.zeros(self.neg_samples_sum, dtype=int)
                    for k in range(self.neg_samples_sum):
                        neg_samples[k] = self.unigram_table.sample()
                    data = [[target_index, context_idx]] + [[target_index, neg_sam] for neg_sam in neg_samples]
                    label = [1] + [0] * self.neg_samples_sum
                    examples.append(data)
                    labels.append(label)
                    
        return torch.tensor(examples), torch.tensor(labels)
            
         