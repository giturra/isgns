import numpy as np

#from collate import Collate

from utils import round_number
from vocab import Vocab


class UnigramTable:

    def __init__(self, max_size: int=100_000_000):
        self.max_size = max_size
        self.size = 0
        self.z = 0
        self.table = np.zeros(self.max_size)

    def sample(self) -> int:
        assert(0 < self.size)
        unigram_idx = self.table[np.random.randint(0, self.size)]
        return unigram_idx

    
    def build(self, vocab: Vocab, alpha: float):
        

        reserved_idxs = set(vocab.counter.keys())
        free_idxs = vocab.free_idxs
        # print(reserved_idxs | free_idxs)
        # print(counters.to_numpy(reserved_idxs | free_idxs))
        # print(counters)
        # print(numpy2dict(counters.to_numpy(reserved_idxs | free_idxs)))
        #print(f'vocab counter = {vocab.counter}')
        counts = vocab.counter.to_numpy(reserved_idxs | free_idxs)
        #print(f'counts = {counts}')
        vocab_size = len(counts)
        counts_pow = np.power(counts, alpha)
        z = np.sum(counts_pow)
        #print(f'z2 = {z}')
        
        
        nums = self.max_size * counts_pow / z
        #print(nums)
        nums = np.vectorize(round_number)(nums)
        sum_nums = np.sum(nums)

        # print(f'nums = {nums}')
        # print(f'sum_nums = {sum_nums}')
        # print(f'max_size {self.max_size}')

        while (self.max_size < sum_nums):
            w = int(np.random.randint(0, vocab_size))
            if 0 < nums[w]:
                nums[w] -= 1
                sum_nums -= 1
        # print(
        #     f'nums = {nums}'
        # )

        # todo: hacer el sampleo

        self.z = z
        self.size = 0

        # w = 0
        # while w < vocab_size:

        # i = 0
        # while i < len(nums):
        #     self.table[i:   ]
        # print(f'vocab_size = {vocab_size}')
        for w in range(vocab_size):
            #print(w)
            self.table[self.size: self.size + nums[w]] = w
            self.size += nums[w]
        # print(self.table)
        # print(self.size)

    def update(self, word_idx: int, F: float):
        
        assert(0 <= word_idx)
        assert(0.0 <= F)

        self.z += F
        if self.size < self.max_size:
            
            if F.is_integer():
                copies = min(int(F), self.max_size)
                self.table[self.size: self.size + copies] = word_idx
            else:
                copies = min(round_number(F), self.max_size)
                #print(copies)
                self.table[self.size: self.size + copies] = word_idx
        
            self.size += copies
        
        else:
            n = round_number((F / self.z) * self.max_size)
            #print(f'n = {n}')
            for _ in range(n):
                table_idx = np.random.randint(0, self.max_size)
                #print(f'table_idx ? {table_idx}')
                self.table[table_idx] = word_idx

# col = Collate(4)
# col.vocab.add('hello')
# col.total_counts += 1
# col.vocab.add('how')
# col.total_counts += 1
# col.vocab.add('you')
# col.total_counts += 1
# col.vocab.add('are')
# col.total_counts += 1
# print("###########################")
# print(col.vocab.word2idx)
# col.vocab.add('hello')
# col.total_counts += 1
# col.vocab.add("how")
# col.total_counts += 1
# print("###########################")
# print(col.vocab.word2idx)
# print(col.vocab.counter)
# print(col.total_counts)
# col.reduce_vocab()
# print("###########################")
# print(col.vocab.word2idx)
# print(col.vocab.counter)
# print(col.vocab.first_full)
# print(col.vocab.free_idxs)
# print(col.total_counts)
# print("###########################")
# col.vocab.add("#")
# col.total_counts += 1
# col.vocab.add("?")
# col.total_counts += 1
# print(col.vocab.word2idx)
# print(col.vocab.counter)
# print(col.vocab.first_full)
# print(col.vocab.free_idxs)
# print(col.total_counts)
# col.vocab.delete(2)
# ut = UnigramTable(10)

# ut.build(col.vocab, col.vocab.counter, col.alpha)

# ut.update(2, 3.4)
# ut.update(3, 4.3)
# ut.update(4, 3.8)
# print(ut.table)
# ut.update(5, 2.2)
# print(ut.table)