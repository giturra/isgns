from river.utils import VectorDict


class Vocab:

    def __init__(self, max_size: int=1_000_000):
        self.max_size = max_size
        self.size = 0

        self.word2idx = VectorDict()
        self.idx2word = VectorDict()

        self.free_idxs = set()

        self.counter = VectorDict()

        self.first_full = False

    def add(self, word: str) -> int:
        if word not in self.word2idx.keys() and not self.is_full():
            if not self.first_full:
                word_idx = self.size
            else:
                word_idx = self.free_idxs.pop()
            self.word2idx[word] = word_idx
            self.idx2word[word_idx] = word
            self.counter[word_idx] = 1
            self.size += 1

            if self.is_full():
                self.first_full = True
            return word_idx
       
        elif word in self.word2idx.keys():
            word_idx = self.word2idx[word]
            self.counter[word_idx] += 1
            return word_idx

    def add_tokens(self, tokens: list[str]):
        for token in tokens:
            self.add(token)
    
    def is_full(self) -> bool:
        return self.size == self.max_size
    
    def __len__(self) -> int:
        return len(self.word2idx)
    
    def __contains__(self, word: str) -> bool:
        return word in self.word2idx
    
    def __getitem__(self, word: str) -> int:
        if word in self.word2idx:
            word_idx = self.word2idx[word]
            return word_idx
        return -1
    
    def delete(self, idx: int):
        self.free_idxs.add(idx)
        word = self.idx2word[idx]
        del self.word2idx[word]
        del self.idx2word[idx]
        del self.counter[idx]
        self.size -= 1
        
        

# class Vocab:


#     def __init__(self, max_size):
#         self.max_size = max_size
#         self.current_size = 0
#         self.word2idx = dict()
    
#     def add(self, word):
#         if word not in self.word2idx and not self.is_full():
#             word_index = self.current_size
#             self.word2idx[word] = word_index
#             self.current_size += 1
#             return word_index

#         elif word in self.word2idx:
#             word_index = self.word2idx[word]
#             return word_index
#         else:
#             return -1
    
#     def add_tokens(self, tokens):
#         for token in tokens:
#             self.add(token)

#     def is_full(self):
#         return self.current_size == self.max_size
    
#     def is_empty(self):
#         return self.current_size == 0
    

#     def __len__(self):
#         return len(self.word2idx.keys())

                    
#     def __getitem__(self, word: str):
#         if word in self.word2idx:
#             word_index = self.word2idx[word] 
#             return word_index
#         return -1
    
#     def __contains__(self, word):
#         return word in self.word2idx.keys()