from preprocessing import Collate

col = Collate(4, 10)
col.vocab.add('hello')
col.total_counts += 1
col.vocab.add('how')
col.total_counts += 1
col.vocab.add('you')
col.total_counts += 1
col.vocab.add('are')
col.total_counts += 1
print("###########################")
print(col.vocab.word2idx)
col.vocab.add('hello')
col.total_counts += 1
col.vocab.add("how")
col.total_counts += 1
print("###########################")
print(col.vocab.word2idx)
print(col.vocab.counter)
print(col.total_counts)
col.reduce_vocab()
print("###########################")
print(col.vocab.word2idx)
print(col.vocab.counter)
print(col.vocab.first_full)
print(col.vocab.free_idxs)
print(col.total_counts)
print("###########################")
col.vocab.add("#")
col.total_counts += 1
col.vocab.add("?")
col.total_counts += 1
print(col.vocab.word2idx)
print(col.vocab.counter)
print(col.vocab.first_full)
print(col.vocab.free_idxs)
print(col.total_counts)
col.vocab.delete(2)


col.unigram_table.build(col.vocab, col.alpha)

col.unigram_table.update(2, 3.4)
col.unigram_table.update(3, 4.3)
col.unigram_table.update(4, 3.8)
print(col.unigram_table.table)
col.unigram_table.update(5, 2.2)
print(col.unigram_table.table)
print(col.unigram_table.sample())
