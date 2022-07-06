from math import floor, ceil
import numpy as np

def round_number(num):
    c = ceil(num)
    f = floor(num)
    uni = np.random.uniform(0.0, 1.0)
    if uni < (num - f):
        return c
    else:
        return f


print(round_number(1.2))

# arr = np.random.randint(10, size=20)

# arr[0] = 1

# arr[6] = 1

# arr[11] = 1

# print(arr)

counts = np.array([1, 7, 1, 8, 7, 3, 1, 4, 7, 3, 1, 1, 7, 9, 2, 0, 1, 5, 9, 3])

print(counts)
s = set()
reduced_vocab_size = 0

for i, num in enumerate(counts):
    if 1 < num:
        s.add(i)
        reduced_vocab_size += 1

print(s)
print(reduced_vocab_size)

total_counts = 0
new_word_index = 0

for i, num in enumerate(counts):
    if 1 < num:
        counts[new_word_index] = counts[i] - 1
        total_counts += counts[new_word_index]
        new_word_index += 1

print(counts)

for i in range(reduced_vocab_size, len(counts)):
    counts[i] = 0

print(counts)

# unigram table
pow_counts = np.power(counts, 0.75)
z = np.sum(pow_counts)

max_size = 3000

nums = (max_size * pow_counts / z)
nums = np.vectorize(round_number)(nums)
print(nums)
sum_nums = np.sum(nums)
print(sum_nums)