import torch
import unidecode
import string
import random


all_characters = string.printable
n_characters = len(all_characters)

file = unidecode.unidecode(open('./text_files/alma.txt').read())
file_len = len(file)
print('file_len =', file_len)

chunk_len = 200


def random_chunk():
    start_index = random.randint(0, file_len - chunk_len)
    # start_index = 0
    end_index = start_index + chunk_len + 1
    return file[start_index:end_index]

print(random_chunk())

# Turn string into list of longs
def char_tensor(string):
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        tensor[c] = all_characters.index(string[c])
    return tensor

print(char_tensor('abcDEF'))

def onehot(tnsr):
    out = torch.zeros((tnsr.size(0), n_characters))
    for i in range(tnsr.size(0)):
        out[i][tnsr[i]] = 1
    return out

def random_training_set():
    chunk = random_chunk()
    inp = char_tensor(chunk[:-1])
    target = char_tensor(chunk[1:])
    # inp = onehot(inp)
    return inp, target
