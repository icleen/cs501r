import json
import csv
import sys
from random import shuffle


with open(sys.argv[1], 'r') as f:
  reader = csv.reader(f, delimiter=',', quotechar='|')
  lines = [inst for inst in reader][1:]

images = {}
for line in lines:
  if line[0] not in images:
    images[line[0]] = line

keys = list(images.keys())
print(len(keys))
shuffle(keys)

t = int(0.9 * len(keys))
train = keys[:t]
val = keys[t:]

print(len(train), len(val))

with open("data/trainset.csv", 'w') as f:
  writer = csv.writer(f, delimiter=',', quotechar='|')
  for key in train:
    writer.writerow(images[key])

with open("data/valset.csv", 'w') as f:
  writer = csv.writer(f, delimiter=',', quotechar='|')
  for key in val:
    writer.writerow(images[key])
