import six.moves.cPickle as Pickle
import os

datasetA_dir = '/home/suka/다운로드/cezanne2photo/trainA'
datasetB_dir = '/home/suka/다운로드/cezanne2photo/trainB'


datasetA = []
datasetB = []
print("start")

for filename in os.listdir(datasetA_dir):
    if filename.endswith('.jpg'):
        datasetA.append(filename)

for filename in os.listdir(datasetB_dir):
    if filename.endswith('.jpg'):
        datasetB.append(filename)

with open('cezanneA.pkl', 'wb') as tableA:
    Pickle.dump(datasetA, tableA)
with open('cezanneB.pkl', 'wb') as tableB:
    Pickle.dump(datasetB, tableB)

print('done')