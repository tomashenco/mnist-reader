from ann import ANN
from data_reader import Dataset

data = Dataset('train.csv')
network = ANN([2000, 1000, 500])
network.fit(data)
