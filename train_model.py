from ann import ANN
from data_reader import TrainDataset

data = TrainDataset('train.csv')
network = ANN([2000, 1000, 500])
network.build(data)
network.fit(data)
