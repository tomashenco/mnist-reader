from ann import ANN
from data_reader import TestDataset

data = TestDataset('test.csv')
network = ANN([2000, 1000, 500])
network.build(data)
print network.predict(data.images_train)
