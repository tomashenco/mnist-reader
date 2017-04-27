import numpy as np

from ann import ANN
from data_reader import TestDataset

data = TestDataset('test.csv')
network = ANN([2000, 1000, 500])
network.build(data)
labels = network.predict(data.images_train)


# save results
np.savetxt('submission.csv', np.c_[range(1, len(labels)+1), labels],
           delimiter=',', header='ImageId,Label', comments='', fmt='%d')
