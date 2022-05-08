
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

working_directory = "C:/Users/57834/Documents/thesis/python/MNIST"
os.chdir(working_directory)

import torch
from torchvision import datasets
import MNIST.SuperpixelDataset as dataset
#import train as main_train
import MNIST.mnist_train as train



import MNIST.GnnStack as GnnStack
import MNIST.EquivariantGnnStack as EquivariantModel
import MNIST.Gcn as Gcn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    # just to download the dataset
    mnist_train = datasets.MNIST('../data', train=True, download=True)

    index_data = list(range(0, 5000))

    data = torch.utils.data.Subset(mnist_train, index_data)

    dataset_mnist = dataset.SuperpixelDataset(data)

    size_hiffen_features = 32
    input_size = max(dataset_mnist.num_node_features, 1)
    model = Gcn.GNNStack(input_size, size_hiffen_features,
                              dataset_mnist.num_classes)
    description_model = "testing with mnist"
    trained_model = train.train_graph(model, dataset_mnist, 400, "graph")

if __name__ == "__main__":
    try:
        main()
    except Exception as exception:

        print(exception)
        raise exception