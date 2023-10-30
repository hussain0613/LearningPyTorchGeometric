from torch_geometric.datasets import ShapeNet

# download the ShapeNet dataset
# dataset_shapenet = ShapeNet(root='D:/learning_pytorch/learning_pyg/ShapeNet/', categories=['Airplane']) # downloads the Airplane category of the ShapeNet dataset to here, a collection of 2690 graphs (Data objects)
# dataset_shapenet

# applying transforms to the dataset
import torch_geometric.transforms as T
dataset_shapenet_transformed = ShapeNet(root='ShapeNet_Transformed/', categories=['Airplane'], pre_transform=T.KNNGraph(k=6)) # creates a k-NN graph for each graph in the dataset