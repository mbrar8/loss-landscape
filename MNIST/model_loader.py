import os
import torch, torchvision
import MNIST.models.dense as dense
import MNIST.models.cnn as cnn
import MNIST.models.vit as vit
import cifar10.model_loader

models = {
    'dense_entropy': dense.DenseEntropy(),
        'dense_nll': dense.Densenll(),
        'dense_kldiv': dense.Densekldiv(),
        'dense_gaussiannll': dense.Densegaussiannll(),
        'dense_poissonnll': dense.Densepoissoinnll(),
        'cnn_entropy'   : cnn.CNNentropy(),
        'cnn_nll'       : cnn.CNNnll(),
        'cnn_kldiv'     : cnn.CNNkldiv(),
        'cnn_gaussiannll': cnn.CNNgaussiannll(),
        'cnn_poissonnll': cnn.CNNpoissoinnll(),
        #'vit_entropy'   : vit.entropy(),
        }



def load(model_name, model_file=None):
    (net, loss) = models[model_name]()

    if model_file:
        assert os.path.exists(model_file), model_name+model_file + "does not exist."
        stored = torch.load(model_file, map_location=lambda storage, loc: storage)
    if 'state_dict' in stored.keys():
        net.load_state_dict(stored['state_dict'])
    else:
        net.load_state_dict(stored) 

    net.eval()
    return (net, loss)

