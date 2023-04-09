import os
import torch, torchvision
import MNIST.models.dense as dense
import MNIST.models.cnn as cnn
import MNIST.models.vit as vit
import cifar10.model_loader

models = {
        'dense_entropy' : dense.entropy,
        #'dense_sigent'  :
        #'dense_01'      : 
        #'dense_per'     :
        #'dense_log'     :
        #'dense_exp'     :
        #'dense_hinge'   :
        #'dense_ramp'    :
        #'dense_pinball' :
        #'dense_trupin'  :
        #'dense_rehinge' :
        'cnn_entropy'   : cnn.CNNentropy,
        'vit_entropy'   : vit.entropy,
        }



def load(model_name, model_file=None):
    (net, loss) = models[model_name]()

    if model_file:
        assert os.path.exists(model_file), model_name+model_file + "does not exist."
        # some stuff 

    net.eval()
    return (net, loss)

