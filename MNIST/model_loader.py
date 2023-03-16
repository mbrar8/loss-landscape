import os
import torch, torchvision
import MNIST.models.dense as dense
import MNIST.models.cnn as cnn
import MNIST.models.vit as vit


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
        'cnn_entropy'   : cnn.CNNentropy
        }



def load(model_name, model_file=None):
    (net, loss) = models[model_name]()

    if model_file:
        assert os.path.exists(model_file), model+file + "does not exist."
        # Some stuff

    net.eval()
    return (net, loss)

