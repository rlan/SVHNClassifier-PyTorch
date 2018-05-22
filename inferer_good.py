from __future__ import division

import torch.utils.data
from torch.autograd import Variable
import torch.nn
from dataset import Dataset
import torchvision.datasets
import numpy as np



class Inferer(object):
    def __init__(self, path_to_lmdb_dir):
        self._loader = torch.utils.data.DataLoader(Dataset(path_to_lmdb_dir), batch_size=1, shuffle=False)
        # https://pytorch.org/docs/master/torchvision/datasets.html#imagefolder
        # class torchvision.datasets.ImageFolder(root, transform=None, target_transform=None, loader=<function default_loader>)

    def infer(self, model):
        model.eval()
        num_correct = 0
        #needs_include_length = False

        for batch_idx, (images, length_labels, digits_labels) in enumerate(self._loader):
        
            print(type(images))
            print(type(length_labels))
            print(type(digits_labels))
            images, length_labels, digits_labels = (Variable(images.cuda(), requires_grad=True),
                                                    Variable(length_labels.cuda(), requires_grad=True),
                                                    [Variable(digit_labels.cuda(), requires_grad=True) for digit_labels in digits_labels])
            print("length_labels", length_labels)
            print("digits_labels", digits_labels)
            with torch.no_grad():
                print("images.size()", images.size())
                length_logits, digits_logits = model(images)
                #print("length_logits", length_logits)
                #print("digits_logits", digits_logits)
                # See paper appendix for the math
                lsm = torch.nn.LogSoftmax(1)
                length_logsoftmax = lsm(length_logits)
                #print("length_logsoftmax", length_logsoftmax)

                digits_logsoftmax = [lsm(digit_logits) for digit_logits in digits_logits]
                digits_max = [x.max() for x in digits_logsoftmax]
                digits_argmax = [torch.argmax(x) for x in digits_logsoftmax]
                #print("digits_logsoftmax", digits_logsoftmax)
                #print("digits_max", digits_max)
                #print("digits_argmax", digits_argmax)
                sequence_cumulative = np.zeros(7)
                length_cumulative = np.zeros(7)
                for i in range(1, 6):
                    sequence_cumulative[i] = sequence_cumulative[i-1] + digits_max[i-1].item()
                for i in range(0, 7):
                    length_cumulative[i] = sequence_cumulative[i] + length_logsoftmax[0][i].item()
                #print("sequence_cumulative", sequence_cumulative)
                #print("length_cumulative", length_cumulative)
                length_predition = np.argmax(length_cumulative)
                print("length_predition", length_predition)
                print("digit_predition", digits_argmax[:length_predition])
            
            break

        #accuracy = num_correct.item() / len(self._loader.dataset)
        #return accuracy
        return 0.0
