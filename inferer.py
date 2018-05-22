from __future__ import division

import torch.utils.data
from torch.autograd import Variable
import torch.nn
from dataset import Dataset
import torchvision.datasets
import numpy as np

from PIL import Image
from torchvision import transforms
import math
import time


class Inferer(object):
    def __init__(self, path_to_lmdb_dir):
        pass
        self._loader = torch.utils.data.DataLoader(Dataset(path_to_lmdb_dir), batch_size=1, shuffle=False)
        # https://pytorch.org/docs/master/torchvision/datasets.html#imagefolder
        # class torchvision.datasets.ImageFolder(root, transform=None, target_transform=None, loader=<function default_loader>)

    def infer(self, model):
        print("Hello world")
        model.eval()
        #num_correct = 0
        #needs_include_length = False

        #for batch_idx, (images, length_labels, digits_labels) in enumerate(self._loader):
        """
        print(type(images))
        print(type(length_labels))
        print(type(digits_labels))
        <class 'torch.Tensor'>
        <class 'torch.Tensor'>
        <type 'list'>            
        """

        image_files = [
            "/data/infer/7.png",
            "/data/infer/8.png",
            "/data/infer/8a.png",
            "/data/infer/9.png",
            "/data/infer/10.png",
            "/data/infer/11.png",
            "/data/infer/12.png",
            "/data/infer/L.png",
            "/data/infer/La.png",
        ]

        # measure time
        start_time = time.time()

        for file in image_files:

            image = Image.open(file)
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
            image = transform(image)
            #print("image.size()", image.size())
            images = image[None]
            #print("images.size()", images.size())

            #gpu
            #images = Variable(images.cuda(), requires_grad=True)
            #cpu
            images = Variable(images, requires_grad=True)

            with torch.no_grad():
                #print("Infer>", file)
                #print("images.size()", images.size())
                length_logits, digits_logits = model(images)
                #print("length_logits", length_logits)
                #print("digits_logits", digits_logits)
                # See paper appendix for the math
                lsm = torch.nn.LogSoftmax(1)
                length_logsoftmax = lsm(length_logits)
                #print("length_logsoftmax", length_logsoftmax)

                digits_logsoftmax = [lsm(digit_logits) for digit_logits in digits_logits]
                #print("digits_logsoftmax", digits_logsoftmax)
                digits_max = [x.max() for x in digits_logsoftmax]
                #print("digits_max", digits_max)
                digits_argmax = [torch.argmax(x) for x in digits_logsoftmax]
                #print("digits_argmax", digits_argmax)
                digits_prob = [torch.pow(10.0, x) for x in digits_max]
                #print("digits_prob", digits_prob)
                sequence_cumulative = np.zeros(7)
                length_cumulative = np.zeros(7)
                for i in range(1, 6):
                    sequence_cumulative[i] = sequence_cumulative[i-1] + digits_max[i-1].item()
                for i in range(0, 7):
                    length_cumulative[i] = sequence_cumulative[i] + length_logsoftmax[0][i].item()
                #print("sequence_cumulative", sequence_cumulative)
                #print("length_cumulative", length_cumulative)
                length_prediction = np.argmax(length_cumulative)
                length_prob = math.pow(10.0, length_cumulative[length_prediction])
                print("==>", file)
                print("length_prediction", length_prediction)
                print("length_prob", length_prob)
                print("digits_prediction", digits_argmax[:length_prediction])
                print("digits_prob", digits_prob[:length_prediction])
                
                #break

        #accuracy = num_correct.item() / len(self._loader.dataset)
        #return accuracy

        # report speed
        duration = time.time() - start_time
        duration_mean = duration / len(image_files)
        print("duration", duration)
        print("duration_mean", duration_mean)
        print("fps", 1.0 / duration_mean)


        return 0.0
