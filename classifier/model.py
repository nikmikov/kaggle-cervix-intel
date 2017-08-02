import torch

import torchvision.models.vgg as vgg

class CervixClassificationModel(torch.nn.Module):
    def __init__(self, num_classes, batch_norm = False):
        super(CervixClassificationModel, self).__init__()

        # vgg16 model features
        self.features = [
            vgg.make_layers(vgg.cfg['D'], batch_norm=True),
            vgg.make_layers(vgg.cfg['A'], batch_norm=True),
            vgg.make_layers(vgg.cfg['A'], batch_norm=True),
            vgg.make_layers(vgg.cfg['A'], batch_norm=True),
            vgg.make_layers(vgg.cfg['A'], batch_norm=True)
        ]
        self.f0 = self.features[0]
        self.f1 = self.features[1]
        self.f2 = self.features[2]
        self.f3 = self.features[3]
        self.f4 = self.features[4]
        #self.features = torch.nn.Sequential ( *features )
        self.classifier = torch.nn.Sequential (
            torch.nn.Linear(512 * 7 * 7 + 2048 * 4, 4096 * 2),
#            torch.nn.BatchNorm1d(4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(4096 * 2, 4096),
#            torch.nn.BatchNorm1d(4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.5),
#            torch.nn.Linear(1024, 1024),
#            torch.nn.ReLU(inplace=True),
#            torch.nn.Dropout(p=0.15),
            torch.nn.Linear(4096, num_classes)
        )


    def forward(self, X):
        """
        """
        assert(len(X) == len(self.features))
        T = []
        for i in range(len(X)):
            x = self.features[i]( X[i] )
            x = x.view(x.size(0), -1)
            T.append(x)

        x = torch.cat(T, dim=1)
        x = self.classifier(x)
        return x
