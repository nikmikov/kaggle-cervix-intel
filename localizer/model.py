import torch
import torch.nn as nn

class CervixLocalisationModel(torch.nn.Module):
    def __init__(self, num_classes, batch_norm = False):
        super(CervixLocalisationModel, self).__init__()

        def conv2d(in_channels, out_channels, pool = True):
            # add layers: conv, [batch_norm], relu, max_pool
            l = [ torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1) ]
            if batch_norm:
                l += [ torch.nn.BatchNorm2d(out_channels) ]
            l +=  [ torch.nn.ReLU(inplace=True) ]
            if pool:
                l += [torch.nn.MaxPool2d(kernel_size=3, stride=2)]
            self._initialize_weights()
            return l


        features = []
        features += conv2d(3, 16)
        features += conv2d(16, 64)
        features += conv2d(64, 64)
        features += conv2d(64, 64)
        features += conv2d(64, 64)
        self.features = torch.nn.Sequential ( *features )
        self.classifier = torch.nn.Sequential (
            torch.nn.Dropout(p=0.25),
            torch.nn.Linear(64 * 7 * 7, 4096),
            torch.nn.BatchNorm1d(4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.25),
            torch.nn.Linear(4096, 4096),
            torch.nn.BatchNorm1d(4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(4096, num_classes)
        )

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, torch.nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, torch.nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(

            nn.Conv2d(3, 16, kernel_size=3,   padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(16, 64, kernel_size=3, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),


            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),


            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(64*7*7  , 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
#            nn.Linear(4096, 4096),
#            nn.BatchNorm1d(4096),
#            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def alexnet(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNet(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
    return model
