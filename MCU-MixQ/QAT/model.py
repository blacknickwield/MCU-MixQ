import torch
from torch import nn
from quant import MixQActivConv2d, QActivLinear

cfg_tiny = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M']

class MixQTinyVGG(nn.Module):
    def __init__(self, w_bits: list, a_bits: list, share_weight: bool=False, 
                in_channels: int=3, cfg: list=None, batch_norm: bool=True, num_classes: int=10,
                *args, **kwargs) -> None:
        super(MixQTinyVGG, self).__init__(*args, **kwargs)
        self.quant_space = {
            'w_bits': w_bits,
            'a_bits': a_bits,
            'share_weight': share_weight,
        }

        self.in_channels = in_channels
        self.cfg = cfg if cfg is not None else cfg_tiny
        self.batch_norm = batch_norm
        self.layers = self.make_layers(in_channels=in_channels, cfg=self.cfg, batch_norm=batch_norm, quant_space=self.quant_space)
        # self.classifier = nn.Linear(self.cfg[-2] * 4 * 4, num_classes)
        self.classifier = QActivLinear(w_bit=8, a_bit=8, in_features=self.cfg[-2] * 4 * 4, out_features=num_classes)

        

    def forward(self, X: torch.Tensor):
        for layer in self.layers:
            X = layer(X)

        X = X.view(X.size(0), -1)
        y = self.classifier(X)
        return y

    def make_layers(self, in_channels: int, cfg: list, batch_norm: bool, quant_space: dict):
        w_bits = quant_space['w_bits']
        a_bits = quant_space['a_bits']
        share_weight = quant_space['share_weight']
        layers = []

        for layer in cfg:
            if layer == 'M': # max pooling
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else: # conv
                quant_conv2d = MixQActivConv2d(w_bits=w_bits, a_bits=a_bits, share_weight=share_weight,
                                               in_channels=in_channels, out_channels=layer, kernel_size=3, stride=1, padding=1)
                # conv2d = nn.Conv2d(in_channels=in_channels, out_channels=layer, kernel_size=3, padding=1, bias=False)
                layers += [quant_conv2d, nn.BatchNorm2d(num_features=layer)] if batch_norm else [quant_conv2d]
                in_channels = layer

        return nn.Sequential(*layers)
    
    def complexity_loss(self):
        loss = 0
        flops = []
        for module in self.modules():
            if isinstance(module, MixQActivConv2d) and hasattr(module, 'complexity_loss'):
                loss += module.complexity_loss()
                flops.append(module.flops)

        loss /= flops[0].item()
        return loss
    
    def fetch_arch(self):
        arch = { 'w_bit': [], 'a_bit': [], 'mix_w_bit': [], 'mix_a_bit': [] }
        for module in self.modules():
            if isinstance(model, MixQActivConv2d) and hasattr(self, 'fetch_arch'):
                m_arch = module.fetch_arch()
                for k in arch.keys():
                    if k in m_arch:
                        arch[k].append(m_arch[k])

        return arch
    
if __name__ == '__main__':
    w_btis = [bit for bit in range(2, 9, 1)]
    a_bits = [bit for bit in range(2, 9, 1)]
    share_weight = False
    in_channels = 3
    batch_norm = True
    cfg = None
    num_classes = 10
    model = MixQTinyVGG(
        w_bits=w_btis, a_bits=a_bits,
        share_weight=share_weight, in_channels=in_channels, cfg=cfg, batch_norm=batch_norm, num_classes=num_classes,
    )

    X = torch.randn((8, 3, 32, 32))
    y = model(X)
    print(y)
    print(y.shape)