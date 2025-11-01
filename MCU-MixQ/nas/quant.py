import torch
from torch import nn
import torch.nn.functional as F

gauss_steps = {
    1: 1.596,
    2: 0.996,
    3: 0.586,
    4: 0.336,
    5: 0.190,
    6: 0.106,
    7: 0.059,
    8: 0.032,
}
hwgq_steps = {
    1: 0.799,
    2: 0.538,
    3: 0.3217,
    4: 0.185,
    5: 0.104,
    6: 0.058,
    7: 0.033,
    8: 0.019,
}


class HWGQActiv(nn.Module):
    def __init__(self, bit: int, *args, **kwargs) -> None:
        super(HWGQActiv, self).__init__(*args, **kwargs)
        self.bit = bit
        self.step = hwgq_steps[bit]

    def forward(self, X: torch.Tensor):
        q_min, q_max = 0.0, float(2**self.bit - 1) * self.step
        y = torch.clamp(input=X, min=q_min, max=q_max)
        y = HWGQ.apply(y, self.step)

        return y


class HWGQ(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X: torch.Tensor, step: float):
        y = torch.round(X / step) * step

        return y

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class GaussQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X: torch.Tensor, bit: int, step: float):
        q_min, q_max = -(2 ** (bit - 1)), 2 ** (bit - 1) - 1
        y = torch.clamp(input=torch.round(X / step), min=q_min, max=q_max) * step
        return y

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


class GaussSymmetricQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X: torch.Tensor, bit: int, step: float):
        q_min, q_max = -(2 ** (bit - 1)), 2 ** (bit - 1) - 1
        alpha = X.std().item()
        step *= alpha
        y = torch.clamp(input=torch.round(X / step), q_min=q_min, q_max=q_max) * step
        return y

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


class QLinear(nn.Linear):
    def __init__(
        self, bit: int, in_features: int, out_features: int, *args, **kwargs
    ) -> None:
        self.bit = bit
        super(QLinear, self).__init__(
            in_features=in_features, out_features=out_features, *args, **kwargs
        )

    def forward(self, X):
        if self.bit == 32:
            output = F.linear(input=X, weight=self.weight, bias=self.bias)
        else:
            qweight = self.weight
            output = F.linear(input=X, weight=qweight, bias=self.bias)

        return output


class QConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs) -> None:
        self.bit = kwargs.pop("bit", 1)
        super(self, QConv2d).__init__(*args, **kwargs)

    def forward(self, X):
        if self.bit == 32:
            output = F.conv2d(
                input=X,
                weight=self.weight,
                bias=self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )
        else:
            qweight = self.weight
            output = F.conv2d(
                input=X,
                weight=qweight,
                bias=self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )

        return output


class QActivLinear(nn.Module):
    def __init__(
        self, w_bit: int, a_bit: int, in_features, out_features, *args, **kwargs
    ) -> None:
        super(QActivLinear, self).__init__(*args, **kwargs)
        self.w_bit = w_bit
        self.a_bit = a_bit
        self.q_activ = HWGQActiv(bit=a_bit)
        self.q_linear = QLinear(
            bit=w_bit, in_features=in_features, out_features=out_features
        )

    def forward(self, X: torch.Tensor):
        y = self.q_activ(X)
        y = self.q_linear(X)

        return y


class MixQConv2d(nn.Module):
    def __init__(
        self,
        bits: list,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride,
        padding,
        *args,
        **kwargs
    ):
        super(MixQConv2d, self).__init__(*args, **kwargs)
        self.bits = bits
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                )
                for bit in bits
            ]
        )  # Each bit with a conv
        self.alphas = nn.Parameter(torch.Tensor(len(bits)))
        self.alphas.data.fill_(0.01)
        self.steps = [gauss_steps[bit] for bit in bits]
        self.quantizer = GaussQuantize
        self.conv_params = {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "kernel_size": kernel_size,
            "stride": stride,
            "padding": padding,
            "bias": None,
        }

    def forward(self, X):
        quant_weights = []
        alphas = F.softmax(input=self.alphas, dim=0)
        for index, bit in enumerate(self.bits):
            weight: torch.Tensor = self.convs[index].weight
            weight_std = weight.std().item()
            step = self.steps[index] * weight_std
            quant_weight = self.quantizer.apply(weight, bit, step)
            quant_weight *= alphas[index]
            quant_weights.append(quant_weight)

        mix_quant_weight = sum(quant_weights)
        y = F.conv2d(
            input=X,
            weight=mix_quant_weight,
            bias=self.conv_params["bias"],
            stride=self.conv_params["stride"],
            padding=self.conv_params["padding"],
        )

        return y


class SharedMixQConv2d(nn.Module):
    def __init__(
        self,
        bits: list,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride,
        padding,
        *args,
        **kwargs
    ):
        super(SharedMixQConv2d, self).__init__(*args, **kwargs)
        self.bits = bits
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=None)
        self.alphas = nn.Parameter(torch.Tensor(len(bits)))
        self.alphas.data.fill_(0.01)
        self.steps = [gauss_steps[bit] for bit in bits]
        self.quantizer = GaussQuantize
        self.conv_params = {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "kernel_size": kernel_size,
            "stride": stride,
            "padding": padding,
            "bias": None,
        }

    def forward(self, X: torch.Tensor):
        quant_weights = []
        alphas = F.softmax(input=self.alphas, dim=0)
        weight = self.conv.weight
        weight_std = weight.std().item()

        for index, bit in enumerate(self.bits):
            step = self.steps[index] * weight_std
            quant_weight = self.quantizer.apply(X, bit, step)
            quant_weight *= alphas[index]
            quant_weights.append(quant_weight)

        mix_quant_weight = sum(quant_weights)
        y = F.conv2d(
            input=X,
            weight=mix_quant_weight,
            bias=self.conv_params["bias"],
            stride=self.conv_params["stride"],
            padding=self.conv_params["padding"],
        )

        return y


class MixQLinear(nn.Module):
    def __init__(self, bits: list, in_features: int, out_features: int, *args, **kwargs) -> None:
        super(MixQLinear, self).__init__(*args, **kwargs)
        self.bits = bits
        self.linears = nn.ModuleList([nn.Linear(in_features=in_features, out_features=out_features) for bit in bits])
        self.alphas = nn.Parameter(torch.Tensor(len(bits)))
        self.alphas.data.fill_(0.01)
        self.steps = [gauss_steps[bit] for bit in bits]
        self.quantizer = GaussQuantize
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, X: torch.Tensor):
        quant_weights = []
        alphas = F.softmax(input=self.alphas, dim=0)

        for index, bit in enumerate(self.bits):
            weight = self.linears[index].weight
            weight_std = weight.std().item()
            step = self.steps[index] * weight_std
            quant_weight = self.quantizer.apply(weight, bit, step)
            quant_weight *= alphas[index]
            quant_weights.append(quant_weight)

        mix_quant_weight = sum(quant_weights)
        y = F.linear(input=X, weight=mix_quant_weight)

        return y


class SharedMixQLinear(nn.Module):
    def __init__(self, bits: list, in_features: int, out_features: int, *args, **kwargs) -> None:
        super(SharedMixQLinear, self).__init__(*args, **kwargs)
        self.bits = bits
        self.linear = nn.Linear(in_features=in_features, out_features=out_features)
        self.quantizer = GaussQuantize
        self.steps = [gauss_steps[bit] for bit in bits]
        self.alphas = nn.Parameter(torch.Tensor(len(bits)))
        self.alphas.data.fill_(0.01)
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, X: torch.Tensor):
        quant_weights = []
        weight = self.linear.weight
        weight_std = weight.std().item()
        alphas = F.softmax(input=self.alphas, dim=0)

        for index, bit in enumerate(self.bits):
            step = self.steps[index] * weight_std
            quant_weight = self.quantizer.apply(weight, bit, step)
            quant_weight *= alphas[index]
            quant_weights.append(quant_weight)

        mix_quant_weight = sum(quant_weights)
        y = F.linear(input=X, weight=mix_quant_weight)

        return y


class MixQActiv(nn.Module):
    def __init__(self, bits: list, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.bits = bits
        self.alphas = nn.Parameter(torch.Tensor(len(bits)))
        self.alphas.data.fill_(0.01)
        self.activs = nn.ModuleList([HWGQActiv(bit=bit) for bit in bits])

    def forward(self, X: torch.Tensor):
        mix_activations = []
        alphas = F.softmax(input=self.alphas, dim=0)

        for index, activ in enumerate(self.activs):
            activation = activ(X) * alphas[index]
            mix_activations.append(activation)

        activation = sum(mix_activations)
        return activation


class MixQActivConv2d(nn.Module):
    def __init__(
        self,
        w_bits: list,
        a_bits: list,
        share_weight: bool,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride,
        padding,
        *args,
        **kwargs
    ):
        super(MixQActivConv2d, self).__init__(*args, **kwargs)
        self.w_bits = w_bits
        self.a_bits = a_bits
        self.share_weight = share_weight

        self.mix_activ = MixQActiv(bits=a_bits)
        self.mix_conv2d = (
            SharedMixQConv2d(
                bits=w_bits,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
            if share_weight
            else MixQConv2d(
                bits=w_bits,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
        )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.params = (
            out_channels * in_channels * kernel_size[0] * kernel_size[1]
            if isinstance(kernel_size, tuple)
            else out_channels * in_channels * kernel_size
        ) * 1e-6
        self.filters = self.params / (stride ** 2)

        self.register_buffer("flops", torch.tensor(0, dtype=torch.float))
        self.register_buffer("memory", torch.tensor(0, dtype=torch.float))

    def forward(self, X: torch.Tensor):
        shape = X.shape
        memory = torch.tensor(shape[1] * shape[2] * shape[3] * 1e-3, dtype=torch.float)
        flops = torch.tensor(self.filters * shape[-1] * shape[-2], dtype=torch.float)
        self.memory.copy_(memory)
        self.flops.copy_(flops)

        y = self.mix_activ(X)
        y = self.mix_conv2d(y)

        return y

    def complexity_loss(self):
        conv_alphas = self.mix_conv2d.alphas
        activ_alphas = self.mix_activ.alphas
        conv_alphas = F.softmax(input=conv_alphas, dim=-1)
        activ_alphas = F.softmax(input=activ_alphas, dim=-1)
        w_bit, a_bit = 0, 0
        for index, _ in enumerate(self.w_bits):
            w_bit += conv_alphas[index] * self.w_bits[index]

        for index, _ in enumerate(self.a_bits):
            a_bit += activ_alphas[index] * self.a_bits[index]

        complexity = w_bit * a_bit * self.flops

    def fetch_arch(self):
        conv_alphas = F.softmax(input=self.mix_conv2d.alphas, dim=0)
        activ_alphas = F.softmax(input=self.mix_activ.alphas, dim=0)
        w_bit = self.w_bits[conv_alphas.argmax().item()]
        a_bit = self.a_bits[activ_alphas.argmax().item()]
        mix_w_bit = sum((self.w_bits[index] * conv_alphas[index] for index, _ in enumerate(self.w_bits)))
        mix_a_bit = sum((self.a_bits[index] * activ_alphas[index] for index, _ in enumerate(self.a_bits)))

        arch = { 'w_bit': w_bit, 'a_bit': a_bit, 'mix_w_bit': mix_w_bit, 'mix_a_bit': mix_a_bit }

        return arch


class MixQActivLinear(nn.Module):
    def __init__(self, w_bits: int, a_bits: int, share_weight: bool, in_features: int, out_featuers: int, *args, **kwargs):
        super(MixQActivLinear, self).__init__(*args, **kwargs)
        self.w_bits = w_bits
        self.a_bits = a_bits
        self.share_weight = share_weight
        self.mix_activ = MixQActiv(bits=a_bits)
        self.mix_linear = SharedMixQLinear(bits=w_bits) if share_weight else MixQLinear(bits=w_bits)
        self.in_features = in_features
        self.out_features = out_featuers
        self.params = in_features * out_featuers * 1e-6
        
        self.register_buffer('flops', torch.tensor(0, dtype=torch.float))
        self.register_buffer('memory', torch.tensor(0, dtype=torch.float))

    def forward(self, X: torch.Tensor):
        
        y = self.mix_activ(X)
        y = self.mix_linear(X)

        return y

    def complexity_loss(self):
        linear_alphas = self.mix_linear.alphas
        activ_alphas = self.mix_activ.alphas
        linear_alphas = F.softmax(input=linear_alphas, dim=0)
        activ_alphas = F.softmax(input=activ_alphas, dim=0)
        w_bit = sum((linear_alphas[index] * self.w_bits[index] for index, _ in enumerate(self.w_bits)))
        a_bit = sum((activ_alphas[index] * self.a_bits[index] for index, _ in enumerate(self.a_bits)))
        complexity = w_bit * a_bit * self.flops.item()

        return complexity