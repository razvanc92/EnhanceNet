import torch.nn as nn
import torch.nn.functional as F

from tcn.model.basic_structure import MLP


class Conv2dWrapper(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, num_nodes=325,
                 memory_size=16, use_memory=False, dropout_ratio=0.):
        super(Conv2dWrapper, self).__init__()
        self.use_memory = use_memory
        if use_memory:
            self.convLayer = MemoryConv2d(in_channels, out_channels, kernel_size, stride, dilation, num_nodes,
                                          memory_size, dropout_ratio=dropout_ratio)
        else:
            self.convLayer = nn.Conv2d(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=kernel_size,
                                       dilation=dilation,
                                       stride=stride)

    def forward(self, inputs, memory):
        if self.use_memory:
            return self.convLayer(inputs, memory)
        else:
            return self.convLayer(inputs)


class MemoryConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, num_nodes=325,
                 memory_size=16, dropout_ratio=0.):
        super(MemoryConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.bias_size = out_channels
        self.num_nodes = num_nodes
        self.memory_size = memory_size

        self.metalearner = MemoryConvLearner(in_channels, out_channels, kernel_size, stride=stride,
                                             dilation=dilation, num_nodes=num_nodes, memory_size=memory_size,
                                             dropout_ratio=dropout_ratio)

    def convolve(self, input, weights, bias):
        batch_size = input.size(0)
        t = input.size(-1)
        input = input.permute(0, 2, 1, 3).reshape(batch_size, self.in_channels * self.num_nodes, t)
        y_grouped = F.conv1d(input, weights, bias=bias, groups=self.num_nodes, dilation=self.dilation)
        result = y_grouped.view(batch_size, self.num_nodes, self.out_channels, -1).permute(0, 2, 1, 3)
        return result

    def forward(self, inputs, memory):
        weights, bias = self.metalearner(memory)
        return self.convolve(inputs, weights, bias)


class MemoryConvLearner(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 dilation=1, memory_size=11, num_nodes=207, dropout_ratio=0.):
        super(MemoryConvLearner, self).__init__()
        hiddens = [16, 3]

        print('Bottle neck size:', hiddens[-1])
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.bias_size = out_channels
        self.num_nodes = num_nodes

        self.meta_weights = MLP(hiddens + [in_channels * out_channels * kernel_size[1]], memory_size,
                                activation_function=nn.ReLU(), out_act=False)
        self.meta_bias = MLP(hiddens + [out_channels], memory_size, activation_function=nn.ReLU(), out_act=False)

    def forward(self, features):
        weights = self.meta_weights(features)
        weights = weights.reshape(self.num_nodes * self.out_channels, self.in_channels, self.kernel_size[1])

        bias = self.meta_bias(features).reshape(-1)

        return weights, bias
