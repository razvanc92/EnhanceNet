import torch
import torch.nn as nn
import torch.nn.functional as F

from tcn.model.memory_conv2d import Conv2dWrapper


class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('ncvl, nvw->ncwl', (x, A))
        return x.contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)


class gcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super(gcn, self).__init__()
        self.nconv = nconv()
        self.c_in = c_in
        c_in = (order * support_len + 1) * c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = [x]
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class gwnet(nn.Module):
    def __init__(self, device, num_nodes, dropout=0.3, supports=None, gcn_bool=True, addaptadj=True,
                 in_dim=2, out_dim=12, residual_channels=32, dilation_channels=32, skip_channels=256, end_channels=512,
                 kernel_size=2, blocks=4, layers=2, temporal_memory=0, temporal_num_features=16,                  adaptive_supports=0):

        super(gwnet, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.temporal_memory = temporal_memory
        self.adaptive_supports = adaptive_supports != 0
        self.num_nodes = num_nodes

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()
        self.device = device

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))

        self.supports = supports

        receptive_field = 1

        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        if gcn_bool and addaptadj:
            print('Added Graph Wavenet support')
            self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
            self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)
            self.supports_len += 1

        if temporal_memory == 0:
            print('Using NO temporal memory')
        if temporal_memory == 1:
            print('Using trainable temporal memory of size: ', temporal_num_features)

        self.temporal_memory = nn.ParameterList()
        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions
                if temporal_memory == 1:
                    self.temporal_memory.append(nn.Parameter(torch.randn(num_nodes, temporal_num_features).to(device),
                                                             requires_grad=True).to(device))
                else:
                    self.temporal_memory.append(nn.Parameter(torch.randn(1).to(device),
                                                             requires_grad=True).to(device))

                self.filter_convs.append(Conv2dWrapper(
                    num_nodes=num_nodes,
                    memory_size=temporal_num_features,
                    in_channels=residual_channels,
                    out_channels=dilation_channels,
                    kernel_size=(1, kernel_size), dilation=new_dilation,
                    use_memory=temporal_memory != 0,
                    dropout_ratio=0.))

                self.gate_convs.append(Conv2dWrapper(
                    memory_size=temporal_num_features,
                    in_channels=residual_channels,
                    out_channels=dilation_channels,
                    num_nodes=num_nodes,
                    kernel_size=(1, kernel_size),
                    dilation=new_dilation,
                    use_memory=temporal_memory != 0,
                    dropout_ratio=0.))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))

                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.gcn_bool:
                    self.gconv.append(gcn(dilation_channels, residual_channels, dropout, support_len=self.supports_len))

        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                    out_channels=end_channels,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1, 1),
                                    bias=True)

        if adaptive_supports != 0:
            self.adaptive_gcn = nn.ModuleList()
            self.conv_a = nn.ModuleList()
            self.conv_b = nn.ModuleList()
            self.nodevec1_adp = nn.ParameterList()
            self.nodevec2_adp = nn.ParameterList()

            self.alpha1 = nn.Parameter(torch.ones(self.supports_len).to(device), requires_grad=True).to(device)
            self.alpha2 = nn.Parameter(torch.zeros(self.supports_len).to(device) + 0.01, requires_grad=True).to(device)
            self.alpha3 = nn.Parameter(torch.zeros(self.supports_len).to(device) + 0.01, requires_grad=True).to(device)

            for i in range(self.supports_len):
                self.conv_a.append(nn.Conv2d(in_channels=in_dim,
                                             out_channels=32,
                                             kernel_size=(1, 1)))
                self.conv_b.append(nn.Conv2d(in_channels=in_dim,
                                             out_channels=32,
                                             kernel_size=(1, 1)))

                self.nodevec1_adp.append(
                    nn.Parameter(torch.rand(num_nodes, 10).to(device), requires_grad=True).to(device))
                self.nodevec2_adp.append(
                    nn.Parameter(torch.rand(10, num_nodes).to(device), requires_grad=True).to(device))

        self.receptive_field = receptive_field
        self.dropout = nn.Dropout(dropout)
        print('Supports len', self.supports_len)

    def forward(self, input):
        bsize = input.size(0)
        in_len = input.size(3)

        if in_len < self.receptive_field:
            x = nn.functional.pad(input, (self.receptive_field - in_len, 0, 0, 0))
        else:
            x = input

        x = self.start_conv(x)
        skip = 0

        if not self.adaptive_supports:
            if self.gcn_bool and self.addaptadj:
                adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
                supports = self.supports + [adp]
                new_supports = torch.stack(supports).unsqueeze(1).expand(self.supports_len, bsize, self.num_nodes,
                                                                         self.num_nodes)
            else:
                new_supports = torch.stack(self.supports).unsqueeze(1).expand(self.supports_len, bsize, self.num_nodes,
                                                                              self.num_nodes)
        else:
            if self.gcn_bool and self.addaptadj:
                adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
                supports = self.supports + [adp]
            else:
                supports = self.supports

            ts = x.shape[-1]
            new_supports = []
            x_padded = nn.functional.pad(input, (1, 0, 0, 0)) if input.size(3) == self.out_dim else input
            for i in range(len(supports)):
                B = F.softmax(F.relu(torch.mm(self.nodevec1_adp[i], self.nodevec2_adp[i])), dim=1)

                C1 = self.conv_a[i](x_padded).permute(0, 3, 1, 2).contiguous().view(bsize, self.num_nodes, 32 * ts)
                C2 = self.conv_b[i](x_padded).view(bsize, 32 * ts, self.num_nodes)
                C = F.softmax(F.relu(torch.matmul(C1, C2)), dim=2)

                support = self.alpha1[i] * supports[i].unsqueeze(0).repeat(bsize, 1, 1) + \
                          self.alpha2[i] * C + self.alpha3[i] * B.unsqueeze(0).repeat(bsize, 1, 1)

                new_supports.append(support)

        # WaveNet layers
        for i in range(self.blocks * self.layers):
            residual = x
            # dilated convolution
            filter = self.filter_convs[i](residual, self.temporal_memory[i])
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual, self.temporal_memory[i])
            gate = torch.sigmoid(gate)
            x = filter * gate

            # parametrized skip connection
            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :, -s.size(3):]
            except:
                skip = 0
            skip = s + skip

            if self.gcn_bool:
                x = self.gconv[i](x, new_supports)

            x = x + residual[:, :, :, -x.size(3):]
            x = self.bn[i](x)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)

        return x
