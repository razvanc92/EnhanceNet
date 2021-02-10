import numpy as np
import torch
import torch.nn as nn
from rnn.lib import utils
import torch.nn.functional as F

from rnn.model.dcrnn_cell import DCGRUCell

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Seq2SeqAttrs:
    def __init__(self, **model_kwargs):
        self.max_diffusion_step = int(model_kwargs.get('max_diffusion_step', 2))
        self.cl_decay_steps = int(model_kwargs.get('cl_decay_steps', 1000))
        self.filter_type = model_kwargs.get('filter_type', 'laplacian')
        self.num_nodes = int(model_kwargs.get('num_nodes', 1))
        self.num_rnn_layers = int(model_kwargs.get('num_rnn_layers', 1))
        self.rnn_units = int(model_kwargs.get('rnn_units'))
        self.hidden_state_size = self.num_nodes * self.rnn_units


class EncoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, **model_kwargs):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, **model_kwargs)
        self.input_dim = int(model_kwargs.get('input_dim', 1))
        self.seq_len = int(model_kwargs.get('seq_len'))  # for the encoder
        self.dcgru_layers = nn.ModuleList(
            [DCGRUCell(self.rnn_units, self.max_diffusion_step, self.num_nodes,
                       function=model_kwargs['function'],
                       bottleneck_dim=model_kwargs['bottleneck_dim'],
                       memory_dim=model_kwargs['memory_dim']) for _ in range(self.num_rnn_layers)])

    def forward(self, inputs, hidden_state=None, supports=None, full_input=None):
        """
        Encoder forward pass.

        :param inputs: shape (batch_size, self.num_nodes * self.input_dim)
        :param hidden_state: (num_layers, batch_size, self.hidden_state_size)
               optional, zeros if not provided
        :return: output: # shape (batch_size, self.hidden_state_size)
                 hidden_state # shape (num_layers, batch_size, self.hidden_state_size)
                 (lower indices mean lower layers)
        """
        batch_size, _ = inputs.size()
        if hidden_state is None:
            hidden_state = torch.zeros((self.num_rnn_layers, batch_size, self.hidden_state_size),
                                       ).to(device)
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num], supports=supports, full_input=full_input)
            hidden_states.append(next_hidden_state)
            output = next_hidden_state

        return output, torch.stack(hidden_states)  # runs in O(num_layers) so not too slow


class DecoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, **model_kwargs):
        # super().__init__(is_training, adj_mx, **model_kwargs)
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, **model_kwargs)
        self.output_dim = int(model_kwargs.get('output_dim', 1))
        self.horizon = int(model_kwargs.get('horizon', 1))  # for the decoder
        self.projection_layer = nn.Linear(self.rnn_units, self.output_dim)
        self.dcgru_layers = nn.ModuleList(
            [DCGRUCell(self.rnn_units, self.max_diffusion_step, self.num_nodes,
                       function=model_kwargs['function'],
                       bottleneck_dim=model_kwargs['bottleneck_dim'],
                       memory_dim=model_kwargs['memory_dim']) for _ in range(self.num_rnn_layers)])

    def forward(self, inputs, hidden_state=None, supports=None, full_input=None):
        """
        Decoder forward pass.

        :param inputs: shape (batch_size, self.num_nodes * self.output_dim)
        :param hidden_state: (num_layers, batch_size, self.hidden_state_size)
               optional, zeros if not provided
        :return: output: # shape (batch_size, self.num_nodes * self.output_dim)
                 hidden_state # shape (num_layers, batch_size, self.hidden_state_size)
                 (lower indices mean lower layers)
        """
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num], supports=supports, full_input=full_input)
            hidden_states.append(next_hidden_state)
            output = next_hidden_state

        projected = self.projection_layer(output.view(-1, self.rnn_units))
        output = projected.view(-1, self.num_nodes * self.output_dim)

        return output, torch.stack(hidden_states)


class DCRNNModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, adj_mx, logger, **model_kwargs):
        super().__init__()
        Seq2SeqAttrs.__init__(self, **model_kwargs)
        self.encoder_model = EncoderModel(**model_kwargs)
        self.decoder_model = DecoderModel(**model_kwargs)
        self.cl_decay_steps = int(model_kwargs.get('cl_decay_steps', 1000))
        self.use_curriculum_learning = bool(model_kwargs.get('use_curriculum_learning', False))
        self._logger = logger
        self.adaptive_supports = int(model_kwargs['adaptive_supports'])
        self.supports = []
        self.supports_len = 0

        self.gcn_bool = model_kwargs['function'] in ['gconv', 'gconv_dynamic']

        supports = []
        if model_kwargs['filter_type'] == "laplacian":
            supports.append(utils.calculate_scaled_laplacian(adj_mx, lambda_max=None))
        elif model_kwargs['filter_type'] == "random_walk":
            supports.append(utils.calculate_random_walk_matrix(adj_mx).T)
        elif model_kwargs['filter_type'] == "dual_random_walk":
            supports.append(utils.calculate_random_walk_matrix(adj_mx).T)
            supports.append(utils.calculate_random_walk_matrix(adj_mx.T).T)
        else:
            supports.append(utils.calculate_scaled_laplacian(adj_mx))
        for support in supports:
            self.supports_len += 1
            self.supports.append(torch.from_numpy(support.todense()).to(device))  # no more sparse

        if self.adaptive_supports:
            self.conv_a = nn.ModuleList()
            self.conv_b = nn.ModuleList()
            self.nodevec1 = nn.ParameterList()
            self.nodevec2 = nn.ParameterList()

            self.alpha1 = nn.Parameter(torch.ones(self.supports_len).to(device), requires_grad=True).to(device)
            self.alpha2 = nn.Parameter(torch.zeros(self.supports_len).to(device) + 0.001, requires_grad=True).to(device)
            self.alpha3 = nn.Parameter(torch.zeros(self.supports_len).to(device) + 0.001, requires_grad=True).to(device)

            for i in range(self.supports_len):
                self.conv_a.append(nn.Conv2d(in_channels=model_kwargs['input_dim'],
                                             out_channels=32,
                                             kernel_size=(1, 1)))
                self.conv_b.append(nn.Conv2d(in_channels=model_kwargs['input_dim'],
                                             out_channels=32,
                                             kernel_size=(1, 1)))

                self.nodevec1.append(
                    nn.Parameter(torch.rand(self.num_nodes, 10).to(device), requires_grad=True).to(device))
                self.nodevec2.append(
                    nn.Parameter(torch.rand(10, self.num_nodes).to(device), requires_grad=True).to(device))

    def _compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (
                self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def encoder(self, inputs, supports=None, full_input=None):
        """
        encoder forward pass on t time steps
        :param inputs: shape (seq_len, batch_size, num_sensor * input_dim)
        :return: encoder_hidden_state: (num_layers, batch_size, self.hidden_state_size)
        """
        encoder_hidden_state = None
        for t in range(self.encoder_model.seq_len):
            _, encoder_hidden_state = self.encoder_model(inputs[t], encoder_hidden_state, supports=supports, full_input=full_input)

        return encoder_hidden_state

    def decoder(self, encoder_hidden_state, labels=None, batches_seen=None, supports=None, full_input=None):
        """
        Decoder forward pass
        :param encoder_hidden_state: (num_layers, batch_size, self.hidden_state_size)
        :param labels: (self.horizon, batch_size, self.num_nodes * self.output_dim) [optional, not exist for inference]
        :param batches_seen: global step [optional, not exist for inference]
        :return: output: (self.horizon, batch_size, self.num_nodes * self.output_dim)
        """
        batch_size = encoder_hidden_state.size(1)
        go_symbol = torch.zeros((batch_size, self.num_nodes * self.decoder_model.output_dim)).to(device)
        decoder_hidden_state = encoder_hidden_state
        decoder_input = go_symbol

        outputs = []

        for t in range(self.decoder_model.horizon):
            decoder_output, decoder_hidden_state = self.decoder_model(decoder_input, decoder_hidden_state,
                                                                      supports=supports, full_input=full_input)
            decoder_input = decoder_output
            outputs.append(decoder_output)
            if self.training and self.use_curriculum_learning:
                c = np.random.uniform(0, 1)
                if c < self._compute_sampling_threshold(batches_seen):
                    decoder_input = labels[t]
        outputs = torch.stack(outputs)
        return outputs

    def forward(self, inputs, labels=None, batches_seen=None):
        """
        seq2seq forward pass
        :param inputs: shape (seq_len, batch_size, num_sensor * input_dim)
        :param labels: shape (horizon, batch_size, num_sensor * output)
        :param batches_seen: batches seen till now
        :return: output: (self.horizon, batch_size, self.num_nodes * self.output_dim)
        """

        new_supports = None
        batch_size = inputs.shape[1]
        horizon = inputs.shape[0]

        if self.gcn_bool:
            if self.adaptive_supports != 1:
                new_supports = torch.stack(self.supports).unsqueeze(1).expand(self.supports_len, batch_size,
                                                                              self.num_nodes, self.num_nodes)
            else:
                x = inputs.reshape(horizon, batch_size, self.num_nodes, -1)
                x = x.permute(1, 3, 2, 0)
                ts = x.shape[-1]
                new_supports = []
                for i in range(self.supports_len):
                    B = F.softmax(F.relu(torch.mm(self.nodevec1[i], self.nodevec2[i])), dim=1)

                    C1 = self.conv_a[i](x).permute(0, 3, 1, 2).contiguous().view(batch_size, self.num_nodes, 32 * ts)
                    C2 = self.conv_b[i](x).view(batch_size, 32 * ts, self.num_nodes)
                    C = F.softmax(F.relu(torch.matmul(C1, C2)), dim=1)

                    support = self.alpha1[i] * self.supports[i].unsqueeze(0).repeat(batch_size, 1, 1) + \
                              self.alpha2[i] * C + self.alpha3[i] * B.unsqueeze(0).repeat(batch_size, 1, 1)
                    new_supports.append(support)

        full_input = inputs.reshape(horizon, batch_size, self.num_nodes, -1).permute(1, 2, 3, 0).reshape(
            batch_size, self.num_nodes, -1)
        encoder_hidden_state = self.encoder(inputs, new_supports, full_input=full_input)
        self._logger.debug("Encoder complete, starting decoder")
        outputs = self.decoder(encoder_hidden_state, labels, batches_seen=batches_seen, supports=new_supports, full_input=full_input)
        self._logger.debug("Decoder complete")
        if batches_seen == 0:
            print('Encoder', sum(p.numel() for p in self.encoder_model.parameters() if p.requires_grad))
            self._logger.info(
                "Total trainable parameters {}".format(count_parameters(self)))
        return outputs
