import numpy as np
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class LayerParams:
    def __init__(self, rnn_network: torch.nn.Module, layer_type: str):
        self._rnn_network = rnn_network
        self._params_dict = {}
        self._biases_dict = {}
        self._type = layer_type
        self.weights = None
        self.biases = None

    def get_weights(self, shape):
        if shape not in self._params_dict:
            nn_param = torch.nn.Parameter(torch.empty(*shape, device=device))
            torch.nn.init.xavier_normal_(nn_param)
            self._params_dict[shape] = nn_param
            self._rnn_network.register_parameter('{}_weight_{}'.format(self._type, str(shape)),
                                                 nn_param)
        return self._params_dict[shape]

    def get_biases(self, length, bias_start=0.0):
        if length not in self._biases_dict:
            biases = torch.nn.Parameter(torch.empty(length, device=device))
            torch.nn.init.constant_(biases, bias_start)
            self._biases_dict[length] = biases
            self._rnn_network.register_parameter('{}_biases_{}'.format(self._type, str(length)),
                                                 biases)

        return self._biases_dict[length]


class DCGRUCell(torch.nn.Module):
    def __init__(self, num_units, max_diffusion_step, num_nodes, nonlinearity='tanh', function='gconv',
                 bottleneck_dim=4, memory_dim=16):
        """
        :param num_units:
        :param adj_mx:
        :param max_diffusion_step:
        :param num_nodes:
        :param nonlinearity:
        :param filter_type: "laplacian", "random_walk", "dual_random_walk".
        :param use_gc_for_ru: whether to use Graph convolution to calculate the reset and update gates.
        """

        super().__init__()
        self._activation = torch.tanh if nonlinearity == 'tanh' else torch.relu
        self._num_nodes = num_nodes
        self._num_units = num_units
        self._max_diffusion_step = max_diffusion_step
        self._function = function
        self._memory_dim = memory_dim
        self._bottleneck_dim = bottleneck_dim

        self._rnn_params = LayerParams(self, 'rnn_params')
        self._rnn_params2 = LayerParams(self, 'rnn_params2')

    @staticmethod
    def _build_sparse_matrix(L):
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        # this is to ensure row-major ordering to equal torch.sparse.sparse_reorder(L)
        indices = indices[np.lexsort((indices[:, 0], indices[:, 1]))]
        L = torch.sparse_coo_tensor(indices.T, L.data, L.shape).to(device)
        return L

    def forward(self, inputs, hx, supports=None, full_input=None):
        """Gated recurrent unit (GRU) with Graph Convolution.
        :param inputs: (B, num_nodes * input_dim)
        :param hx: (B, num_nodes * rnn_units)

        :return
        - Output: A `2-D` tensor with shape `(B, num_nodes * rnn_units)`.
        """
        output_size = 2 * self._num_units
        if self._function == 'gconv':
            fn = self._gconv
        elif self._function == 'gconv_dynamic':
            fn = self._gconv_dynamic
        elif self._function == 'fc':
            fn = self._fc
        elif self._function == 'fc_dynamic':
            fn = self._fc_dynamic
        else:
            raise Exception('FN' + self._function + 'NOT IMPLEMENTED')

        value = torch.sigmoid(
            fn(inputs, hx, output_size, bias_start=1.0, param_layer=self._rnn_params, supports=supports,
               full_input=full_input))
        value = torch.reshape(value, (-1, self._num_nodes, output_size))
        r, u = torch.split(tensor=value, split_size_or_sections=self._num_units, dim=-1)
        r = torch.reshape(r, (-1, self._num_nodes * self._num_units))
        u = torch.reshape(u, (-1, self._num_nodes * self._num_units))

        c = fn(inputs, r * hx, self._num_units, param_layer=self._rnn_params2, supports=supports, full_input=full_input)
        if self._activation is not None:
            c = self._activation(c)

        if self._function in ['fc_dynamic', 'fc']:
            c = torch.reshape(c, (-1, self._num_nodes, output_size))
            c = torch.reshape(c, (-1, self._num_nodes * self._num_units))

        new_state = u * hx + (1.0 - u) * c
        return new_state

    @staticmethod
    def _concat(x, x_):
        x_ = x_.unsqueeze(0)
        return torch.cat([x, x_], dim=0)

    def _fc(self, inputs, state, output_size, bias_start=0.0, param_layer=None, supports=None, full_input=None):
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size * self._num_nodes, -1))
        state = torch.reshape(state, (batch_size * self._num_nodes, -1))
        inputs_and_state = torch.cat([inputs, state], dim=-1)

        input_size = inputs_and_state.shape[-1]
        weights = param_layer.get_weights((input_size, output_size))
        value = torch.sigmoid(torch.matmul(inputs_and_state, weights))
        biases = param_layer.get_biases(output_size, bias_start)
        value = value + biases
        return value

    def _fc_dynamic(self, inputs, state, output_size, bias_start=0.0, param_layer=None, supports=None, full_input=None):
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size * self._num_nodes, -1))
        state = torch.reshape(state, (batch_size * self._num_nodes, -1))
        inputs_and_state = torch.cat([inputs, state], dim=-1)
        input_size = inputs_and_state.shape[-1]

        if self.training:
            memory = param_layer.get_weights((self._num_nodes, self._memory_dim))
            w1 = param_layer.get_weights((memory.shape[1], memory.shape[1]))
            b1 = param_layer.get_biases(memory.shape[1], bias_start)

            w2 = param_layer.get_weights((memory.shape[1], self._bottleneck_dim))
            b2 = param_layer.get_biases(self._bottleneck_dim, bias_start)

            w3 = param_layer.get_weights((self._bottleneck_dim, input_size * output_size))
            b3 = param_layer.get_biases(input_size * output_size, bias_start)

            mem = torch.tanh(torch.matmul(memory, w1) + b1)
            mem = torch.tanh(torch.matmul(mem, w2) + b2)
            weights = (torch.matmul(mem, w3) + b3).reshape([self._num_nodes, input_size, output_size])
            weights = weights.unsqueeze(0).repeat(batch_size, 1, 1, 1)
            weights = weights.reshape([batch_size * self._num_nodes, input_size, output_size])

            param_layer.weights = weights
        else:
            weights = param_layer.weights

        b_out = param_layer.get_biases(output_size, bias_start)
        value = torch.sigmoid(torch.matmul(inputs_and_state.unsqueeze(1), weights).squeeze())
        value = value + b_out
        return value

    def _gconv(self, inputs, state, output_size, bias_start=0.0, param_layer=None, supports=None, full_input=None):
        if supports is None:
            raise Exception("No supports in gconv")

        # Reshape input and state to (batch_size, num_nodes, input_dim/state_dim)
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size, self._num_nodes, -1))
        state = torch.reshape(state, (batch_size, self._num_nodes, -1))
        inputs_and_state = torch.cat([inputs, state], dim=2)
        input_size = inputs_and_state.size(2)

        x0 = inputs_and_state
        x = x0.unsqueeze(0)

        if self._max_diffusion_step == 0:
            pass
        else:
            for support in supports:
                x1 = torch.matmul(support, x0)
                x = self._concat(x, x1)

                for k in range(2, self._max_diffusion_step + 1):
                    x2 = 2 * torch.matmul(support, x1) - x0
                    x = self._concat(x, x2)
                    x1, x0 = x2, x1

        num_matrices = len(supports) * self._max_diffusion_step + 1  # Adds for x itself.
        x = x.permute(1, 2, 3, 0)  # (batch_size, num_nodes, input_size, order)
        x = torch.reshape(x, shape=[batch_size * self._num_nodes, input_size * num_matrices])

        weights = param_layer.get_weights((input_size * num_matrices, output_size))
        x = torch.matmul(x, weights)  # (batch_size * self._num_nodes, output_size)

        biases = param_layer.get_biases(output_size, bias_start)
        x = x + biases
        # Reshape res back to 2D: (batch_size, num_node, state_dim) -> (batch_size, num_node * state_dim)
        return torch.reshape(x, [batch_size, self._num_nodes * output_size])

    def _gconv_dynamic(self, inputs, state, output_size, bias_start=0.0, param_layer=None, supports=None,
                       full_input=None):
        if supports is None:
            raise Exception("No supports in gconv")

        # Reshape input and state to (batch_size, num_nodes, input_dim/state_dim)
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size, self._num_nodes, -1))
        state = torch.reshape(state, (batch_size, self._num_nodes, -1))
        inputs_and_state = torch.cat([inputs, state], dim=2)
        input_size = inputs_and_state.size(2)

        x0 = inputs_and_state
        x = x0.unsqueeze(0)

        if self._max_diffusion_step == 0:
            pass
        else:
            for support in supports:
                x1 = torch.matmul(support, x0)
                x = self._concat(x, x1)

                for k in range(2, self._max_diffusion_step + 1):
                    x2 = 2 * torch.matmul(support, x1) - x0
                    x = self._concat(x, x2)
                    x1, x0 = x2, x1

        num_matrices = len(supports) * self._max_diffusion_step + 1  # Adds for x itself.
        x = x.permute(1, 2, 3, 0)  # (batch_size, num_nodes, input_size, order)
        x = torch.reshape(x, shape=[batch_size * self._num_nodes, input_size * num_matrices])

        if self.training:
            memory = param_layer.get_weights((self._num_nodes, self._memory_dim))
            w1 = param_layer.get_weights((memory.shape[1], memory.shape[1]))
            b1 = param_layer.get_biases(memory.shape[1], bias_start)

            w2 = param_layer.get_weights((memory.shape[1], self._bottleneck_dim))
            b2 = param_layer.get_biases(self._bottleneck_dim, bias_start)

            w3 = param_layer.get_weights((self._bottleneck_dim, input_size * output_size * num_matrices))
            # b3 = param_layer.get_biases(input_size * output_size * num_matrices, bias_start)

            mem = torch.tanh(torch.matmul(memory, w1) + b1)
            mem = torch.tanh(torch.matmul(mem, w2) + b2)
            weights = (torch.matmul(mem, w3)).reshape([self._num_nodes, input_size * num_matrices, output_size])
            weights = weights.unsqueeze(0).repeat(batch_size, 1, 1, 1)
            weights = weights.reshape([batch_size * self._num_nodes, input_size * num_matrices, output_size])

            param_layer.weights = weights
        else:
            weights = param_layer.weights

        b_out = param_layer.get_biases(output_size, bias_start)
        x = torch.matmul(x.unsqueeze(1), weights).squeeze()  # (batch_size * self._num_nodes, output_size)
        x = x + b_out
        # Reshape res back to 2D: (batch_size, num_node, state_dim) -> (batch_size, num_node * state_dim)
        return torch.reshape(x, [batch_size, self._num_nodes * output_size])
