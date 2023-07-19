from torchdistill.models.registry import register_model_class
from torch import nn
from torchdistill.common.tensor_util import QuantizedTensor, quantize_tensor, dequantize_tensor


def linear_batch_relu(num_in_nodes, num_out_nodes):
    return [nn.Linear(num_in_nodes, num_out_nodes), nn.BatchNorm1d(num_out_nodes), nn.ReLU(inplace=True)]


class BaseModel(nn.Module):
    def encode(self, *args, **kwargs):
        raise NotImplementedError()

    def decode(self, *args, **kwargs):
        raise NotImplementedError()


@register_model_class
class HtoV(BaseModel):
    @staticmethod
    def _make_block(node_counts):
        layer_list = list()
        for i in range(len(node_counts) - 2):
            layer_list.extend(linear_batch_relu(node_counts[i], node_counts[i + 1]))

        layer_list.append(nn.Linear(node_counts[-2], node_counts[-1]))
        return layer_list

    def __init__(self, encoder_node_counts, decoder_node_counts, num_q_bits=0):
        super().__init__()
        encoder_layers = self._make_block(encoder_node_counts)
        decoder_layers = self._make_block(decoder_node_counts)
        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)
        self.num_q_bits = num_q_bits

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def encode(self, x):
        x = self.encoder(x)
        return quantize_tensor(x, self.num_q_bits) if self.num_q_bits > 0 and not self.training else x

    def decode(self, z):
        if isinstance(z, QuantizedTensor):
            z = dequantize_tensor(z)
        return self.decoder(z)
