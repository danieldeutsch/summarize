import torch
import unittest
from allennlp.modules import FeedForward
from allennlp.nn import Activation

from summarize.modules.bridge import Bridge


class TestBridge(unittest.TestCase):
    def test_single_layer_not_shared(self):
        batch_size, input_hidden_size, output_hidden_size = 3, 5, 7
        layers = [FeedForward(input_hidden_size, 1, output_hidden_size, Activation.by_name('linear')())]
        bridge = Bridge(layers)

        input_tensor = torch.rand(batch_size, input_hidden_size)
        output_tensor = bridge(input_tensor)
        assert output_tensor.size() == (batch_size, output_hidden_size)

    def test_single_layer_shared(self):
        batch_size, input_hidden_size, output_hidden_size = 3, 8, 11
        layers = [FeedForward(input_hidden_size // 2, 1, output_hidden_size, Activation.by_name('linear')())]
        bridge = Bridge(layers, share_bidirectional_parameters=True)

        input_tensor = torch.rand(batch_size, input_hidden_size)
        output_tensor = bridge(input_tensor)
        assert output_tensor.size() == (batch_size, output_hidden_size * 2)

    def test_multi_layer_not_shared(self):
        batch_size, input_hidden_size, output_hidden_size = 3, 5, 7
        layers = [
            FeedForward(input_hidden_size, 1, output_hidden_size, Activation.by_name('linear')()),
            FeedForward(input_hidden_size, 1, output_hidden_size, Activation.by_name('linear')())
        ]
        bridge = Bridge(layers)

        # Pass the same tensor in both positions of the tuple. Then we are able
        # to verify the output tensors are different, which means they went
        # through different linear layers
        tensor = torch.rand(batch_size, input_hidden_size)
        input_tensors = (tensor, tensor)
        output_tensors = bridge(input_tensors)
        assert len(output_tensors) == 2
        assert output_tensors[0].size() == (batch_size, output_hidden_size)
        assert output_tensors[1].size() == (batch_size, output_hidden_size)
        assert not torch.equal(output_tensors[0], output_tensors[1])

    def test_multi_layer_shared(self):
        batch_size, input_hidden_size, output_hidden_size = 3, 8, 11
        layers = [
            FeedForward(input_hidden_size // 2, 1, output_hidden_size, Activation.by_name('linear')()),
            FeedForward(input_hidden_size // 2, 1, output_hidden_size, Activation.by_name('linear')())
        ]
        bridge = Bridge(layers, share_bidirectional_parameters=True)

        # Pass the same tensor in both positions of the tuple. Then we are able
        # to verify the output tensors are different, which means they went
        # through different linear layers
        tensor = torch.rand(batch_size, input_hidden_size)
        input_tensors = (tensor, tensor)
        output_tensors = bridge(input_tensors)
        assert len(output_tensors) == 2
        assert output_tensors[0].size() == (batch_size, output_hidden_size * 2)
        assert output_tensors[1].size() == (batch_size, output_hidden_size * 2)
        assert not torch.equal(output_tensors[0], output_tensors[1])
