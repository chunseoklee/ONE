import torch


# Generate Add operator with Float32, Rank-4
class net_add(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        return torch.add(inputs[0], inputs[1])

    def onnx_opset_version(self):
        # TODO set version
        return 10


_model_ = net_add()

_inputs_ = [torch.randn(1, 2, 3, 3), torch.randn(1, 2, 3, 3)]
