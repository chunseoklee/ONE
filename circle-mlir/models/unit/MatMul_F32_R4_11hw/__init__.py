import torch


# Generate MatMul+Add operator with Float32, Rank-4(11HW)
class net_Linear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.op = torch.nn.Linear(4, 6)

    def forward(self, input):
        return self.op(input)

    def onnx_opset_version(self):
        # TODO set to appropriate value
        return 13


_model_ = net_Linear()

_inputs_ = torch.randn(1, 1, 3, 4)
