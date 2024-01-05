import torch
from torch import nn
from opt_einsum import contract

class Triaffine(nn.Module):
    r"""
    The code is based on Supar. But add n_out dim to the output
    Triaffine layer for second-order scoring.

    This function has a tensor of weights :math:`W` and bias terms if needed.
    The score :math:`s(x, y, z)` of the vector triple :math:`(x, y, z)` is computed as :math:`x^T z^T W y`.
    Usually, :math:`x` and :math:`y` can be concatenated with bias terms.

    References:
        - Yu Zhang, Zhenghua Li and Min Zhang. 2020.
          `Efficient Second-Order TreeCRF for Neural Dependency Parsing`_.
        - Xinyu Wang, Jingxian Huang, and Kewei Tu. 2019.
          `Second-Order Semantic Dependency Parsing with End-to-End Neural Networks`_.

    Args:
        n_in (int):
            The size of the input feature.
        n_out (int):
            The size of the output feature.
        bias_x (bool):
            If ``True``, adds a bias term for tensor :math:`x`. Default: ``False``.
        bias_y (bool):
            If ``True``, adds a bias term for tensor :math:`y`. Default: ``False``.
    """

    def __init__(self, n_in, n_out = 1, bias_x=False, bias_y=False, mean = 0, std = 1):
        super().__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.mean = mean
        self.std = std
        self.weight = nn.Parameter(torch.randn(n_out, n_in+bias_x, n_in, n_in+bias_y))
        # self.weight = nn.Parameter(torch.Tensor(n_out, n_in+bias_x, n_in, n_in+bias_y))

        self.reset_parameters()

    def __repr__(self):
        s = f"n_in={self.n_in}, n_out={self.n_out}"
        if self.bias_x:
            s += f", bias_x={self.bias_x}"
        if self.bias_y:
            s += f", bias_y={self.bias_y}"

        return f"{self.__class__.__name__}({s})"

    def reset_parameters(self):
        # nn.init.zeros_(self.weight)
        nn.init.normal_(self.weight, self.mean, self.std)

    def forward(self, x, y, z):
        r"""
        Args:
            x (torch.Tensor): ``[batch_size, seq_len, n_in]``.
            y (torch.Tensor): ``[batch_size, seq_len, n_in]``.
            z (torch.Tensor): ``[batch_size, seq_len, n_in]``.

        Returns:
            ~torch.Tensor:
                A scoring tensor of shape ``[batch_size, seq_len, seq_len, seq_len]``.
        """

        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)

        w = contract('bzk,oikj->bozij', z, self.weight, backend='torch')
        # [batch_size, seq_len, seq_len, seq_len, n_out]
        s = contract('bxi,bozij,byj->bzxyo', x, w, y, backend='torch')
        return s