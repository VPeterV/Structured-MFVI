# -*- coding: utf-8 -*-

from multiprocessing.sharedctypes import Value
import pdb
import torch
import torch.nn as nn

def apply_fencepost(x):
    x_f, x_b = x.chunk(2, -1)
    x_boundary = torch.cat((x_f[:, :-1], x_b[:, 1:]), -1)
    return x_boundary

class Biaffine(nn.Module):
    r"""
    Biaffine layer for first-order scoring :cite:`dozat-etal-2017-biaffine`.

    This function has a tensor of weights :math:`W` and bias terms if needed.
    The score :math:`s(x, y)` of the vector pair :math:`(x, y)` is computed as :math:`x^T W y / d^s`,
    where `d` and `s` are vector dimension and scaling factor respectively.
    :math:`x` and :math:`y` can be concatenated with bias terms.

    Args:
        n_in (int):
            The size of the input feature.
        n_out (int):
            The number of output channels.
        scale (float):
            Factor to scale the scores. Default: 0.
        bias_x (bool):
            If ``True``, adds a bias term for tensor :math:`x`. Default: ``True``.
        bias_y (bool):
            If ``True``, adds a bias term for tensor :math:`y`. Default: ``True``.
    """

    def __init__(self, n_in, n_out=1, scale=0, bias_x=True, bias_y=True, init='zero'):
        super().__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.scale = scale
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.weight = nn.Parameter(torch.Tensor(n_out, n_in+bias_x, n_in+bias_y))
        self.init = init

        self.reset_parameters()

    def __repr__(self):
        s = f"n_in={self.n_in}"
        if self.n_out > 1:
            s += f", n_out={self.n_out}"
        if self.scale != 0:
            s += f", scale={self.scale}"
        if self.bias_x:
            s += f", bias_x={self.bias_x}"
        if self.bias_y:
            s += f", bias_y={self.bias_y}"
        if self.init:
            s += f", initialization={self.init}"

        return f"{self.__class__.__name__}({s})"

    def reset_parameters(self):
        if self.init == 'zero':
            nn.init.zeros_(self.weight)
        elif self.init == 'normal':
            nn.init.normal_(self.weight, 0., 1.)
        else:
            raise ValueError

    def forward(self, x, y):
        r"""
        Args:
            x (torch.Tensor): ``[batch_size, seq_len, n_in]``.
            y (torch.Tensor): ``[batch_size, seq_len, n_in]``.

        Returns:
            ~torch.Tensor:
                A scoring tensor of shape ``[batch_size, n_out, seq_len, seq_len]``.
                If ``n_out=1``, the dimension for ``n_out`` will be squeezed automatically.
        """

        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
        # [batch_size, n_out, seq_len, seq_len]
        s = torch.einsum('bxi,oij,byj->boxy', x, self.weight, y)
        # remove dim 1 if n_out == 1
        s = s.squeeze(1) / self.n_in ** self.scale

        return s


class Triaffine(nn.Module):
    r"""
    Triaffine layer for second-order scoring :cite:`zhang-etal-2020-efficient,wang-etal-2019-second`.

    This function has a tensor of weights :math:`W` and bias terms if needed.
    The score :math:`s(x, y, z)` of the vector triple :math:`(x, y, z)` is computed as :math:`x^T z^T W y / d^s`,
    where `d` and `s` are vector dimension and scaling factor respectively.
    :math:`x` and :math:`y` can be concatenated with bias terms.

    Args:
        n_in (int):
            The size of the input feature.
        n_out (int):
            The number of output channels.
        scale (float):
            Factor to scale the scores. Default: 0.
        bias_x (bool):
            If ``True``, adds a bias term for tensor :math:`x`. Default: ``False``.
        bias_y (bool):
            If ``True``, adds a bias term for tensor :math:`y`. Default: ``False``.
        decompose (bool):
            If ``True``, represents the weight as the product of 3 independent matrices. Default: ``False``.
    """

    def __init__(self, n_in, n_out=1, scale=0, bias_x=False, bias_y=False, decompose=False, init='zero'):
        super().__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.scale = scale
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.decompose = decompose
        self.init = init

        if not decompose:
            self.weight = nn.Parameter(torch.Tensor(n_out, n_in+bias_x, n_in, n_in+bias_y))
        else:
            self.weight = nn.ParameterList((nn.Parameter(torch.Tensor(n_out, n_in+bias_x)),
                                            nn.Parameter(torch.Tensor(n_out, n_in)),
                                            nn.Parameter(torch.Tensor(n_out, n_in+bias_y))))

        self.reset_parameters()

    def __repr__(self):
        s = f"n_in={self.n_in}"
        if self.n_out > 1:
            s += f", n_out={self.n_out}"
        if self.scale != 0:
            s += f", scale={self.scale}"
        if self.bias_x:
            s += f", bias_x={self.bias_x}"
        if self.bias_y:
            s += f", bias_y={self.bias_y}"
        if self.decompose:
            s += f", decompose={self.decompose}"
        if self.init:
            s += f", initialization={self.init}"

        return f"{self.__class__.__name__}({s})"

    def reset_parameters(self):
        if self.init =='zero':
            if self.decompose:
                for i in self.weight:
                    nn.init.zeros_(i)
            else:
                nn.init.zeros_(self.weight)
        elif self.init == 'normal':
            if self.decompose:
                for i in self.weight:
                    nn.init.normal_(i, 0., 0.25)
            else:
                nn.init.normal_(self.weight, 0., 0.25)
        else:
            raise ValueError
                
    def forward(self, x, y, z):
        r"""
        Args:
            x (torch.Tensor): ``[batch_size, seq_len, n_in]``.
            y (torch.Tensor): ``[batch_size, seq_len, n_in]``.
            z (torch.Tensor): ``[batch_size, seq_len, n_in]``.

        Returns:
            ~torch.Tensor:
                A scoring tensor of shape ``[batch_size, n_out, seq_len, seq_len, seq_len]``.
                If ``n_out=1``, the dimension for ``n_out`` will be squeezed automatically.
        """

        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
        if self.decompose:
            wx = torch.einsum('bxi,oi->box', x, self.weight[0])
            wz = torch.einsum('bzk,ok->boz', z, self.weight[1])
            wy = torch.einsum('byj,oj->boy', y, self.weight[2])
            # [batch_size, n_out, seq_len, seq_len, seq_len]
            s = torch.einsum('box,boz,boy->bozxy', wx, wz, wy)
        else:
            w = torch.einsum('bzk,oikj->bozij', z, self.weight)
            # [batch_size, n_out, seq_len, seq_len, seq_len]
            s = torch.einsum('bxi,bozij,byj->bozxy', x, w, y)
        # remove dim 1 if n_out == 1
        s = s.squeeze(1) / self.n_in ** self.scale

        return s
