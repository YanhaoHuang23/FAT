import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
class FANLayer(nn.Module):
    """
    FANLayer: The layer used in FAN (https://arxiv.org/abs/2410.02675).
    
    Args:
        input_dim (int): The number of input features.
        output_dim (int): The number of output features.
        p_ratio (float): The ratio of output dimensions used for cosine and sine parts (default: 0.25).
        activation (str or callable): The activation function to apply to the g component. If a string is passed,
            the corresponding activation from torch.nn.functional is used (default: 'gelu').
        use_p_bias (bool): If True, include bias in the linear transformations of p component (default: True). 
            There is almost no difference between bias and non-bias in our experiments.
    """
    
    def __init__(self, input_dim, output_dim, p_ratio=0.25, activation='gelu', use_p_bias=True):
        super(FANLayer, self).__init__()
        
        # Ensure the p_ratio is within a valid range
        assert 0 < p_ratio < 0.5, "p_ratio must be between 0 and 0.5"

        self.p_ratio = p_ratio
        p_output_dim = int(output_dim * self.p_ratio)
        g_output_dim = output_dim - p_output_dim * 2  # Account for cosine and sine terms
        """

        # 保持线性部分的参数量不变
        self.p_ratio = p_ratio
        p_output_dim = int(output_dim * self.p_ratio)
        g_output_dim = output_dim  # g 部分保持原始 output_dim
        new_output_dim = g_output_dim + 2 * p_output_dim  # 最终输出为扩展后的维度
        """
        # Linear transformation for the p component (for cosine and sine parts)
        self.input_linear_p = nn.Linear(input_dim, p_output_dim, bias=use_p_bias)
        
        # Linear transformation for the g component
        self.input_linear_g = nn.Linear(input_dim, g_output_dim)
        
        # Set the activation function
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

        # Apply Kaiming initialization
        self._initialize_weights()

    def _initialize_weights(self):
        # Kaiming initialization for linear layers
        init.kaiming_uniform_(self.input_linear_p.weight, nonlinearity='relu')
        init.kaiming_uniform_(self.input_linear_g.weight, nonlinearity='relu')
        if self.input_linear_p.bias is not None:
            init.zeros_(self.input_linear_p.bias)
        if self.input_linear_g.bias is not None:
            init.zeros_(self.input_linear_g.bias)

    def forward(self, src):
        """
        Args:
            src (Tensor): Input tensor of shape (batch_size, input_dim).
        
        Returns:
            Tensor: Output tensor of shape (batch_size, output_dim), after applying the FAN layer.
        """
        
        # Apply the linear transformation followed by the activation for the g component
        g = self.activation(self.input_linear_g(src))
        
        # Apply the linear transformation for the p component
        p = self.input_linear_p(src)
        
        # Concatenate cos(p), sin(p), and activated g along the last dimension
        output = torch.cat((torch.cos(p), torch.sin(p), g), dim=-1)
        
        return output

