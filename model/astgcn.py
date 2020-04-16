import math

import torch
from torch import FloatTensor
from torch.nn import Conv2d, ModuleList, LayerNorm, Module, Parameter, Sequential


class Attention(Module):
	def __init__(self, dk, requires_value=False):
		super(Attention, self).__init__()
		self.sqrt_dk = math.sqrt(dk)
		self.requires_value = requires_value
		self.W1 = Parameter(torch.zeros(dk, 10), requires_grad=True)
		self.W2 = Parameter(torch.zeros(10, dk), requires_grad=True)

	def forward(self, x: FloatTensor):
		x_out = x.reshape(*x.shape[:2], -1)
		# [B * A * Dk] @ [Dk * Dk] @ [B * Dk * A] => [B * A * A]
		att = x_out @ self.W1 @ self.W2 @ x_out.transpose(1, 2)
		att = torch.softmax(att / self.sqrt_dk, dim=-1)
		return (att @ x_out).reshape_as(x) if self.requires_value else att


class GCN(Module):
	def __init__(self, in_channels, in_timesteps, gcn_filters, A):
		super(GCN, self).__init__()
		self.A = A
		# U, S, V = torch.svd(A)
		# self.adaptive_0 = Parameter(U[:, :10] @ S[:10].sqrt().diag(), requires_grad=True)
		# self.adaptive_1 = Parameter(S[:10].sqrt().diag() @ V[:, :10].T, requires_grad=True)
		self.W = Parameter(torch.zeros(in_channels, gcn_filters), requires_grad=True)
		# self.fc = Conv2d(2 * gcn_filters, gcn_filters, kernel_size=1)
		self.s_att = Attention(in_channels * in_timesteps, requires_value=False)

	def forward(self, x: FloatTensor):
		# In : B * V * C_i * T
		# Out: B * V * C_o * T
		att = self.s_att(x)  # => [B * V * V]
		# A_adapt = torch.relu(self.adaptive_0 @ self.adaptive_1)
		# A_adapt = torch.softmax(A_adapt, dim=-1)
		# # A_adapt = torch.dropout(A_adapt, 0.2, self.training)
		x_out = x.permute(3, 0, 1, 2)  # => [T * B * V * C_i]
		# x_out = torch.cat([
		# 	# [B * V * V] @ [T * B * V * C_i] @ [C_i * C_o] => [T * B * V * C_o]
		# 	(att * self.A) @ x_out @ self.W[0],
		# 	(att * A_adapt) @ x_out @ self.W[1]
		# ], dim=-1)  # => [T * B * V * 2C_o]
		# # [T * B * V * 2C_o] => [B * 2C_o * V * T] => [B * C_o * V * T]
		# x_out = self.fc(x_out.permute(1, 3, 2, 0))
		# return x_out.transpose(1, 2)
		x_out = (att * self.A) @ x_out @ self.W
		return x_out.permute(1, 2, 3, 0)


class ASTGCNBlock(Module):
	def __init__(self, in_channels, in_timesteps, out_channels, n_vertices, gcn_filters, tcn_strides, **kwargs):
		super(ASTGCNBlock, self).__init__()
		self.t_att = Attention(n_vertices * in_channels, requires_value=True)
		self.c_att = Attention(n_vertices * in_timesteps, requires_value=True)
		self.gcn = GCN(in_channels, in_timesteps, gcn_filters, kwargs['A'])
		self.tcn = Conv2d(gcn_filters, out_channels, [1, 3], stride=[1, tcn_strides], padding=[0, 1])
		self.res = Conv2d(in_channels, out_channels, [1, 1], stride=[1, tcn_strides])
		self.ln = LayerNorm(normalized_shape=out_channels)

	def forward(self, x: FloatTensor):
		# In : B * V * C_i * T_i
		# Out: B * V * C_o * T_o
		x_res = self.res(x.transpose(1, 2))
		# [B * 1 * C * C] @ [B * V * C * T] => [B * V * C * T]
		x = self.c_att(x.transpose(1, 2)).transpose(1, 2)
		# [B * 1 * T * T] @ [B * V * T * C] => [B * V * T * C]
		x = self.t_att(x.transpose(1, 3)).transpose(1, 3)
		# [B * V * C_i * T] => [B * C_i * V * T] => [B * C' * V * T]
		x = self.gcn(x)
		# [B * C' * V * T] => [B * C_o * V * T]
		x = self.tcn(x.transpose(1, 2)) + x_res
		return self.ln(x.relu_().permute(0, 3, 2, 1)).permute(0, 2, 3, 1)


class ASTGCNSubModule(Module):
	def __init__(self, blocks, **kwargs):
		super(ASTGCNSubModule, self).__init__()
		self.gcn_blocks = Sequential(*[ASTGCNBlock(**block, **kwargs) for block in blocks])
		self.final_conv = Conv2d(in_channels=blocks[-1]['in_timesteps'] // blocks[-1]['tcn_strides'],
								 out_channels=kwargs['n_predictions'],
								 kernel_size=[1, blocks[-1]['gcn_filters']])

	def forward(self, x: FloatTensor):
		# In : B * V * C_i * T_i
		# Out: B * V * C_o * T_o
		x = self.gcn_blocks(x)
		x = self.final_conv(x.permute(0, 3, 1, 2))
		# => (B * Tp * V) -> (B * V * Tp)
		return x[..., 0].transpose(1, 2)


class ASTGCN(Module):
	def __init__(self, submodules, **kwargs):
		super(ASTGCN, self).__init__()
		self.submodules = ModuleList([ASTGCNSubModule(**submodule, **kwargs) for submodule in submodules])
		self.W = Parameter(torch.zeros(len(submodules), kwargs['n_vertices'], kwargs['n_predictions']),
						   requires_grad=True)

	def forward(self, *X):
		return sum(map(lambda fn, x, w: fn(x) * w, self.submodules, X, self.W))
