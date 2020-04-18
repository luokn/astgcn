import torch
from torch.nn import init

from model.astgcn import ASTGCN
from .utils import norm_adj_matrix


def create_model(adj_filepath, points_per_hour, n_predictions, n_vertices, **kwargs):
	device = torch.device(kwargs['device'])
	A = norm_adj_matrix(adj_filepath, n_vertices, device=device)
	mixin = dict(n_vertices=n_vertices, n_predictions=n_predictions, A=A)
	submodules = [{
		"blocks": [
			{
				"in_channels": 3,
				"in_timesteps": points_per_hour,
				'out_channels': 64,
				'gcn_filters': 64,
				'tcn_strides': 1
			},
			{
				"in_channels": 64,
				"in_timesteps": points_per_hour,
				'out_channels': 64,
				'gcn_filters': 64,
				'tcn_strides': 1,
			}
		]
	}] * 6

	astgcn = ASTGCN(submodules=submodules, **mixin).to(device)
	for name, params in astgcn.named_parameters(recurse=True):
		if params.dim() > 1:
			init.xavier_uniform_(params)
		else:
			init.uniform_(params)
	return astgcn
