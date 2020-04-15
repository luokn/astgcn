import json

import torch


def load_settings(name):
	with open(f'./settings/{name}.json') as f:
		return json.loads(f.read())


def norm_adj_matrix(adj_filepath, n_vertices, device='cpu'):
	A = torch.eye(n_vertices, device=device)
	for ln in open(adj_filepath, 'r').readlines()[1:]:
		i, j, _ = ln.split(',')
		i, j = int(i), int(j)
		A[i, j] = A[j, i] = 1

	D_norm = torch.diag(A.sum(1) ** (-0.5))
	A_norm = D_norm @ A @ D_norm
	return A_norm
