import numpy
import torch
from torch.utils.data import DataLoader, TensorDataset


def generate_sequences(data, tp):
	n_timesteps, n_vertices, n_channels = data.shape
	n_sequences = n_timesteps - tp - 168 * tp
	# generate
	ranges = [(-tp, 0), (-2 * tp, -tp), (-3 * tp, -2 * tp), (-24 * tp, -23 * tp), (-168 * tp, -167 * tp)]
	Y = torch.zeros(n_sequences, n_vertices, tp, device=data.device)
	X = torch.zeros(len(ranges), n_sequences, n_vertices, n_channels, tp, device=data.device)
	for i, t in enumerate(range(168 * tp, n_timesteps - tp)):
		for x, (start, end) in zip(X, ranges):
			x[i] = data[t + start:t + end].permute(1, 2, 0)
		Y[i] = data[t:t + tp, :, 0].T
	return X, Y


def normalize_sequences(X, split):
	def normalize(x):
		std = torch.std(x[:split], dim=0, keepdim=True)
		mean = torch.mean(x[:split], dim=0, keepdim=True)
		x -= mean
		x /= std
		return dict(std=std, mean=mean)

	normalizers = [normalize(x) for x in X]
	return normalizers


def create_data_loaders(data_filepath, points_per_hour, batch_size, train_split, **kwargs):
	assert 0 < train_split < 1
	data = torch.from_numpy(numpy.load(data_filepath)['data'])
	data = data.to(kwargs['data_device'])
	X, Y = generate_sequences(data, points_per_hour)
	split = int(len(Y) * train_split)
	normalizers = normalize_sequences(X, split)
	dataset1 = TensorDataset(*[x[:split] for x in X], Y[:split])  # for train
	dataset2 = TensorDataset(*[x[split:] for x in X], Y[split:])  # for validate
	data_loaders = {
		"train": DataLoader(dataset1, batch_size=batch_size, shuffle=True),
		"validate": DataLoader(dataset2, batch_size=batch_size, shuffle=True)
	}
	return data_loaders, normalizers
