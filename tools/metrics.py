import math

import torch


class Metrics:
	def __init__(self):
		self.metrics = {'MSE': mse, 'MAE': mae}
		self.total = {}.fromkeys(self.metrics.keys(), .0)
		self.batch = 0

	def update(self, pred, y):
		for k, metric in self.metrics.items():
			self.total[k] += metric(pred, y)
		self.batch += 1

	def clear(self):
		self.total = {}.fromkeys(self.metrics.keys(), .0)
		self.batch = 0

	@property
	def status(self):
		return {
			'MAE': self.total['MAE'] / self.batch,
			'MSE': self.total['MSE'] / self.batch,
			'RMSE': math.sqrt(self.total['MSE'] / self.batch)
		}


def mae(pred, y):
	return torch.abs(pred - y).mean().item()


def mse(pred, y):
	return torch.mean((pred - y) ** 2).item()
