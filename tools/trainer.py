import json
import os
from datetime import datetime

import torch
from torch.nn import MSELoss
from torch.optim import Adam
from tqdm import tqdm

from tools.data import create_data_loaders
from tools.metrics import Metrics
from tools.model import create_model


class ASTGCNTrainer:
	def __init__(self, settings: dict):
		self.device = torch.device(settings['device'])
		self.saved_dir = settings['saved_dir']
		self.epochs, lr = settings['epochs'], settings['learn_rate']
		self.metrics, self.history = Metrics(), []
		# load
		print('Loading data...')
		self.data_loaders, self.normalizers = create_data_loaders(**settings)
		# creat
		print('Creating model...')
		self.model = create_model(**settings)
		self.optimizer = Adam(self.model.parameters(), lr=lr)
		self.criterion = MSELoss().to(self.device)

	def run(self):
		self.history.clear()
		# train
		print('Training...')
		for epoch in range(self.epochs):
			print(f"EPOCH: {epoch + 1}")
			self.history.append({
				'train': self.train_epoch(),
				'validate': self.validate_epoch()
			})
		self.save()

	def train_epoch(self):
		total_loss, average_loss = .0, .0
		device, data_loader = self.device, self.data_loaders['train']
		with tqdm(total=len(data_loader), desc='TRAINING', unit='batches') as bar:
			for i, xy in enumerate(data_loader):
				xy = [t.to(device) for t in xy]
				x, y = xy[:-1], xy[-1]
				self.optimizer.zero_grad()
				pred = self.model(*x)
				loss = self.criterion(pred, y)
				loss.backward()
				self.optimizer.step()
				# update statistics
				total_loss += loss.item() / len(pred)
				average_loss = total_loss / (i + 1)
				# update progress bar
				bar.update()
				bar.set_postfix(loss=f'{average_loss:.2f}')
		return {'loss': average_loss}

	@torch.no_grad()
	def validate_epoch(self):
		self.model.eval()
		self.metrics.clear()
		total_loss, average_loss = .0, .0
		device, data_loader = self.device, self.data_loaders['validate']
		with tqdm(total=len(data_loader), desc='VALIDATING', unit='batches') as bar:
			for i, xy in enumerate(data_loader):
				xy = [t.to(device) for t in xy]
				x, y = xy[:-1], xy[-1]
				pred = self.model(*x)
				loss = self.criterion(pred, y)
				# update statistics
				self.metrics.update(pred, y)
				total_loss += loss.item() / len(pred)
				average_loss = total_loss / (i + 1)
				# update progress bar
				bar.update()
				bar.set_postfix(**{
					k: f'{v:.2f}' for k, v in self.metrics.status.items()
				}, loss=f'{average_loss:.2f}')
		self.model.train()
		return {'loss': average_loss, 'metrics': self.metrics.status}

	def save(self, history=True, normalizers=False, model=False):
		if not os.path.exists(self.saved_dir):
			os.mkdir(self.saved_dir)
		saved_dir = os.path.join(self.saved_dir, datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
		if not os.path.exists(saved_dir):
			os.mkdir(saved_dir)
		if history:
			# save history
			with open(f'{saved_dir}/history.json', 'w') as f:
				f.write(json.dumps(self.history))
		if normalizers:
			# save normalizers
			torch.save(self.normalizers, f'{saved_dir}/normalizers.pth')
		if model:
			# save model
			torch.save({'model': self.model.state_dict()}, f'{saved_dir}/model.pkl')
