import torch
import torch.nn as nn
import torch.nn.functional as F


class lossAV(nn.Module):
	def __init__(self):
		super(lossAV, self).__init__()
		self.criterion = nn.BCELoss()
		self.FC        = nn.Linear(128, 2)
		
	def forward(self, x, isInference=False):
		x = x.squeeze(1)
		x = self.FC(x)
		predScore = x[:, 1]  # Get the score for the positive class -> shape (B, N)
		if isInference:
			predScore = predScore.squeeze(0).detach().cpu().numpy()
		return predScore


class lossV(nn.Module):
	def __init__(self):
		super(lossV, self).__init__()
		self.criterion = nn.BCELoss()
		self.FC        = nn.Linear(128, 2)

	def forward(self, x, isInference=False):
		x = x.squeeze(1)
		x = self.FC(x)
		predScore = x[:, 1]  # Get the score for the positive class -> shape (B, N)
		predScore = predScore
		if isInference:
			predScore = predScore.squeeze(0).detach().cpu().numpy()
		return predScore
