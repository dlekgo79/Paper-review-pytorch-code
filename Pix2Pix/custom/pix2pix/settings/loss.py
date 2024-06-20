from torch import nn

def getLoss():

	loss_gan = nn.BCELoss()
	loss_l1 = nn.L1Loss()
	
	return loss_gan,loss_l1