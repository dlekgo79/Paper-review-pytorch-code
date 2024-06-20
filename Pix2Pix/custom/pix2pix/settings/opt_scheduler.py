import torch
#https://sanghyu.tistory.com/113
#https://gaussian37.github.io/dl-pytorch-lr_scheduler/
def getOptAndScheduler(model_G,model_D, lr):

	# optimizer = torch.optim.Adam(params = model.parameters(), lr=lr, weight_decay=1e-5)
	# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=10,eta_min=1e-5)

	optimizer_G = torch.optim.Adam(model_G.parameters(), lr=lr, betas=(0.5, 0.999))
	scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_G,T_max=10,eta_min=1e-5)

	optimizer_D= torch.optim.Adam(model_D.parameters(), lr=lr, betas=(0.5, 0.999))
	scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_D,T_max=10,eta_min=1e-5)

	return optimizer_G,optimizer_D