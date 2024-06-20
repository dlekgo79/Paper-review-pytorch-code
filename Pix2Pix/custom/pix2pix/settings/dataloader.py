from torch.utils.data.dataloader import DataLoader

# collate_fn needs for batch
def collate_fn(batch):
    return tuple(zip(*batch))

def getDataloader(trainDataset, validDataset, batch):

	trainDataloader = DataLoader(trainDataset, batch_size=batch, shuffle=True)
	validDataloader = DataLoader(validDataset, batch_size=batch, shuffle=False)

	return trainDataloader, validDataloader

def getInferenceDataloader(dataset, batch):

	return DataLoader(dataset, batch_size=batch, shuffle= False)