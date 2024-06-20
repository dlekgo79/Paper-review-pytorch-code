import torchvision.models as models
import segmentation_models_pytorch as smp 
from .Model.generator import GeneratorUNet
from .Model.discriminator import PatchDiscriminator
 
def getModel():
	Generator = GeneratorUNet()
	Discriminator = PatchDiscriminator()
	return Generator,Discriminator


    #Segmentation
	# model = smp.Unet(
	# 		encoder_name="resnet18",
	# 		encoder_weights="imagenet",
	# 		in_channels=1,
	# 		classes=3
	# 	)
     
