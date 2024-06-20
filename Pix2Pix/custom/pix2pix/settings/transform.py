# import albumentations as A
# from albumentations.pytorch import ToTensorV2
import torchvision.transforms as transforms
# https://albumentations-demo.herokuapp.com/

def getTransform():

    train_x_transform= transforms.Compose([ 
        transforms.Resize((256, 256)),    
        transforms.RandomCrop((256,256)),                                                                                                                                                        
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
     ])    

    train_y_transform= transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((256,256)),   
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
        
     ])    

    val_x_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((256,256)),   
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
  

     ])    

    val_y_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((256,256)),   
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
     ])    
    

    return train_x_transform , train_y_transform


def getInferenceTransform():
    
    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        
     ])    
    
    return    test_transform