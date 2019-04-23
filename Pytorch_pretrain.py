import os
import numpy as np
 
import torch
import torch.nn
import torchvision.models as models
from torch.autograd import Variable 
import torch.cuda
import torchvision.transforms as transforms
 
from PIL import Image
 
import pandas as pd 

Feature_list = []
Format = pd.read_csv('Format.csv')  
img_to_tensor = transforms.ToTensor()

def make_model():
    resmodel=models.resnet34(pretrained=True)
    #resmodel.cuda()#cuda
    return resmodel
 
#classfication
def inference(resmodel,imgpath):
    resmodel.eval()#must
    
    img=Image.open(imgpath)
    img=img.resize((224,224))
    tensor=img_to_tensor(img)
    
    tensor=tensor.resize_(1,3,224,224)
    #tensor=tensor.cuda()#cuda
            
    result=resmodel(Variable(tensor))
    result_npy=result.data.cpu().numpy()
    max_index=np.argmax(result_npy[0])
    
    return max_index#use to print predict
    
#extract_feature
def extract_feature(resmodel,imgpath):
	resmodel.fc=torch.nn.LeakyReLU(0.1)
	resmodel.eval()
	img=Image.open(imgpath)
	img=img.resize((224,224))
	tensor=img_to_tensor(img)
	
	tensor=tensor.resize_(1,3,224,224)
	#tensor=tensor.cuda()#cuda
		
	result=resmodel(Variable(tensor))
	result_npy=result.data.cpu().numpy()
	Feature_list.append(result_npy[0])
	
	return result_npy[0]#use to print feature
    
if __name__=="__main__":
	model=make_model()
	Folder="./test-jpg/Cat/"
	#print (inference(model,imgpath))
	for i in range(0,200):
		imgpath = Folder + "{}.jpg".format(i)
		extract_feature(model, imgpath)
	Folder="./test-jpg/Dog/"
	for i in range(0,200):
		imgpath = Folder + "{}.jpg".format(i)
		extract_feature(model, imgpath)
	ToCsv =  pd.DataFrame(np.array(Feature_list))
	ToCsv = pd.concat([Format, ToCsv], axis = 1)
	ToCsv.to_csv('Feature.csv')