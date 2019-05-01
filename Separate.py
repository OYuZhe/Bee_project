#Use dataset and json by https://beerys.github.io/CaltechCameraTraps/
#separate 1 folder to 16 categories folder
import os
import shutil
import json


#dump :write json data to file
#load :load json data from file
#dumps:dict to str
#loads:str  to dict

def load():
	#load data
	with open('CaltechCameraTrapsECCV18.json', 'r') as f:
		data = json.load(f)
	return data
	
def mkdir(path):
	#delete space and Last \
	path=path.strip()
	path=path.rstrip("\\")
	isExists=os.path.exists(path)

	if not isExists:
		#print (path,"directory estblished")
		os.makedirs(path)
		return True
	else:
		#print (path,"directory already exists")
		return False
		
if __name__ == '__main__':
	#load json
	print ("Start to Load json")
	MyJson=load()
	print ("Load json Finish")
	'''
	#Output Json
	json_to_str = json.dumps(myjson,sort_keys=False, indent=4, separators=(',', ': '))
	print (json_to_str) #print json
	'''
	#Make dic for categories
	print ("Start to generate Category dict, format: Category[id]= name")
	NumOfCategory = 16
	Category = {} 
	for i in range(0,NumOfCategory):
		id = MyJson['categories'][i]['id']
		name = MyJson['categories'][i]['name']
		Category[id] = name
	print("Generate dict Finish")
	'''
	#Output All Category id and correspond name
	for key in Category:
		print( key, 'corresponds to', Category[key])
	'''
	print("Start to makes directories for all Category in ./image/")
	#the folder that you want to save separate data
	mkFolder = "./image/"
	for key in Category:
		FolderName = mkFolder + Category[key]
		mkdir(FolderName)
	print("Make Finish")
	
	#Count number of images for preparation separate 
	print ("Start to count number of images")
	#the dataset will be separate
	folder = "./eccv_18_cropped/"
	Filenames = os.listdir(folder)
	print('Number of image:', len(Filenames))
	
	#Start separate
	#search image_id in ['images'] and then find the category_id in ['annotations']
	#if the image have multi-categories ,then use the first category it search
	print ("Start to separate image for",NumOfCategory,"Category")
	Annotationslen = len(MyJson['annotations'])
	for i in range(0,len(Filenames)):
		image_id = MyJson['images'][i]['id']
		for j in range(0,Annotationslen):
			if image_id == MyJson['annotations'][j]['image_id']:
				category_id = MyJson['annotations'][j]['category_id']
				category_name = Category[category_id]
				image_location = folder + image_id + ".jpg"
				destination = mkFolder + category_name
				shutil.copy(image_location,destination)
				#shutil.move(image_location,destination)
				#print(i,image_id,category_name)
				break;
	print("Finish")



