#####
#Data augmentation on the image datasets:
import cv2
import torch
import tensorflow as tf
import os
os.chdir('/Users/Rhq/Desktop/uc davis/ECS 269/Project/crack_pic_part1/test')
myDir='/Users/Rhq/Desktop/uc davis/ECS 269/Project/crack_pic_part1/test'
imagelist=os.listdir(myDir)
#Add salt_pepper noise:
for i in range(len(imagelist)):
	current_image=cv2.imread(imagelist[i])
	height,width,channel=current_image.shape


#See whether our model can generalize to other domains.
python 'Running.py' --in_dir Salt_pepper001 --meta_file /Users/Rhq/Desktop/'uc davis'/'ECS 269'/Project/crack_pic_part1/model_complete.meta --CP_dir /Users/Rhq/Desktop/'uc davis'/'ECS 269'/Project/crack_pic_part1 --save_dir Result_Salt_pepper_2

#Add brightness:


#Try Gaussian Blurring:
for i in range(5):
	current_image=cv2.imread(imagelist[i])
	processed_image=cv2.GaussianBlur(current_image,(35,35),5)
	save_file='/Users/Rhq/Desktop/gaussian_filter/'+str(i)+'.png'
	cv2.imwrite(save_file,processed_image)

for i in range(6,len(imagelist)):
	current_image=cv2.imread(imagelist[i])
	processed_image=cv2.GaussianBlur(current_image,(35,35),5)
	save_file='/Users/Rhq/Desktop/gaussian_filter/'+str(i)+'.png'
	cv2.imwrite(save_file,processed_image)

#Try  smoothing:
kernel_smoothing=np.ones((8,8),np.float32)/64
for i in range(5):
	current_image=cv2.imread(imagelist[i])
	processed_image=cv2.filter2D(current_image,-1,kernel_smoothing)
	save_file='/Users/Rhq/Desktop/L1_median_filtering/'+str(i)+'.png'
	cv2.imwrite(save_file,processed_image)
for i in range(6,len(imagelist)):
	current_image=cv2.imread(imagelist[i])
	processed_image=cv2.filter2D(current_image,-1,kernel_smoothing)
	save_file='/Users/Rhq/Desktop/L1_median_filtering/'+str(i)+'.png'
	cv2.imwrite(save_file,processed_image)

#Try salt-pepper noise:
s_vs_p=0.5
amount=0.005

for i in range(5):
	current_image=cv2.imread(imagelist[i],0)
	out=current_image
	num_salt=np.ceil(amount*current_image.size*s_vs_p)
	coords=[np.random.randint(0,i-1,int(num_salt)) for i in current_image.shape]
	out[coords]=1
	num_pepper=np.ceil(amount*current_image.size*(1-s_vs_p))
	coords=[np.random.randint(0,i-1,int(num_pepper)) for i in current_image.shape]
	out[coords]=0
	save_file='/Users/Rhq/Desktop/Salt_pepper001/'+str(i)+'.png'
	cv2.imwrite(save_file,out)

for i in range(6,len(imagelist)):
	current_image=cv2.imread(imagelist[i],0)
	out=current_image
	num_salt=np.ceil(amount*current_image.size*s_vs_p)
	coords=[np.random.randint(0,i-1,int(num_salt)) for i in current_image.shape]
	out[coords]=1
	num_pepper=np.ceil(amount*current_image.size*(1-s_vs_p))
	coords=[np.random.randint(0,i-1,int(num_pepper)) for i in current_image.shape]
	out[coords]=0
	save_file='/Users/Rhq/Desktop/Salt_pepper001/'+str(i)+'.png'
	cv2.imwrite(save_file,out)

#Try gamma correction:
def adjust_gamma(image,gamma=1.0):
	invGamma=1.0/gamma
	table=np.array([((i/255.0)**invGamma)*255 for i in np.arange(0,256)]).astype("uint8")
	return cv2.LUT(image,table)

for i in range(5):
	current_image=cv2.imread(imagelist[i])
	processed_image=adjust_gamma(current_image,gamma=0.6)
	save_file='/Users/Rhq/Desktop/test_gamma_correction_0.6/'+str(i)+'.png'
	cv2.imwrite(save_file,processed_image)

for i in range(6,len(imagelist)):
	current_image=cv2.imread(imagelist[i])
	processed_image=adjust_gamma(current_image,gamma=0.6)
	save_file='/Users/Rhq/Desktop/test_gamma_correction_0.6/'+str(i)+'.png'
	cv2.imwrite(save_file,processed_image)
#Try vary the lightness:

#Try 