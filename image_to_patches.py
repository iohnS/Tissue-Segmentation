import cv2
import os
import numpy as np
import datetime
import math
from scipy import ndimage


def cv2_rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

def rotate(image,angle,use_binary=False):
    dim = math.ceil(2*math.sqrt(image.shape[0]**2/2))
    exp = dim-image.shape[0]

    exp_img = np.zeros(shape=(dim,dim,image.shape[2]), dtype=image.dtype)
    s = int(exp/2)
    exp_img[s:s+image.shape[0],s:s+image.shape[1],:] = image

    # Mirroring
    top = image[0:s,:,:]
    top = top[::-1,:,:]
    exp_img[0:s,s:s+image.shape[1],:] = top
    
    bot = image[image.shape[0]-s-1:,:,:]
    bot = bot[::-1,:,:]
    exp_img[image.shape[0]+s:,s:s+image.shape[1],:] = bot

    left = image[:,0:s,:]
    left = left[:,::-1,:]
    exp_img[s:s+image.shape[0],0:s,:] = left

    right = image[:,image.shape[1]-s-1:,:]
    right = right[:,::-1,:]
    exp_img[s:s+image.shape[0],image.shape[1]+s:,:] = right

    if use_binary:
        exp_img = np.uint8(exp_img/255)
    rotated = ndimage.rotate(exp_img,angle,reshape=False)
    #rotated = cv2_rotate_image(exp_img,angle)

    if use_binary:
        cropped = rotated[s:s+image.shape[0],s:s+image.shape[1],:]*255
    else:
        cropped = rotated[s:s+image.shape[0],s:s+image.shape[1],:]

    return cropped

def rotate_and_save(patch,mask,count,savedir):
    pi = 0

    while pi < 359:
        #rotation angle in degree
        rotated = rotate(patch,pi)
        corresp = rotate(mask,pi,use_binary=True)
        corr_labels = get_labels(corresp)

        savepath = os.path.join(savedir,'Patches','patch_' + str(count) + '.png')
        cv2.imwrite(savepath,rotated)
        savepath = os.path.join(savedir,'Masks','mask_' + str(count) + '.png')
        cv2.imwrite(savepath,corresp)
        savepath = os.path.join(savedir,'Labels','labels_' + str(count) + '.png')
        cv2.imwrite(savepath,corr_labels)
        count += 1
        pi += 45
    
    return count

# En mask har 256 element där varje element har en matrix med 256 element. 

def get_labels(rgb_mask):
    label_mask = np.zeros(shape=rgb_mask.shape[:2],dtype='float32')
    for j,label_name in enumerate(gt_color_dict):
        idx = np.where(np.all(gt_color_dict[label_name] == rgb_mask, axis=2))
        label_mask[idx] = float(j)
    return label_mask


local_dir = "/home/john/Documents/FMAN40/h&e_segmentation/"

print("Train or test? (press Enter for test, write anything for train)")
train = input()
imgseg = []
savedir = os.path.join(local_dir, 'train_patches') if train else os.path.join(local_dir, 'test_patches')

if train:
    for i in range(5):
        img = cv2.imread(os.path.join(local_dir,'train', 'images','image_' + str(i) + '.png'))
        gt = cv2.imread(os.path.join(local_dir,'train','gt', 'seg_' + str(i) + '.png'))
        imgseg.append((img, gt))
else:
    for i in range(5, 8):
        img = cv2.imread(os.path.join(local_dir,'test', 'images','image_' + str(i) + '.png'))
        gt = cv2.imread(os.path.join(local_dir,'test','gt', 'seg_' + str(i) + '.png'))
        imgseg.append((img,gt))
    
if not os.path.isdir(savedir):
    os.mkdir(savedir)
if not os.path.isdir(os.path.join(savedir,'Patches')):
    os.mkdir(os.path.join(savedir,'Patches'))
if not os.path.isdir(os.path.join(savedir,'Masks')):
    os.mkdir(os.path.join(savedir,'Masks'))
if not os.path.isdir(os.path.join(savedir,'Labels')):
    os.mkdir(os.path.join(savedir,'Labels'))

for img, gt in imgseg:
    [ydim,xdim] = img.shape[:2]
    
    size_dim = 256
    over_x = int((size_dim-xdim%size_dim)/((xdim-xdim%size_dim)/size_dim))+1
    over_y = int((size_dim-ydim%size_dim)/((ydim-ydim%size_dim)/size_dim))+1

    gt_color_dict = {
        'Background' : [255,255,255],
        'Stroma' : [255,255,0],
        'Cytoplasm' : [255,0,0],
        'Nuclei' : [255,0,255],
    }

    x = 0
    count = 0
    while x < xdim-size_dim:
        y = 0
        while y < ydim-size_dim:
            patch = img[y:y+size_dim,x:x+size_dim]
            mask = gt[y:y+size_dim,x:x+size_dim]
            #cv2.imshow('Patch',patch)
            #cv2.imshow('Mask',mask)
            #cv2.waitKey(0)

            label_mask = get_labels(mask)

            classes = np.unique(label_mask)
            if np.array_equal(classes, [0]):
                print("Empty mask")
                y = y + size_dim - over_y
                continue

            count = rotate_and_save(patch,mask,count,savedir)
            flip_patch = patch[:,::-1,:]
            flip_mask = mask[:,::-1,:]
            count = rotate_and_save(flip_patch,flip_mask,count,savedir)

            y = y + size_dim - over_y
        x = x + size_dim - over_x

    print('saved ' + str(count) + ' patches')
