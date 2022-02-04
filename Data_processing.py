from ctypes import sizeof
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import pandas as pd

#  Need a method here to load all image/data in
# load image need to have:
# img_train, mask_train, img_test
    
# metadata loc: ~/code/Segmentation/metadata.csv
# image data under column: 'CT_image_path'; corresponding lung mask column: 'lung_mask_path'

rows = 512
cols = 512

# class dataProcess(object):
#     ## Initialization ##
#     def __init__(self, rows, cols):
#         self.rows = rows
#         self.cols = cols
#         # self.data_path = data_path
    
def create_train_data():
    ## get metadata ##
    metadata_path = '/home/zitiantang/code/Segmentation/metadata.csv'
    metadata_df = pd.read_csv(metadata_path, index_col=0)
    
    ## Initialize list for images and corresponding masks
    image_train = []
    lung_mask_train = []

    ## Loop through metadata ##
    for i in range(metadata_df.shape[0]):
        curr_row = metadata_df.iloc[i]
        image_path = curr_row['CT_image_path']
        isTrain = curr_row['is_Train']
        if '20-03-24' in image_path:
            if isTrain:
                # mod_im = np.load(image_path)
                # mod_im = mod_im[:,:,None]
                # image_train.append(mod_im)
                image_train.append(np.load(image_path))
                # mod_msk = np.load(curr_row['lung_mask_path'])
                # mod_msk = mod_msk[:,:,None]
                # lung_mask_train.append(mod_msk)
                lung_mask_train.append(np.load(curr_row['lung_mask_path']))
    
    ## Image modification ##
    # image_train = image_train.astype('float32')
    # image_train /= 255
    # lung_mask_train = lung_mask_train.astype('float32')
    # lung_mask_train /= 255

    ## Return ##
    # return tuple(image_train), tuple(lung_mask_train)
    
    ## Create big nd arrays ##
    imgs = np.ndarray((len(image_train), rows, cols), dtype=np.uint8)
    lung_masks = np.ndarray((len(lung_mask_train), rows, cols), dtype=np.uint8)
    for idx, img in enumerate(image_train):
        imgs[idx, :, :] = img
    for idx, img in enumerate(lung_mask_train):
        lung_masks[idx, :, :] = img
    
    np.save('temp_npy_file/imgs_train.npy', imgs)
    np.save('temp_npy_file/lung_mask_train.npy', lung_masks)
    
def load_train_data():
    image_train = np.load('temp_npy_file/imgs_train.npy')
    lung_mask_train = np.load('temp_npy_file/lung_mask_train.npy')
    return image_train, lung_mask_train


def create_test_data():
    ## get metadata ##
    metadata_path = '/home/zitiantang/code/Segmentation/metadata.csv'
    metadata_df = pd.read_csv(metadata_path, index_col=0)

    ## Initialize list for testing images ##
    image_test = []

    ## Loop through metadata ##
    for i in range(metadata_df.shape[0]):
        curr_row = metadata_df.iloc[i]
        image_path = curr_row['CT_image_path']
        isTrain = curr_row['is_Train']
        if '20-03-24' in image_path:
            if ~isTrain:
                # mod_im = np.load(image_path)
                # mod_im = mod_im[:,:,None]
                # image_test.append(mod_im)
                image_test.append(np.load(image_path))
    
    ## Image modification ##
    # image_test = image_test.astype('float32')
    # image_test /= 255

    ## Return ##
    # return tuple(image_test)

    ## Create big nd array ##
    imgtest = np.ndarray((len(image_test), rows, cols), dtype=np.uint8)
    for idx, img in enumerate(image_test):
        imgtest[idx, :, :] = img
    np.save('temp_npy_file/imgs_test.npy', imgtest)

def load_test_data():
    image_test = np.load('temp_npy_file/imgs_test.npy')
    return image_test

if __name__ == "__main__":
    # mydata = dataProcess(512,512)
    # i_t, l_m_t = mydata.load_train_data()
    # i_te = mydata.load_test_data()
    # print(len(i_t))
    # # print(i_t[1].shape)
    # print(len(i_te[1]))
    # print(i_te[1].shape)
    create_train_data()
    create_test_data()