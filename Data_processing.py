from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import pandas as pd

#  Need a method here to load all image/data in
# load image need to have:
# img_train, mask_train, img_test
    
# metadata loc: ~/code/Segmentation/metadata.csv
# image data under column: 'CT_image_path'; corresponding lung mask column: 'lung_mask_path'

class dataProcess(object):
    ## Initialization ##
    def __init__(self, rows, cols, data_path):
        self.rows = rows
        self.cols = cols
        self.data_path = data_path
    
    def load_train_data(self):
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
            if '20-03-24' in image_path & isTrain:
                image_train.append(np.load(image_path))
                lung_mask_train.append(np.load(curr_row['lung_mask_path']))
        
        ## Image modification ##
        image_train = image_train.astype('float32')
        image_train /= 255
        lung_mask_train = lung_mask_train.astype('float32')
        lung_mask_train /= 255

        ## Return ##
        return image_train, lung_mask_train

    def load_test_data(self):
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
            if '20-03-24' in image_path & ~isTrain:
                image_test.append(np.load(image_path))
        
        ## Image modification ##
        image_test = image_test.astype('float32')
        image_test /= 255

        ## Return ##
        return image_test

if __name__ == "__main__":
    mydata = dataProcess(512,512)