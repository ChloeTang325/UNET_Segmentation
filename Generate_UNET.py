from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import array_to_img
import cv2

# Implement UNET to generate lung masks for COVID Dataset1
# Image size: (512,512,100)

class lungMaskUnet(object):
    def __init__(self, img_rows=512, img_cols=512):
        self.img_rows = img_rows
        self.img_cols = img_cols
    
    # need a method here to load all image/data in
    # load image need to have:
    # img_train, mask_train, img_test
    
    # metadata loc: ~/code/Segmentation/metadata.csv
    # image data under column: 'CT_image_path'; corresponding lung mask column: 'lung_mask_path'

    def test_unet(self):
        inputs = Input((self.img_rows, self.img_cols, 3))

        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        # print "conv1 shape:", conv1.shape
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        # print "conv1 shape:", conv1.shape
        maxpool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        # print "pool1 shape:", pool1.shape

        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(maxpool1)
        # print "conv2 shape:", conv2.shape
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        # print "conv2 shape:", conv2.shape
        maxpool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        # print "pool2 shape:", pool2.shape

        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(maxpool2)
        # print "conv3 shape:", conv3.shape
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        # print "conv3 shape:", conv3.shape
        maxpool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        # print "pool3 shape:", pool3.shape

        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(maxpool3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        maxpool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(maxpool4)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(drop5))
        merge6 = merge([drop4, up6], mode='concat', concat_axis=3)
        # print(up6)
        # print(merge6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        # print(conv6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
        # print(conv6)
        up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv6))
        merge7 = merge([conv3, up7], mode='concat', concat_axis=3)
        # print(up7)
        # print(merge7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        # print(conv7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
        # print(conv7)
        up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv7))
        merge8 = merge([conv2, up8], mode='concat', concat_axis=3)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

        up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv8))
        merge9 = merge([conv1, up9], mode='concat', concat_axis=3)
        # print(up9)
        # print(merge9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        # print(conv9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        # print(conv9)
        conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        # print "conv9 shape:", conv9.shape

        conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)
        # print(conv10)
        model = Model(input=inputs, output=conv10)

        model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

        return model
    
    def train(self):
        # load image
            # Code here
            
        # model train
        model = self.test_unet()
        model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='loss', verbose=1, save_best_only=True)
        #fit model according to checkpoint
        model.fit(img_train, mask_train, batch_size=2, epochs=50, verbose=1, validation_split=0.2, shuffle=True, callbacks=[model_checkpoint])
        # predict testing data
        mask_test = model.predict(img_test, batch_size=1, verbose=1)
        # save results as numpy array
        # np.save([address to be saved])
    
if __name__ == '__main__':
    myunet = lungMaskUnet()
    model = myunet.test_unet()
    myunet.train()
    # save