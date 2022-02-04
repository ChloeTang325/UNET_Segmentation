from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.preprocessing.image import array_to_img
from Data_processing import *
from Model import *

# Implement UNET to generate lung masks for COVID Dataset1
# Image size: (512,512,1); total image num: 100

# class lungMaskUnet(object):
def __init__(self, img_rows=512, img_cols=512):
    self.img_rows = img_rows
    self.img_cols = img_cols

## Load image in ##
# def load_data(self):
#     image_data = dataProcess(self.img_rows, self.img_cols)
#     image_train, lung_mask_train = image_data.load_train_data()
#     image_test = image_data.load_test_data()
#     return image_train, lung_mask_train, image_test


# ## define UNET structure ##
# def test_unet(self):
#     ## Inputs ##
#     inputs = Input((self.img_rows, self.img_cols, 1))
#     s = Lambda(lambda x: x/255)(inputs)

#     ## Contraction path ##
#     c1 = Conv2D(64, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(s)
#     c1 = Dropout(0.1)(c1)
#     c1 = Conv2D(64, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(c1)
#     maxpool1 = MaxPooling2D(pool_size=(2, 2))(c1)


#     c2 = Conv2D(128, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(maxpool1)
#     c2 = Dropout(0.1)(c2)
#     c2 = Conv2D(128, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(c2)
#     maxpool2 = MaxPooling2D(pool_size=(2, 2))(c2)


#     c3 = Conv2D(256, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(maxpool2)
#     c3 = Dropout(0.2)(c3)
#     c3 = Conv2D(256, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(c3)
#     maxpool3 = MaxPooling2D(pool_size=(2, 2))(c3)


#     c4 = Conv2D(512, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(maxpool3)
#     c4 = Dropout(0.3)(c4)
#     c4 = Conv2D(512, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(c4)
#     # drop4 = Dropout(0.5)(conv4)
#     maxpool4 = MaxPooling2D(pool_size=(2, 2))(c4)


#     c5 = Conv2D(1024, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(maxpool4)
#     c5 = Dropout(0.5)(c5)
#     c5 = Conv2D(1024, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(c5)
#     # drop5 = Dropout(0.5)(conv5)


#     ## Expansion path ##
#     # up6 = Conv2D(512, (2,2), activation='relu', padding='same', kernel_initializer='he_normal')(
#     #     UpSampling2D(size=(2, 2))(c5))
#     up6 = Conv2DTranspose(512, (2,2), strides=(2,2), padding='same')(c5)
#     # merge6 = merge([drop4, up6], mode='concat', concat_axis=3)
#     up6 = concatenate([up6, c4])
#     c6 = Conv2D(512, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(up6)
#     c6 = Dropout(0.3)(c6)
#     c6 = Conv2D(512, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(c6)


#     # up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
#     #     UpSampling2D(size=(2, 2))(conv6))
#     up7 = Conv2DTranspose(256, (2,2), strides=(2,2), padding='same')(c6)
#     # merge7 = merge([conv3, up7], mode='concat', concat_axis=3)
#     up7 = concatenate([up7, c3])
#     c7 = Conv2D(256, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(up7)
#     c7 = Dropout(0.2)(c7)
#     c7 = Conv2D(256, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(c7)


#     # up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
#     #     UpSampling2D(size=(2, 2))(conv7))
#     up8 = Conv2DTranspose(128, (2,2), strides=(2,2), padding='same')(c7)
#     # merge8 = merge([conv2, up8], mode='concat', concat_axis=3)
#     up8 = concatenate([up8, c2])
#     c8 = Conv2D(128, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(up8)
#     c8 = Dropout(0.1)(c8)
#     c8 = Conv2D(128, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(c8)


#     # up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
#     #     UpSampling2D(size=(2, 2))(conv8))
#     up9 = Conv2DTranspose(64, (2,2), strides=(2,2), padding='same')(c8)
#     # merge9 = merge([conv1, up9], mode='concat', concat_axis=3)
#     up9 = concatenate([up9, c1], axis=3)
#     c9 = Conv2D(64, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(up9)
#     c9 = Dropout(0.1)(c9)
#     c9 = Conv2D(64, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(c9)
#     # conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

#     ## Output ##
#     outputs = Conv2D(1, (1,1), activation='sigmoid')(c9)

#     model = Model(inputs=[inputs], outputs=[outputs])

#     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#     model.summary()

#     return model

def train():
    # load image
    # image_train, lung_mask_train, image_test = self.load_data()

    ## load training data ##
    image_train, lung_mask_train = load_train_data()
    print("Data loading complete!")
    image_train = image_train[..., None]
    lung_mask_train = lung_mask_train[..., None]
    image_train = image_train.astype('float32')
    lung_mask_train = lung_mask_train.astype('float32')

    # get model
    # model = self.test_unet()
    model = unet() 
    # define checkpoint and callbacks 
    model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='loss', verbose=1, save_best_only=True)
    # callbacks = [EarlyStopping(patience=2, monitor='val_loss'), TensorBoard(log_dir='logs')]
    #fit model according to checkpoint
    model.fit(image_train, lung_mask_train, batch_size=10, epochs=10, verbose=1, validation_split=0.2, shuffle=True, callbacks=[model_checkpoint])
    # predict testing data
    ## load testing data ##
    image_test = load_test_data()
    image_test = image_test[..., None]
    image_test = image_test.astype('float32')

    lung_mask_test = model.predict(image_test, verbose=1)
    # save results as numpy array
    # save_to_path = '/datadrive/COVID_CT_Images/UNET_pred_result_220203/lung_mask_predict_1.npy'
    save_to_path = '/temp_npy_file/lung_mask_test.npy'
    np.save(save_to_path, lung_mask_test)

## Need a method to visualize the predicted testing mask ##
    
if __name__ == '__main__':
    train()