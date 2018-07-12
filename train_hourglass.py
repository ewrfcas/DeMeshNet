import Hourglass
from keras.models import *
from keras.layers import *
from keras.optimizers import *
import os
import pickle
from keras.callbacks import *
from keras.utils import multi_gpu_model
from skimage import io,transform

os.environ["CUDA_VISIBLE_DEVICES"] = '0,2'

with open('dataset/train_dict.pkl','rb') as f:
    train_dict = pickle.load(f)
with open('dataset/test_dict.pkl','rb') as f:
    test_dict = pickle.load(f)

data_path = 'dataset/person_meshed_low/'
base_path = 'dataset/person/'
X_train = [];mask_train = [];y_train = []
X_test = [];mask_test = [];y_test = []
for p in os.listdir(data_path):
    if '_m.png' in p:
        continue
    name=p.split('_')[0]+'.jpg'
    mask=p.split('.jpg')[0]+'_m.png'
    if name in train_dict:
        X_train.append(p)
        mask_train.append(mask)
        y_train.append(name)
    else:
        X_test.append(p)
        mask_test.append(mask)
        y_test.append(name)

X_train=np.array(list(map(lambda x:data_path+x,X_train)))
mask_train=np.array(list(map(lambda x:data_path+x,mask_train)))
y_train=np.array(list(map(lambda x:base_path+x,y_train)))
X_test=np.array(list(map(lambda x:data_path+x,X_test)))
mask_test=np.array(list(map(lambda x:data_path+x,mask_test)))
y_test=np.array(list(map(lambda x:base_path+x,y_test)))
print(X_train.shape,mask_train.shape,y_train.shape)
print(X_test.shape,mask_train.shape,y_test.shape)

def generate_mesh(X_list, mask_list, y_list, batch_size=32, shuffle=True):
    while True:
        count=0
        x, mask, y = [],[],[]
        if shuffle:
            random_index=np.arange(X_list.shape[0])
            np.random.shuffle(random_index)
            X_list=X_list[random_index]
            mask_list=mask_list[random_index]
            y_list=y_list[random_index]
        for i,path in enumerate(X_list):
            img = io.imread(path)
            img = transform.resize(img,(224,176))
            img = (img - 0.5) * 2
            img_mask = io.imread(mask_list[i])
            img_mask = transform.resize(img_mask, (224, 176))
            y_temp = io.imread(y_list[i])
            y_temp = transform.resize(y_temp,(224,176))
            y_temp = (y_temp - 0.5) * 2
            x.append(img)
            mask.append(img_mask)
            y.append(y_temp)
            count+=1
            if count == batch_size:
                x = np.array(x)
                x = np.reshape(x,(batch_size, 224, 176, 3))
                mask = np.array(mask)
                mask = np.reshape(mask, (batch_size, 224, 176, 1))
                y = np.array(y)
                y = np.reshape(y, (batch_size, 224, 176, 3))
                yield [x, mask], [y, mask * y, y, mask * y]
                x, mask, y = [], [], []
                count = 0

epoch_num=20
batch_size=32
batch_size_test=32
model=Hourglass.model()
model=multi_gpu_model(model,gpus=2)
model.summary()
early_stop = EarlyStopping(monitor='loss', patience=3, verbose=1, mode='auto')
check_point= ModelCheckpoint('model/Hourglass_weightsV2.h5', monitor='loss',verbose=0, save_best_only=True,\
                             save_weights_only=True, mode='auto', period=1)
# def schedule(epoch):
#     # 动态调整
#     return 0.0005*(0.98**epoch)
# learning_rate=LearningRateScheduler(schedule)
optimizer = Adam(lr=0.0005, beta_1=0.9, beta_2=0.99, epsilon=1e-08)
model.compile(loss='mse', optimizer=optimizer)

model.fit_generator(generate_mesh(X_train,mask_train,y_train,batch_size),len(X_train)//batch_size,\
                    epochs=epoch_num,\
                    validation_data=generate_mesh(X_test,mask_test,y_test,batch_size_test,shuffle=False),\
                    validation_steps=len(X_test)//batch_size_test,\
                    callbacks=[early_stop,check_point])