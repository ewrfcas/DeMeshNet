from SegNet import SegNet
from keras.models import *
from keras.layers import *
from keras.optimizers import *
import os
import pickle
from keras.callbacks import *
from skimage import io,transform

os.environ["CUDA_VISIBLE_DEVICES"] = '2'

with open('dataset/train_dict.pkl','rb') as f:
    train_dict = pickle.load(f)
with open('dataset/test_dict.pkl','rb') as f:
    test_dict = pickle.load(f)

data_path = 'dataset/person_meshed/'
base_path = 'dataset/person/'
X_train = [];mask_train = [];y_train = []
X_test = [];mask_test = [];y_test = []
for p in os.listdir(data_path):
    if '_m.png' in p:
        continue
    name=p.split('_')[0]+'.jpg'
    mask=p.split('.png')[0]+'_m.png'
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
            random_index=np.random.shuffle(np.arange(len(X_list)))
            X_list=X_list[random_index]
            mask_list=mask_list[random_index]
            y_list=y_list[random_index]
        for i,path in enumerate(X_list):
            img = io.imread(path)
            img = transform.resize(img,(220,178))
            img = (img - 0.5) * 2
            img_mask = io.imread(mask_list[i])
            img_mask = transform.resize(img_mask, (220, 178))
            y_temp = io.imread(y_list[i])
            y_temp = transform.resize(y_temp,(220,178))
            y_temp = (y_temp - 0.5) * 2
            x.append(img)
            mask.append(img_mask)
            y.append(y_temp)
            count+=1
            if count == batch_size:
                x = np.array(x)
                x = np.reshape(x,(batch_size, 220, 178, 3))
                mask = np.array(mask)
                mask = np.reshape(mask, (batch_size, 220, 178, 1))
                y = np.array(y)
                y = np.reshape(y, (batch_size, 220, 178, 3))
                yield [x, mask], [y, mask * y]
                x, mask, y = [], [], []
                count = 0

epoch_num=20
batch_size=32
model=SegNet()
early_stop = EarlyStopping(monitor='loss', patience=10, verbose=1, mode='auto')
check_point= ModelCheckpoint('model/SegNet_weights.h5', monitor='val_loss',verbose=0, save_best_only=True,\
                             save_weights_only=True, mode='auto', period=1)
def schedule(epoch):
    # 动态调整
    return 0.0001*(0.98**epoch)
learning_rate=LearningRateScheduler(schedule)
optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.99, epsilon=1e-08)
model.compile(loss='mse', optimizer=optimizer, loss_weights=[1,1])

model.fit_generator(generate_mesh(X_train,mask_train,y_train),len(X_train)//batch_size,epochs=epoch_num, \
                    validation_data=generate_mesh(X_test,mask_test,y_test,shuffle=False),validation_steps=len(X_test)//batch_size,\
                    callbacks=[early_stop,check_point,learning_rate])