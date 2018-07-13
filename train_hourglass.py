import Hourglass
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

def cal_ETA(t_start, i, n_batch):
    t_temp = time.time()
    t_avg = float(int(t_temp) - int(t_start)) / float(i + 1)
    if n_batch - i - 1 > 0:
        return int((n_batch - i - 1) * t_avg)
    else:
        return int(t_temp)-int(t_start)

def data_generate(X_list, mask_list, y_list, nstack):
        x, mask, y = [], [], []
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

        x = np.array(x)
        x = np.reshape(x,(batch_size, 224, 176, 3))
        mask = np.array(mask)
        mask = np.reshape(mask, (batch_size, 224, 176, 1))
        y = np.array(y)
        y = np.reshape(y, (batch_size, 224, 176, 3))
        return [x, mask], [y, mask * y] * nstack

# hyper-parameter
epoch_num=20
batch_size=32
batch_size_test=32
patience = 2

nstack=2
level=4
filter=128

with tf.device('/cpu:0'):
    model=Hourglass.model(input_shape=(224, 176, 3), nstack=nstack, level=level, module=1, filters=filter)

model.summary()
parallel_model = multi_gpu_model(model,gpus=2)

optimizer = Adam(lr=0.0005, beta_1=0.9, beta_2=0.99, epsilon=1e-08)
parallel_model.compile(loss='mse', optimizer=optimizer)

# train on batch
n_batch = len(X_train)//batch_size
n_batch_val = len(X_test)//batch_size
lower_count = 0
best_loss = 999.

for epoch in range(epoch_num):
    total_loss = [0] * nstack
    t_start = time.time()
    last_train_str = "\r"
    random_index = np.arange(X_train.shape[0])
    np.random.shuffle(random_index)
    X_train = X_train[random_index]
    mask_train = mask_train[random_index]
    y_train = y_train[random_index]

    # train
    for i in range(10):#n_batch
        x_temp, y_temp = data_generate(X_train[i*batch_size:(i+1)*batch_size],
                                       mask_train[i*batch_size:(i+1)*batch_size],
                                       y_train[i*batch_size:(i+1)*batch_size],
                                       nstack = nstack)
        loss_value = parallel_model.train_on_batch(x_temp, y_temp)
        for s in range(nstack):
            total_loss[s] += (loss_value[2*s+1]+loss_value[2*s+2])
        last_train_str = "\r[epoch:%d/%d, steps:%d/%d] -ETA: %ds -loss1:%.3f -loss2:%.3f" % (
            epoch + 1, epoch_num, i + 1, n_batch, cal_ETA(t_start, i, n_batch), total_loss[0] / (i + 1), total_loss[1] / (i + 1))
        print(last_train_str, end='     ', flush=True)

    # validate
    total_loss = [0] * nstack
    for i in range(10):#n_batch_val
        x_temp, y_temp = data_generate(X_test[i*batch_size:(i+1)*batch_size],
                                       mask_test[i*batch_size:(i+1)*batch_size],
                                       y_test[i*batch_size:(i+1)*batch_size],
                                       nstack = nstack)
        loss_value = parallel_model.train_on_batch(x_temp, y_temp)
        for s in range(nstack):
            total_loss[s] += (loss_value[2*s+1]+loss_value[2*s+2])
        last_val_str = last_train_str + "  [validate:%d/%d] -loss1:%.3f -loss2:%.3f" % (
            i + 1, n_batch_val, total_loss[0] / (i + 1), total_loss[1] / (i + 1))
        print(last_val_str, end='      ', flush=True)
    val_loss = min(total_loss[0] / n_batch_val, total_loss[1] / n_batch_val)
    print('\n')
    if val_loss < best_loss:
        lower_count = 0
        best_loss = val_loss
        model.save('model/Hourglass_modelsV3_epoch+' + str(epoch + 1) + '.h5')
    else:
        lower_count +=1
        if lower_count>=patience:
            print('stopped with early stopping...')
            break