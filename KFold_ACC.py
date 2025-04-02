import numpy as np

# Select dataset
dataset = ['A', 'B', 'C','D']
dataset_id = 3
print(dataset[dataset_id])

# Select model
models = ['fNIRS-T', 'fNIRS-PreT']
models_id = 0
print(models[models_id])


test_acc = []
for tr in range(1, 17):
    # /data1/zxj_log/save/D/KFold/cross/dep79andhea17/fNIRS-T
    # /data1/zxj_log/save/D/KFold/cross/dep79andhea17/dep79andhea17_allchannel_dropout0.5/fNIRS-T
    # path = '/data1/zxj_log/save/' + dataset[dataset_id] + '/KFold/augmentation/health100_batchsize64/' + models[models_id] + '/' + str(tr)
    path = '/data1/zxj_log/save/D/KFold/1086_GELU/'+ models[models_id] + '/' + str(tr)
    # /data1/zxj_log/save/C/KFold/augmentation/crossentropy/health_and_not
    # /data1/zxj_log/save/D/KFold/multifocal
    # /data1/zxj_log/save/D/KFold/augmentation/health100/fNIRS-T
    acc = open(path + '/test_acc.txt', "r")
    acc = acc.read()
    acc = float(acc)
    test_acc.append(acc)

test_acc = np.array(test_acc)
print('mean = %.2f' % np.mean(test_acc))
print('std = %.2f' % np.std(test_acc))