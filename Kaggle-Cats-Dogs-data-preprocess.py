import os, shutil

orig_data_dir = 'D:\Deep-learning\\test-scripts\dogs-vs-cats\\train'
bas_dir = 'D:\Deep-learning\\test-scripts'

train_dir = os.path.join(bas_dir,'train')
print(train_dir)
os.mkdir(train_dir)
val_dir = os.path.join(bas_dir,'val')
print(val_dir)
os.mkdir(val_dir)
test_dir = os.path.join(bas_dir,'test')
print(test_dir)
os.mkdir(test_dir)

def data_split(dir, clas, range_l, range_r):
    class_dir = os.path.join(dir,clas)
    os.mkdir(class_dir)
    fnames = [clas+'.{}.jpg'.format(i) for i in range(range_l,range_r)]
    print(len(fnames))
    for fname in fnames:
        src = os.path.join(orig_data_dir,fname)
        dst = os.path.join(class_dir,fname)
        shutil.copyfile(src,dst)
    return

data_split(train_dir,'cat',0,1000)
data_split(train_dir,'dog',0,1000)

data_split(val_dir,'cat',1000,1500)
data_split(val_dir,'dog',1000,1500)

data_split(test_dir,'cat',1500,2000)
data_split(test_dir,'dog',1500,2000)