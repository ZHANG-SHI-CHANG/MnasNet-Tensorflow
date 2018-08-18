import numpy as np
import cv2

import os 
import glob
import platform
if platform.system()=='Windows':
    SymSplit = '\\'
else:
    SymSplit = '/'

class BatchGenerator():
    def __init__(self, root, batch_size=1024, shuffle=True, input_size=224, input_channel=3):
        self.root = root
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.ext = ['.jpg', '.png']
        self.input_size = input_size
        self.input_channel = input_channel
        
    def next(self):
        self.prepare()
        self.in_epoch_batch_count = 0
        print('start...')
        while True:
            x_batch, y_batch = self.__getitem__(self.in_epoch_batch_count)
            self.in_epoch_batch_count += 1
            if self.in_epoch_batch_count*self.batch_size>self.len():
                #self.in_epoch_batch_count = 0
                #self.prepare()
                #print('------------------------------next epoch------------------------------')
                print('end')
                break
            yield x_batch, y_batch

    def __getitem__(self, idx):
        l_bound = idx*self.batch_size
        r_bound = (idx+1)*self.batch_size
        if r_bound>self.len():
            r_bound = self.len()
            l_bound = r_bound - self.batch_size
        
        x_batch = np.zeros((self.batch_size,self.input_size,self.input_size,self.input_channel), dtype=np.float32)
        y_batch = np.zeros((self.batch_size), dtype=np.int32)

        feed_num = 0
        for image_path,image_label in self.dataset[l_bound:r_bound]:
            try:
                image = cv2.imread(image_path)
                
                try:
                    c = image.shape[2]
                    if self.input_channel==1:
                        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                        image = image[:,:,np.newaxis]
                except:
                    image = np.stack([image,image,image],axis=2)
                
                image = cv2.resize(image,(self.input_size, self.input_size))
            except:
                continue

            x_batch[feed_num,:,:,:] = image
            y_batch[feed_num] = image_label
            feed_num += 1

        return x_batch[:feed_num,:,:,:], y_batch[:feed_num]

    def len(self):
        return len(self.dataset)
    def prepare(self):
        self.dataset = []
        self.label_names = []
        label_id = -1
        for folder_path in glob.glob(os.path.join(self.root,'*')):
            label_name = folder_path.split(SymSplit)[-1]
            label_id += 1
            self.label_names.append({label_name: label_id})
            
            for ext in self.ext:
                for image_path in glob.glob(os.path.join(folder_path,'*'+ext)):
                    self.dataset.append((image_path,label_id))

        if self.shuffle:
            np.random.shuffle(self.dataset)

        print('labels: {} \n'.format(self.label_names))
        self.classes = len(self.label_names)
        print('classes: {}\n'.format(len(self.label_names)))


if __name__=='__main__':
    dataloader = BatchGenerator('/data1/ZhangShiChang/TensorflowWork/Classifier/dataset/scene_photos')
    for x_batch,y_batch in dataloader.next():
        print(x_batch.shape)
