import tensorflow as tf

import numpy as np

from MnasNet import MnasNet as Model
from MnasNet import Detection_or_Classifier

from detection_dataloader import create_training_instances
from detection_dataloader import BatchGenerator as DetectionBatchGenerator
from ImagenetDataloader import BatchGenerator as ClassifierBatchGenerator

import os
import shutil

NumEpoch = 1000
StartEpoch = 0
BestScore = 0
DetectFirstTrain = True
PrintFreq = 10
BatchSize = 96
Mode = 'Train'

def Main():
    
    sess,model = EnvironmentSetup()
    sess,model = LoadModel(sess,model)

    train_loader = ClassifierBatchGenerator(root=self.dataset_root+'scene_photos',batch_size=BatchSize)
    test_loader = ClassifierBatchGenerator(root=self.dataset_root+'scene_photos',batch_size=BatchSize)

    if Mode=='Val':
        _ = Test(sess, model, test_loader)
        return
    elif Mode=='Test':
        _ = Test(sess, model, test_loader)
        return
    else:
        for epoch in range(StartEpoch,NumEpoch):
            sess, model = Train(sess, model, train_loader, epoch)
            
            top1 = Test(sess, model, test_loader, epoch)
            
            SaveModel(sess, model, epoch, top1)

def Train(sess, model, dataloder,epoch):
    batch = 0

    if Detection_or_Classifier=='classifier':
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        for X_batch, y_batch in dataloder.next():
            feed_dict = {model.input_image: X_batch,
                         model.y: y_batch,
                         model.is_training: True}
                    
            _, loss, acc, acc_5 = sess.run(
                            [model.train_op,model.all_loss,model.accuracy,model.accuracy_top_5],
                            feed_dict=feed_dict)
            losses.update(loss, BatchSize)
            top1.update(acc, BatchSize)
            top5.update(acc_5, BatchSize)

            if batch % PrintFreq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'top_1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'top_5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, batch, dataloder.len()//BatchSize, loss=losses, top1=top1, top5=top5))
            batch += 1

    elif Detection_or_Classifier=='detection':
        losses = AverageMeter()
        for input_list,dummy_yolo in dataloder.next():

            x_batch, anchors_batch,t_batch, yolo_1, yolo_2, yolo_3 = input_list

            print('hide image')
            S_ratio = 32
            S = x_batch.shape[1]/S_ratio
            for _batch in range(x_batch.shape[0]):
                IndexList = []
                for _ in range(S_ratio*S_ratio):
                    IndexList.append(np.random.randint(0,2))
                IndexList = np.array(IndexList).reshape(S_ratio,S_ratio).tolist()
                for i in range(S_ratio):
                    for j in range(S_ratio):
                        if IndexList[i][j]==1:
                            pass
                        else:
                            x_batch[_batch,int(S*i):int(S*(i+1)),int(S*j):int(S*(j+1)),:] = 0.0
            print('hide success')
                    
            feed_dict = {model.input_image:x_batch,
                         model.anchors:anchors_batch,
                         model.is_training:True,
                         model.true_boxes:t_batch,
                         model.true_yolo_1:yolo_1,
                         model.true_yolo_2:yolo_2,
                         model.true_yolo_3:yolo_3}

            _, loss = sess.run(
                        [model.train_op, model.all_loss],
                        feed_dict=feed_dict)
            losses.update(loss, BatchSize)
            if batch % PrintFreq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   epoch, batch, dataloder.len()//BatchSize, loss=losses))
            batch += 1
    return sess, model

def Test(sess, model, dataloder, epoch):
    def draw_boxes(self,image, boxes, labels, obj_thresh=0.8):
        def _constrain(min_v, max_v, value):
            if value < min_v: return min_v
            if value > max_v: return max_v
            return value
        
        for box in boxes:
            label_str = ''
            
            max_idx = -1
            max_score = 0
            for i in range(len(labels)):

                if box[5:][i] > obj_thresh:
                    label_str += labels[i]

                    if box[5:][i]>max_score:
                        max_score = box[5:][i]
                        max_idx = i
                    print(labels[i] + ': ' + str(box[5:][i]*100) + '%')
                    
            if max_idx >= 0:
                cv2.rectangle(image, (int(_constrain(1,image.shape[1]-2,box[0])),int(_constrain(1,image.shape[0]-2,box[1]))), 
                                     (int(_constrain(1,image.shape[1]-2,box[2])),int(_constrain(1,image.shape[0]-2,box[3]))), (0,0,255), 3)
                cv2.putText(image, 
                            labels[max_idx] + ' ' + str(max(box[5:])), 
                            (int(box[0]), int(box[1]) - 13), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, 
                            (0,0,255), 2)
        return image

    if Detection_or_Classifier=='classifier':
        batch = 0
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        for X_batch, y_batch in dataloder.next():
            feed_dict = {model.input_image: X_batch,
                         model.y: y_batch,
                         model.is_training: False}
                    
            loss, acc, acc_5 = sess.run(
                            [model.all_loss,model.accuracy,model.accuracy_top_5],
                            feed_dict=feed_dict)
            losses.update(loss, BatchSize)
            top1.update(acc, BatchSize)
            top5.update(acc_5, BatchSize)

            if batch % PrintFreq == 0:
                print('Test: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'top_1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'top_5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, batch, dataloder.len()//BatchSize, loss=losses, top1=top1, top5=top5))
            batch += 1
        return top1.avg
    elif Detection_or_Classifier=='detection':
        labels = None
                
        if not os.path.exists(os.path.join(os.getcwd(),'test_results',Detection_or_Classifier)):
            os.makedirs(os.path.join(os.getcwd(),'test_results',Detection_or_Classifier))
            
        for image_path in glob.glob(os.path.join(os.getcwd(),'test_images',Detection_or_Classifier,'*.jpg')):
            image_name = image_path.split('/')[-1]
            print('processing image {}'.format(image_name))

            image = cv2.imread(image_path)
            image_h,image_w,_ = image.shape
            _image = cv2.resize(image,(32*10,32*10))[np.newaxis,:,:,::-1]
            infos = sess.run(model.infos,
                                          feed_dict={model.input_image:_image,
                                                     model.is_training:False,
                                                     model.original_wh:[[image_w,image_h]]
                                                     }
                                          )
            infos = model.do_nms(infos,0.3)
            image = draw_boxes(image, infos.tolist(), labels)
            cv2.imwrite(os.path.join(os.getcwd(),'test_results',Detection_or_Classifier,image_name),image)
    

def LoadModel(sess,model):
    path = os.path.join(os.getcwd(),'model_store',Detection_or_Classifier)

    var = tf.global_variables()
    var_list = [val for val in var]

    if Detection_or_Classifier=='detection' and DetectFirstTrain:
        var_list = [val for val in var if 'zsc_classifier' not in val.name]
        saver = tf.train.Saver(var_list=var_list)

        ClassifierBestCheckpoint = os.path.join(os.getcwd(),'model_store','classifier','classifier_best')
        if not os.path.exists(ClassifierBestCheckpoint+'.data-00000-of-00001'):
            raise ValueError('when first train detection model, train classifier model first, unless you use my model that can train from zero in my github')
        else:
            print("Loading classifier model checkpoint from {} ...\n".format(ClassifierBestCheckpoint))
            saver.restore(sess, ClassifierBestCheckpoint)
            print("Checkpoint loaded\n")
    else:
        saver = tf.train.Saver(var_list=var_list)
        if os.path.exists(os.path.join(path,Detection_or_Classifier+'_best.data-00000-of-00001')):
            print("Loading {} model checkpoint from {} ...\n".format(Detection_or_Classifier,path))
            saver.restore(sess, os.path.join(path,Detection_or_Classifier+'_best'))
            print("Checkpoint loaded\n")
        else:
            print("First time to train!\n")

    global StartEpoch
    global BestScore
    StartEpoch = model.global_epoch_tensor.eval(sess)
    BestScore = model.BestScore_tensor.eval(sess)
    print('StartEpoch:{}, BestScore:{}'.format(StartEpoch,BestScore))
    return sess, model

def SaveModel(sess, model, epoch, score):
    global BestScore
    
    is_best = score>BestScore

    model.global_epoch_assign_op.eval(session=sess,feed_dict={model.global_epoch_input: epoch})
    if is_best:
        model.BestScore_assign_op.eval(session=sess,feed_dict={model.BestScore_input: score})

    var = tf.global_variables()
    var_list = [val for val in var]
    saver = tf.train.Saver(var_list=var_list)
    
    checkpoints_path = os.path.join(os.getcwd(),'model_store',Detection_or_Classifier)
    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path)
    print("Saving a checkpoint")
    saver.save(sess, os.path.join(checkpoints_path,Detection_or_Classifier))
    print("Checkpoint Saved\n")

    if is_best:
        BestScore = score
        shutil.copyfile(os.path.join(checkpoints_path,Detection_or_Classifier+'.data-00000-of-00001'), os.path.join(checkpoints_path,Detection_or_Classifier+'_best.data-00000-of-00001'))
        shutil.copyfile(os.path.join(checkpoints_path,Detection_or_Classifier+'.index'), os.path.join(checkpoints_path,Detection_or_Classifier+'_best.index'))
        shutil.copyfile(os.path.join(checkpoints_path,Detection_or_Classifier+'.meta'), os.path.join(checkpoints_path,Detection_or_Classifier+'_best.meta'))
        print('save new best model')
    
def EnvironmentSetup():
    num_classes = 10
    
    tf.reset_default_graph()
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    print("Building the model...")
    model = Model(num_classes=num_classes)
    print("Model is built successfully\n")

    sess.run(tf.group(tf.global_variables_initializer()))

    num_params = get_num_params()
    print('all params:{}\n'.format(num_params))

    return sess, model

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

from functools import reduce
from operator import mul
def get_num_params():
    num_params = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        num_params += reduce(mul, [dim.value for dim in shape], 1)
    return num_params

if __name__=='__main__':
    Main()
