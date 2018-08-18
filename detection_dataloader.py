import numpy as np
import cv2

import copy
import os
import glob

import xml.etree.ElementTree as ET
import pickle

##########################################################################################
##########################################################################################
#VOC data parse
def parse_voc_annotation(ann_dir, img_dir, cache_name, labels=[]):
    if os.path.exists(cache_name):
        with open(cache_name, 'rb') as handle:
            cache = pickle.load(handle)
        all_insts, seen_labels = cache['all_insts'], cache['seen_labels']
    else:
        all_insts = []
        seen_labels = {}
        
        for ann in sorted(os.listdir(ann_dir)):
            img = {'object':[]}

            try:
                tree = ET.parse(ann_dir + ann)
            except Exception as e:
                print(e)
                print('Ignore this bad annotation: ' + ann_dir + ann)
                continue
            
            for elem in tree.iter():
                if 'filename' in elem.tag:
                    img['filename'] = img_dir + elem.text
                if 'width' in elem.tag:
                    img['width'] = int(elem.text)
                if 'height' in elem.tag:
                    img['height'] = int(elem.text)
                if 'object' in elem.tag or 'part' in elem.tag:
                    obj = {}
                    
                    for attr in list(elem):
                        if 'name' in attr.tag:
                            obj['name'] = attr.text

                            if obj['name'] in seen_labels:
                                seen_labels[obj['name']] += 1
                            else:
                                seen_labels[obj['name']] = 1
                            
                            if len(labels) > 0 and obj['name'] not in labels:
                                break
                            else:
                                img['object'] += [obj]
                                
                        if 'bndbox' in attr.tag:
                            for dim in list(attr):
                                if 'xmin' in dim.tag:
                                    obj['xmin'] = int(round(float(dim.text)))
                                if 'ymin' in dim.tag:
                                    obj['ymin'] = int(round(float(dim.text)))
                                if 'xmax' in dim.tag:
                                    obj['xmax'] = int(round(float(dim.text)))
                                if 'ymax' in dim.tag:
                                    obj['ymax'] = int(round(float(dim.text)))

            if len(img['object']) > 0:
                all_insts += [img]

        cache = {'all_insts': all_insts, 'seen_labels': seen_labels}
        #print(cache)
        with open(cache_name, 'wb') as handle:
            pickle.dump(cache, handle, protocol=pickle.HIGHEST_PROTOCOL)    
                        
    return all_insts, seen_labels
def create_training_instances(
    train_annot_folder,
    train_image_folder,
    train_cache,
    valid_annot_folder,
    valid_image_folder,
    valid_cache,
    labels,
):
    # parse annotations of the training set
    train_ints, train_labels = parse_voc_annotation(train_annot_folder, train_image_folder, train_cache, labels)

    # parse annotations of the validation set, if any, otherwise split the training set
    if os.path.exists(valid_annot_folder):
        valid_ints, valid_labels = parse_voc_annotation(valid_annot_folder, valid_image_folder, valid_cache, labels)
    else:
        print("valid_annot_folder not exists. Spliting the trainining set.")

        train_valid_split = int(0.8*len(train_ints))
        np.random.shuffle(train_ints)

        valid_ints = train_ints[train_valid_split:]
        train_ints = train_ints[:train_valid_split]

    # compare the seen labels with the given labels in config.json
    if len(labels) > 0:
        overlap_labels = set(labels).intersection(set(train_labels.keys()))

        print('Seen labels: \t\t'  + str(train_labels))
        print('Given labels: \t\t' + str(labels))
        print('Overlap labels: \t' + str(list(overlap_labels)))

        # return None, None, None if some given label is not in the dataset
        if len(overlap_labels) < len(labels):
            print('Some labels have no annotations! Please revise the list of labels in the config.json.')
            return None, None, None
    else:
        print('No labels are provided. Train on all seen labels.')
        print(train_labels)
        labels = train_labels.keys()

    return train_ints, valid_ints, sorted(labels)
'''
train_ints, valid_ints, labels = create_training_instances(
        'F:\\Learning\\keras\\yolov3\\raccoon_dataset\\annotations\\',
        'F:\\Learning\\keras\\yolov3\\raccoon_dataset\\images\\',
        'test',
        '','','',
        ['raccoon']
    )
'''
##########################################################################################
##########################################################################################
#my data parse

##########################################################################################
##########################################################################################
#generator
class BatchGenerator():
    def __init__(self, 
        instances, 
        anchors,   
        labels,        
        downsample=32, # ratio between network input's size and network output's size, 32 for YOLOv3
        max_box_per_image=30,
        batch_size=1,
        min_net_size=224,
        max_net_size=320,    
        shuffle=True, 
        jitter=True, 
        norm=None
    ):
        self.instances          = instances
        self.batch_size         = batch_size
        self.labels             = labels
        self.downsample         = downsample
        self.max_box_per_image  = max_box_per_image
        self.min_net_size       = (min_net_size//self.downsample)*self.downsample
        self.max_net_size       = (max_net_size//self.downsample)*self.downsample
        self.shuffle            = shuffle
        self.jitter             = jitter
        self.norm               = norm
        self.anchors            = [BoundBox(0, 0, anchors[2*i], anchors[2*i+1]) for i in range(len(anchors)//2)]
        self.net_h              = 320  
        self.net_w              = 320

        if shuffle: np.random.shuffle(self.instances)
        
        self.epoch_count = 0
        self.in_epoch_batch_count = 0
    def next(self):
        while True:
            input_list,dummy_yolo = self.__getitem__(self.in_epoch_batch_count)
            self.in_epoch_batch_count += 1
            if self.in_epoch_batch_count*self.batch_size>self.__len__():
                #self.in_epoch_batch_count = 0
                #if self.shuffle: np.random.shuffle(self.instances)
                #print('------------------------------next epoch------------------------------')
                break
            yield input_list,dummy_yolo
        
    def __len__(self):
        return int(np.ceil(float(len(self.instances))))           

    def __getitem__(self, idx):
        # get image input size, change every 10 batches
        net_h, net_w = self._get_net_size(idx)
        base_grid_h, base_grid_w = net_h//self.downsample, net_w//self.downsample

        # determine the first and the last indices of the batch
        l_bound = idx*self.batch_size
        r_bound = (idx+1)*self.batch_size

        if r_bound > len(self.instances):
            r_bound = len(self.instances)
            l_bound = r_bound - self.batch_size

        x_batch = np.zeros((r_bound - l_bound, net_h, net_w, 3),dtype=np.float32)             # input images
        t_batch = np.zeros((r_bound - l_bound, 1, 1, 1,  self.max_box_per_image, 4),dtype=np.float32)   # list of groundtruth boxes
        
        anchors_batch = np.zeros((r_bound - l_bound,2*len(self.anchors)),dtype=np.float32)

        # initialize the inputs and the outputs
        yolo_1 = np.zeros((r_bound - l_bound, 1*base_grid_h,  1*base_grid_w, len(self.anchors)//3, 4+1+len(self.labels)),dtype=np.float32) # desired network output 1
        yolo_2 = np.zeros((r_bound - l_bound, 2*base_grid_h,  2*base_grid_w, len(self.anchors)//3, 4+1+len(self.labels)),dtype=np.float32) # desired network output 2
        yolo_3 = np.zeros((r_bound - l_bound, 4*base_grid_h,  4*base_grid_w, len(self.anchors)//3, 4+1+len(self.labels)),dtype=np.float32) # desired network output 3
        yolos = [yolo_3, yolo_2, yolo_1]

        dummy_yolo_1 = np.zeros((r_bound - l_bound, 1),dtype=np.float32)
        dummy_yolo_2 = np.zeros((r_bound - l_bound, 1),dtype=np.float32)
        dummy_yolo_3 = np.zeros((r_bound - l_bound, 1),dtype=np.float32)
        
        instance_count = 0
        true_box_index = 0

        # do the logic to fill in the inputs and the output
        for train_instance in self.instances[l_bound:r_bound]:
            # augment input image and fix object's position and size
            img, all_objs,anchors = self._aug_image(train_instance, net_h, net_w,self.anchors)
            
            for i,anchor in enumerate(anchors):
                anchors_batch[instance_count,2*i] = anchor.xmax
                anchors_batch[instance_count,2*i+1] = anchor.ymax
            
            #print('{} objs in this instance'.format(len(all_objs)))
            for i,obj in enumerate(all_objs):
                # find the best anchor box for this object
                max_anchor = None                
                max_index  = -1
                max_iou    = -1
                print(obj)
                #######################################################
                #choose max iou anchor
                shifted_box = BoundBox(0, 
                                       0,
                                       obj['xmax']-obj['xmin'],                                                
                                       obj['ymax']-obj['ymin'])    
                #print('{} obj in this objs'.format(i))
                #print('obj x,y min max:')
                #print(shifted_box.ymin,shifted_box.xmin,shifted_box.ymax,shifted_box.xmax)
                for j in range(len(anchors)):
                    anchor = anchors[j]
                    iou    = bbox_iou(shifted_box, anchor)
                    #print('{} anchor iou {} with {} obj'.format(j,iou,i))
                    if max_iou < iou:
                        max_anchor = anchor
                        max_index  = j
                        max_iou    = iou                
                #print('max_anchor:{},max_index:{},max_iou:{}'.format([max_anchor.ymin,max_anchor.xmin,max_anchor.ymax,max_anchor.xmax],max_index,max_iou))
                # determine the yolo to be responsible for this bounding box
                yolo = yolos[max_index//(len(self.anchors)//3)]
                #######################################################
                grid_h, grid_w = yolo.shape[1:3]
                #print('yolo{} size:{}'.format(max_index//3,[grid_h, grid_w]))
                # determine the position of the bounding box on the grid
                center_x = .5*(obj['xmin'] + obj['xmax'])
                center_x = (center_x / float(net_w)) * grid_w # sigma(t_x) + c_x
                center_y = .5*(obj['ymin'] + obj['ymax'])
                center_y = (center_y / float(net_h)) * grid_h # sigma(t_y) + c_y
                
                # determine the sizes of the bounding box
                w = np.log((obj['xmax'] - obj['xmin']) / float(max_anchor.xmax)) # t_w
                h = np.log((obj['ymax'] - obj['ymin']) / float(max_anchor.ymax)) # t_h

                box = [center_x, center_y, w, h]
                
                # determine the index of the label
                obj_indx = self.labels.index(obj['name'])  

                # determine the location of the cell responsible for this object
                grid_x = int(np.floor(center_x))
                grid_y = int(np.floor(center_y))
                #print('{} obj in yolo{} have box {}'.format(i,max_index//3,box))
                # assign ground truth x, y, w, h, confidence and class probs to y_batch
                yolo[instance_count, grid_y, grid_x, max_index%3, 0:4] = box
                yolo[instance_count, grid_y, grid_x, max_index%3, 4  ] = 1.
                yolo[instance_count, grid_y, grid_x, max_index%3, 5+obj_indx] = 1
                
                # assign the true box to t_batch
                true_box = [center_x, center_y, obj['xmax'] - obj['xmin'], obj['ymax'] - obj['ymin']]
                t_batch[instance_count, 0, 0, 0, true_box_index] = true_box

                true_box_index += 1
                true_box_index  = true_box_index % self.max_box_per_image    

            # assign input image to x_batch
            if self.norm != None: 
                #x_batch[instance_count] = self.norm(img).astype(np.float32)
                x_batch[instance_count] = img.astype(np.float32)
            else:
                # plot image and bounding boxes for sanity check
                for obj in all_objs:
                    cv2.rectangle(img, (obj['xmin'],obj['ymin']), (obj['xmax'],obj['ymax']), (255,0,0), 3)
                    cv2.putText(img, obj['name'], 
                                (obj['xmin']+2, obj['ymin']+12), 
                                0, 1.2e-3 * img.shape[0], 
                                (0,255,0), 2)
                
                x_batch[instance_count] = img

            # increase instance counter in the current batch
            instance_count += 1                 
                
        return [x_batch, anchors_batch,t_batch, yolo_1, yolo_2, yolo_3], [dummy_yolo_1, dummy_yolo_2, dummy_yolo_3]

    def _get_net_size(self, idx):
        if idx%10 == 0:
            net_size = self.downsample*np.random.randint(self.min_net_size/self.downsample, \
                                                         self.max_net_size/self.downsample+1)
            print("resizing: ", net_size, net_size)
            self.net_h, self.net_w = net_size, net_size
        return self.net_h, self.net_w
    
    def _aug_image(self, instance, net_h, net_w,anchors):
        def _constrain(min_v, max_v, value):
            if value < min_v: return min_v
            if value > max_v: return max_v
            return value
        
        image_name = instance['filename']
        image = cv2.imread(image_name)[:,:,::-1] # RGB image
        
        if image is None: print('Cannot find ', image_name)
            
        image_h, image_w, _ = image.shape
        
        im_sized = cv2.resize(image, (net_w, net_h))
        
        # randomly flip
        flip = np.random.randint(2)
        im_sized = random_flip(im_sized, flip)#随机翻转
        
        boxes = copy.deepcopy(instance['object'])

        # randomize boxes' order
        np.random.shuffle(boxes)

        # correct sizes and positions
        sx, sy = float(net_w)/image_w, float(net_h)/image_h
        zero_boxes = []

        for i in range(len(boxes)):
            boxes[i]['xmin'] = int(_constrain(0, net_w, boxes[i]['xmin']*sx))
            boxes[i]['xmax'] = int(_constrain(0, net_w, boxes[i]['xmax']*sx))
            boxes[i]['ymin'] = int(_constrain(0, net_h, boxes[i]['ymin']*sy))
            boxes[i]['ymax'] = int(_constrain(0, net_h, boxes[i]['ymax']*sy))

            if boxes[i]['xmax'] <= boxes[i]['xmin'] or boxes[i]['ymax'] <= boxes[i]['ymin']:
                zero_boxes += [i]
                continue

            if flip == 1:
                swap = boxes[i]['xmin'];
                boxes[i]['xmin'] = net_w - boxes[i]['xmax']
                boxes[i]['xmax'] = net_w - swap

        boxes = [boxes[i] for i in range(len(boxes)) if i not in zero_boxes]
        
        _anchors = copy.deepcopy(anchors)
        for i,anchor in enumerate(anchors):
            _anchors[i].xmin = int(_constrain(0, net_w,anchor.xmin*sx))
            _anchors[i].xmax = int(_constrain(0, net_w,anchor.xmax*sx))
            _anchors[i].ymin = int(_constrain(0, net_h,anchor.ymin*sy))
            _anchors[i].ymax = int(_constrain(0, net_h,anchor.ymax*sy))
        return im_sized, boxes, _anchors
    '''
    def _aug_image(self, instance, net_h, net_w,anchors):
        image_name = instance['filename']
        image = cv2.imread(image_name)[:,:,::-1] # RGB image
        
        if image is None: print('Cannot find ', image_name)
            
        image_h, image_w, _ = image.shape
        
        # determine the amount of scaling and cropping
        dw = self.jitter * image_w;
        dh = self.jitter * image_h;

        new_ar = (image_w + np.random.uniform(-dw, dw)) / (image_h + np.random.uniform(-dh, dh));
        scale = np.random.uniform(0.25, 2);

        if (new_ar < 1):
            new_h = int(scale * net_h);
            new_w = int(net_h * new_ar);
        else:
            new_w = int(scale * net_w);
            new_h = int(net_w / new_ar);
        
        dx = int(np.random.uniform(0, net_w - new_w));
        dy = int(np.random.uniform(0, net_h - new_h));
        
        # apply scaling and cropping
        im_sized = apply_random_scale_and_crop(image, new_w, new_h, net_w, net_h, dx, dy)#尺度并补0或剪切到网络尺寸，dx,dy补到或剪切图像左侧和上侧
        
        # randomly distort hsv space
        #im_sized = random_distort_image(im_sized)#随机改变像素
        
        # randomly flip
        flip = np.random.randint(2)
        im_sized = random_flip(im_sized, flip)#随机翻转
            
        # correct the size and pos of bounding boxes
        all_objs,anchors = correct_bounding_boxes(instance['object'],anchors, new_w, new_h, net_w, net_h, dx, dy, flip, image_w, image_h)#把box调整适应调整后的图像
        
        return im_sized, all_objs, anchors
    '''
    def on_epoch_end(self):
        if self.shuffle: np.random.shuffle(self.instances)
            
    def num_classes(self):
        return len(self.labels)

    def size(self):
        return len(self.instances)    

    def load_annotation(self, i):
        annots = []

        for obj in self.instances[i]['object']:
            annot = [obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax'], self.labels.index(obj['name'])]
            annots += [annot]

        if len(annots) == 0: annots = [[]]

        return np.array(annots)

    def load_image(self, i):
        return cv2.imread(self.instances[i]['filename'])  
#function
def normalize(image):
    return image/255.
def apply_random_scale_and_crop(image, new_w, new_h, net_w, net_h, dx, dy):
    im_sized = cv2.resize(image, (new_w, new_h))
    
    if dx > 0: 
        im_sized = np.pad(im_sized, ((0,0), (dx,0), (0,0)), mode='constant', constant_values=127)
    else:
        im_sized = im_sized[:,-dx:,:]
    if (new_w + dx) < net_w:
        im_sized = np.pad(im_sized, ((0,0), (0, net_w - (new_w+dx)), (0,0)), mode='constant', constant_values=127)
               
    if dy > 0: 
        im_sized = np.pad(im_sized, ((dy,0), (0,0), (0,0)), mode='constant', constant_values=127)
    else:
        im_sized = im_sized[-dy:,:,:]
        
    if (new_h + dy) < net_h:
        im_sized = np.pad(im_sized, ((0, net_h - (new_h+dy)), (0,0), (0,0)), mode='constant', constant_values=127)
        
    return im_sized[:net_h, :net_w,:] 
def random_distort_image(image, hue=18, saturation=1.5, exposure=1.5):
    def _rand_scale(scale):
        scale = np.random.uniform(1, scale)
        return scale if (np.random.randint(2) == 0) else 1./scale;
    # determine scale factors
    dhue = np.random.uniform(-hue, hue)
    dsat = _rand_scale(saturation);
    dexp = _rand_scale(exposure);     

    # convert RGB space to HSV space
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype('float')
    
    # change satuation and exposure
    image[:,:,1] *= dsat
    image[:,:,2] *= dexp
    
    # change hue
    image[:,:,0] += dhue
    image[:,:,0] -= (image[:,:,0] > 180)*180
    image[:,:,0] += (image[:,:,0] < 0)  *180
    
    # convert back to RGB from HSV
    return cv2.cvtColor(image.astype('uint8'), cv2.COLOR_HSV2RGB)
def random_flip(image, flip):
    if flip == 1: return cv2.flip(image, 1)
    return image
def correct_bounding_boxes(boxes, anchors,new_w, new_h, net_w, net_h, dx, dy, flip, image_w, image_h):
    def _constrain(min_v, max_v, value):
        if value < min_v: return min_v
        if value > max_v: return max_v
        return value
    boxes = copy.deepcopy(boxes)

    # randomize boxes' order
    np.random.shuffle(boxes)

    # correct sizes and positions
    sx, sy = float(new_w)/image_w, float(new_h)/image_h
    zero_boxes = []

    for i in range(len(boxes)):
        boxes[i]['xmin'] = int(_constrain(0, net_w, boxes[i]['xmin']*sx + dx))
        boxes[i]['xmax'] = int(_constrain(0, net_w, boxes[i]['xmax']*sx + dx))
        boxes[i]['ymin'] = int(_constrain(0, net_h, boxes[i]['ymin']*sy + dy))
        boxes[i]['ymax'] = int(_constrain(0, net_h, boxes[i]['ymax']*sy + dy))

        if boxes[i]['xmax'] <= boxes[i]['xmin'] or boxes[i]['ymax'] <= boxes[i]['ymin']:
            zero_boxes += [i]
            continue

        if flip == 1:
            swap = boxes[i]['xmin'];
            boxes[i]['xmin'] = net_w - boxes[i]['xmax']
            boxes[i]['xmax'] = net_w - swap

    boxes = [boxes[i] for i in range(len(boxes)) if i not in zero_boxes]
    
    _anchors = copy.deepcopy(anchors)
    for i,anchor in enumerate(anchors):
        _anchors[i].xmin = int(_constrain(0, net_w,anchor.xmin*sx))
        _anchors[i].xmax = int(_constrain(0, net_w,anchor.xmax*sx))
        _anchors[i].ymin = int(_constrain(0, net_h,anchor.ymin*sy))
        _anchors[i].ymax = int(_constrain(0, net_h,anchor.ymax*sy))

    return boxes,_anchors
class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, c = None, classes = None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        
        self.c       = c
        self.classes = classes

        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)
        
        return self.label
    
    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]
            
        return self.score  
def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2,x4) - x1
    else:
        if x2 < x3:
             return 0
        else:
            return min(x2,x4) - x3  
def bbox_iou(box1, box2):
    intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])  
    
    intersect = intersect_w * intersect_h

    w1, h1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin
    w2, h2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin
    
    union = w1*h1 + w2*h2 - intersect
    
    return float(intersect) / union
##########################################################################################
##########################################################################################

if __name__=='__main__':
    dataset_root = 'F:\\Learning\\tensorflow\\detect\\Dataset\\'
    
    max_box_per_image = 30
    batch_size = 1
    min_input_size = 224#32*7
    max_input_size = 352#32*11

    train_ints, valid_ints, labels = create_training_instances(
        dataset_root+'VOC2012\\Annotations\\',
        dataset_root+'VOC2012\\JPEGImages\\',
        'data.pkl',
        '','','',
        ['person','head','hand','foot','aeroplane','tvmonitor','train','boat','dog','chair',
         'bird','bicycle','bottle','sheep','diningtable','horse','motorbike','sofa','cow',
         'car','cat','bus','pottedplant']
    )
    
    train_generator = BatchGenerator(
        instances           = train_ints, 
        anchors             = [18,27, 28,75, 49,132, 55,43, 65,227, 84,86, 108,162, 109,288, 162,329, 174,103, 190,212, 245,348, 321,150, 343,256, 372,379],   
        labels              = labels,        
        downsample          = 32, # ratio between network input's size and network output's size, 32 for YOLOv3
        max_box_per_image   = max_box_per_image,
        batch_size          = batch_size,
        min_net_size        = min_input_size,
        max_net_size        = max_input_size,   
        shuffle             = True, 
        jitter              = 0.3, 
        norm                = normalize
    )
    
    count = 0
    for input_list,dummy_yolo in train_generator.next():
        x_batch, anchors_batch,t_batch, yolo_1, yolo_2, yolo_3 = input_list
        print(yolo_1.shape)
        
        image = x_batch[0,:,:,:].astype(np.uint8)[:,:,::-1]
        cv2.namedWindow('test',cv2.WINDOW_NORMAL)
        cv2.imshow('test',image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        break
