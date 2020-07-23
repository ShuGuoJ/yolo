import torch
from torch.utils.data import Dataset
from utils import parse_annotation
from PIL import Image
import numpy as np
from torchvision import transforms

class MyDataset(Dataset):
    def __init__(self, img_dir, ann_dir, labels):
        # img_dir: image path
        # ann_dir: annotation xml file path
        # labels: ('sugarbeet', 'weed')
        super(MyDataset, self).__init__()
        self.imgs_path, self.imgs_boxes = parse_annotation(img_dir, ann_dir, labels)
        self.transform = transforms.ToTensor()
        self.IMGSZ = 512
        self.GRIDSZ = 16
        self.anchors = np.array([0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]).reshape((5,2))
        self.scale = self.IMGSZ / self.GRIDSZ

    def __getitem__(self, index):
        img, boxes = self.transform(Image.open(self.imgs_path[index])) , self.imgs_boxes[index]
        return img, self.process_true_boxes(boxes)

    def __len__(self):
        return len(self.imgs_path)

    def process_true_boxes(self, gt_boxes):
        # gt_boxes:[40,5]
        # mask for object
        detector_mask = np.zeros([self.GRIDSZ, self.GRIDSZ, 5, 1], dtype=np.float32)
        #x-y-w-h-l
        matching_gt_box = np.zeros([self.GRIDSZ, self.GRIDSZ, 5, 5], dtype=np.float32)
        # [40,5] x1-y1-x2-y2 => x-y-w-h-l
        gt_boxes_grid = np.zeros(gt_boxes.shape, dtype=np.float32)

        for i, box in enumerate(gt_boxes): #[40, 5]
            # box: [5], x1-y1-x2-y2
            # 512 => 16
            x = (box[0]+box[2])/2/self.scale
            y = (box[1]+box[3])/2/self.scale
            w = (box[2]-box[0]) / self.scale
            h = (box[3]-box[1]) / self.scale
            # [40, 5] x-y-w-h-l
            gt_boxes_grid[i] = np.array([x,y,w,h,box[4]])

            if w*h > 0: # valid box
                # x,y: 7.3 6.8
                best_anchor = 0
                best_iou = 0
                for j in range(self.anchors.shape[0]):
                    interct = np.minimum(w, self.anchors[j, 0]) * np.minimum(h, self.anchors[j, 1])
                    union = w*h + (self.anchors[j,0]*self.anchors[j,1]) - interct
                    iou = interct / union

                    if iou > best_iou: # best iou
                        best_anchor = j
                        best_iou = iou
                # found the best anchors
                if best_iou>0:
                    x_coord = np.floor(x).astype(np.int32)
                    y_coord = np.floor(y).astype(np.int32)
                    # [b, h, w, 5, 1]
                    detector_mask[y_coord, x_coord, best_anchor] = 1
                    # [b,h,w,5,x-y-w-h-l]
                    matching_gt_box[y_coord, x_coord, best_anchor] = np.array([x,y,w,h,box[4]])

        # [40,5] => [16,16,5,5]
        # [16,16,5,5]
        # [16,16,5,1]
        # [40,5]
        return matching_gt_box, detector_mask, gt_boxes_grid

