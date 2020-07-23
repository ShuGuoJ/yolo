import numpy as np
import os, glob
import xml.etree.ElementTree as ET
from matplotlib import pyplot as plt
import torch
from matplotlib import patches
from PIL import Image
from torchvision import transforms, ops
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
anchors = torch.tensor(
        [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828], device=device).reshape((5, 2))
GRIDSZ = 16

def compute_iou(x1,y1,w1,h1,x2,y2,w2,h2):
    # x1...: [b,16,16,5]
    xmin1 = x1 - 0.5*w1
    ymin1 = y1 - 0.5*h1
    xmax1 = x1 + 0.5*w1
    ymax1 = y1 + 0.5*h1

    xmin2 = x2 - 0.5*w2
    ymin2 = y2 - 0.5*h2
    xmax2 = x2 + 0.5*w2
    ymax2 = y2 + 0.5*h2

    interw = torch.min(xmax1, xmax2) - torch.max(xmin1, xmin2)
    interh = torch.min(ymax1, ymax2) - torch.max(ymin1, ymin2)
    inter = interw * interh
    union = w1*h1 + w2*h2 - inter
    iou = inter / (union + 1e-6)
    # [b,16,16,5]
    return iou

def parse_annotation(img_dir, ann_dir, labels):
    # img_dir: image path
    # ann_dir: annotation xml file path
    # labels: ('sugarbeet', 'weed')
    # parse annotation info from xml file
    """
        <annotation>
        <folder>train</folder>
        <filename>X2-10-0.png</filename>
        <path /><source>
            <database>Unknown</database>
        </source>
        <size>
            <width>512</width>
            <height>512</height>
            <depth>3</depth>
        </size>
        <segmented>0</segmented>
        <object>

    """

    imgs_info = []
    max_boxes = 0
    # for each annotation xml file
    for ann in os.listdir(ann_dir):
        tree = ET.parse(os.path.join(ann_dir, ann))

        img_info = dict()
        img_info['object'] = []
        boxes_counter = 0
        for elem in tree.iter():
            if 'filename' in elem.tag:
                img_info['filename'] = os.path.join(img_dir, elem.text)
            if 'width' in elem.tag:
                img_info['width'] = int(elem.text)
                assert img_info['width'] == 512
            if 'heigth' in elem.tag:
                img_info['height'] = int(elem.text)
                assert img_info['height'] == 512

            if 'object' in elem.tag or 'part' in elem.tag:
                # x1-y1-x2-y2-label
                object_info = [0.,]*5
                boxes_counter += 1
                for child in list(elem):
                    if 'name' in child.tag:
                        label = labels.index(child.text)
                        object_info[4] = label

                    if 'bndbox' in child.tag:
                        for pos in list(child):
                            if 'xmin' in pos.tag:
                                object_info[0] = int(pos.text)
                            if 'ymin' in pos.tag:
                                object_info[1] = int(pos.text)
                            if 'xmax' in pos.tag:
                                object_info[2] = int(pos.text)
                            if 'ymax' in pos.tag:
                                object_info[3] = int(pos.text)
                img_info['object'].append(object_info)
        imgs_info.append(img_info) # filename, w/h/box_info
        # (N,5) = (max_objects_num, 5)
        if boxes_counter > max_boxes:
            max_boxes = boxes_counter
    # the maximun boxes number is max_boxes
    # [b, 40, 5]
    boxes = np.zeros((len(imgs_info), max_boxes, 5), dtype=np.float32)
    # print(boxes.shape)
    imgs = [] # filename list
    for i, img_info in enumerate(imgs_info):
        #[N, 5]
        img_boxes = np.array(img_info['object'], dtype=np.float32)
        # overwrite the N boxes info
        boxes[i, :img_boxes.shape[0]] = img_boxes

        imgs.append(img_info['filename'])

        # print(img_info['filename'], boxes[i, :5])
    # imgs: list of image path
    # boxes: [b, 40, 5]
    return imgs, boxes

# obj_names = ('sugarbeet', 'weed')
# imgs, boxes = parse_annotation('data/train/image', 'data/train/annotation', obj_names)

def visualize(dataset):
    img, boxes = dataset[0]
    f, ax1 = plt.subplots(1, figsize=(10, 10))
    img = img.permute(1, 2, 0)
    ax1.imshow(img.numpy())
    for x1, y1, x2, y2, l in boxes:
        x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
        w = x2 - x1
        h = y2 - y1

        if l==1: # green for sugarweet
            color = (0, 1, 0)
        elif l==2:
            color = (1, 0, 0)
        else:
            break

        rect = patches.Rectangle((x1, y1), w, h, linewidth=2,
                                 edgecolor=color, facecolor='none')
        ax1.add_patch(rect)
    plt.show()


def my_collate_fn(batch):
    # [b,3,512,512]
    # [b, 16, 16, 5, 5]
    # [b, 16, 16, 5, 1]
    # [b, 40, 5]
    imgs = []
    boxes = []
    masks = []
    grids = []
    for sample in batch:
        imgs.append(sample[0])
        boxes.append(torch.tensor(sample[1][0]))
        masks.append(torch.tensor(sample[1][1]))
        grids.append(torch.tensor(sample[1][2]))
    # 在batch维度上拼接
    imgs = torch.stack(imgs, dim=0)
    matching_gt_boxes = torch.stack(boxes, dim=0)
    detector_masks = torch.stack(masks, dim=0)
    gt_box_grids = torch.stack(grids, dim=0)
    return imgs, (matching_gt_boxes, detector_masks, gt_box_grids)

def yolo_loss(matching_gt_box, detector_mask, gt_boxes_grid, y_pred):
    # matching_gt_box: [b,16,16,5,5] x-y-w-h-l
    # detector_mask: [b,16,16,5,1]
    # gt_boxes_grid: x-y-w-h
    # y_pred: [b,16,16,5,7]


    # compute coordinate loss
    # create starting position for each grid anchors
    # [16,16]
    x_grid = torch.arange(GRIDSZ, device=device).repeat(GRIDSZ).reshape(GRIDSZ, GRIDSZ)
    # [1,16,16,1,1]
    x_grid = x_grid.reshape(1,GRIDSZ,GRIDSZ,1,1).float()
    # [b,16_1,16_2,1,1] => [b,16_2,16_1,1,1]
    y_grid = x_grid.permute(0,2,1,3,4)
    xy_grid = torch.cat([x_grid, y_grid], dim=-1)
    # [1,16,16,1,2] => [b,16,16,5,2]
    xy_grid = xy_grid.repeat((y_pred.shape[0],1,1,5,1))

    # [b,16,16,5,7] x-y-w-h-conf-l1-l2 => [b,16,16,5,2]
    pred_xy = torch.sigmoid(y_pred[...,0:2])
    pred_xy = pred_xy + xy_grid
    # [b,16,16,5,7] => [b,16,16,5,2]
    pred_wh = torch.exp(y_pred[...,2:4])
    # [b,16,16,5,2] * [5,2] => [b,16,16,5,2]
    pred_wh = pred_wh * anchors

    n_detector_mask = detector_mask.sum().float()

    xy_loss = detector_mask * torch.square(pred_xy-matching_gt_box[...,:2]) / (n_detector_mask + 1e-6)
    xy_loss = xy_loss.sum()
    wh_loss = detector_mask * torch.square(torch.sqrt(pred_wh)-torch.sqrt(matching_gt_box[...,2:4])) / (n_detector_mask + 1e-6)
    wh_loss = wh_loss.sum()

    # coordinate loss
    coord_loss = xy_loss + wh_loss

    # compute label loss
    # [b,16,16,5,2]
    pred_box_class = y_pred[...,5:]
    # [b,16,16,5]
    true_box_class = matching_gt_box[...,-1]
    # [b,16,16,5] vs [b,16,16,5,2]
    class_loss = torch.nn.functional.cross_entropy(pred_box_class.permute(0,4,1,2,3), true_box_class.long(), reduction='none')
    # [b,16,16,5] => [b,16,16,5,1] * [b,16,16,5,1]
    class_loss = class_loss.unsqueeze(-1) * detector_mask
    class_loss = class_loss.sum() / (n_detector_mask + 1e-6)

    # object loss
    # nonobject_mask
    # iou done!
    # [b,16,16,5]
    x1,y1,w1,h1 = matching_gt_box[...,0], matching_gt_box[...,1], matching_gt_box[...,2], matching_gt_box[...,3]
    # [b,16,16,5]
    x2, y2, w2, h2 = pred_xy[...,0], pred_xy[...,1], pred_wh[...,0], pred_wh[...,1]
    ious = compute_iou(x1,y1,w1,h1,x2,y2,w2,h2)
    ious = ious.unsqueeze(-1)

    # [b,16,16,5,1]
    pred_conf = torch.sigmoid(y_pred[...,4:5])
    # [b,16,16,5,2] => [b,16,16,5,1,2]
    pred_xy = pred_xy.unsqueeze(4)
    # [b,16,16,5,2] => [b,16,16,5,1,2]
    pred_wh = pred_wh.unsqueeze(4)
    pred_wh_half = pred_wh / 2.
    pred_xymin = pred_xy - pred_wh_half
    pred_xymax = pred_xy + pred_wh_half

    # [b,40,5] => [b,1,1,1,40,5]
    true_boxes_grid = gt_boxes_grid.view(gt_boxes_grid.shape[0],1,1,1,gt_boxes_grid.shape[1],gt_boxes_grid.shape[2])
    true_xy = true_boxes_grid[...,:2]
    true_wh = true_boxes_grid[...,2:4]
    true_wh_half = true_wh / 2.
    true_xymin = true_xy - true_wh_half
    true_xymax = true_xy + true_wh_half
    # predxymin, predxymax, true_xymin, true_xymax
    # [b,16,16,5,1,2] vs [b,1,1,1,40,2] => [b,16,16,5,40,2]
    intersectxymin = torch.max(pred_xymin, true_xymin)
    # [b,16,16,5,1,2] vs [b,1,1,1,40,2] => [b,16,16,5,40,2]
    intersectxymax = torch.min(pred_xymax, true_xymax)
    # [b,16,16,5,40,2]
    intersect_wh = torch.max(intersectxymax-intersectxymin, torch.tensor([0.], device=device))
    # [b,16,16,5,40] * [b,16,16,5,40] => [b,16,16,5,40]
    intersect_area = intersect_wh[...,0] * intersect_wh[...,1]
    # [b,16,16,5,1]
    pred_area = pred_wh[...,0] * pred_wh[...,1]
    # [b,1,1,1,40]
    true_area = true_wh[...,0] * true_wh[...,1]
    # [b,16,16,5,1] + [b,16,16,5,40] - [b,16,16,5,40] => [b,16,16,5,40]
    union_area = pred_area + true_area - intersect_area
    # [b,16,16,5,40]
    iou_score = intersect_area / union_area
    # [b,16,16,5]
    best_iou, _ = iou_score.max(dim=4, keepdim=True)

    nonobj_detection = (best_iou<0.6).float()
    nonobj_mask = nonobj_detection * (1-detector_mask)
    # nonobj counter
    n_nonobj = (nonobj_mask>0.).float().sum()

    nonobj_loss = torch.sum(nonobj_mask*torch.square(-pred_conf)) / (n_nonobj+1e-6)
    obj_loss = torch.sum(detector_mask*torch.square(ious-pred_conf)) / (n_detector_mask+1e-6)

    loss = coord_loss + class_loss + nonobj_loss + 5*obj_loss
    return loss, (nonobj_loss+5*obj_loss, class_loss, coord_loss)


def visualize_img(net, path):
    # 读取图像
    img = Image.open(path)
    transform = transforms.Compose([
        transforms.Resize(512,512),
        transforms.ToTensor()
    ])
    input = transform(img)
    input = input.unsqueeze(0)

    # get result:[1,16,16,5,7]
    net.eval()
    net.to(device)
    input = input.to(device)
    out = net(input)

    x_grid = torch.arange(GRIDSZ, device=device).float().repeat((GRIDSZ))
    # [1,16,16,1,1]
    x_grid = x_grid.view(1,GRIDSZ,GRIDSZ,1,1)
    y_grid = x_grid.permute(0,2,1,3,4)
    xy_grid = torch.cat([x_grid, y_grid], dim=-1)
    # [1,16,16,5,2]
    xy_grid = xy_grid.repeat((1,1,1,5,1))

    pred_xy = torch.sigmoid(out[...,:2])
    pred_xy = pred_xy + xy_grid
    #normalize 0~1
    pred_xy = pred_xy / torch.tensor([16.], device=device)

    pred_wh = torch.exp(out[...,2:4])
    pred_wh = pred_wh * anchors
    pred_wh = pred_wh / torch.tensor([16.], device=device)

    # [1,16,16,5,1]
    pred_conf = torch.sigmoid(out[...,4:5])
    # l1 l2
    pred_prob = torch.softmax(out[...,5:7], dim=-1)

    pred_xy, pred_wh, pred_conf, pred_prob = \
    pred_xy[0], pred_wh[0], pred_conf[0], pred_prob[0]

    boxes_xymin = pred_xy - 0.5 * pred_wh
    boxes_xymax = pred_xy + 0.5 * pred_wh
    # [16,16,5,2+3] x1-y1-x2-y2
    boxes = torch.cat([boxes_xymin, boxes_xymax], dim=-1)
    # [16,16,5,2]
    box_score = pred_conf * pred_prob
    #[16,16,5]
    box_class_score, box_class = box_score.max(dim=-1)
    # [16,16,5]
    pred_mask = box_class_score >0.45
    pred_mask = pred_mask
    # [16,16,5,4] =>[N,4]
    boxes = boxes.masked_select(pred_mask.unsqueeze(-1)).view(-1,4)
    # [16,16,5] => [N]
    scores = box_class_score.masked_select(pred_mask)
    # [16,16,5] => [N]
    classes = box_class.masked_select(pred_mask)

    boxes = boxes * 512
    #[N] => [n]
    select_idx = ops.nms(boxes, scores, 0.1)
    boxes = boxes.index_select(0, select_idx)
    scores = scores.index_select(0, select_idx)
    classes = classes.index_select(0, select_idx)

    # plot
    fig, ax = plt.subplots(1, figsize=(10,10))
    ax.imshow(np.array(img))
    n_boxes = boxes.shape[0]

    ax.set_title('boxes:{}'.format(n_boxes))
    for i in range(n_boxes):
        x1,y1,x2,y2 = boxes[i]
        w = x2-x1
        h = y2-y1
        label = classes[i].item()

        color = (0,1,0) if label==0 else (1,0,0)
        rect = patches.Rectangle((x1.item(), y1.item()), w.item(), h.item(), linewidth=2,
                                 edgecolor=color, facecolor='none')
        ax.add_patch(rect)
    plt.show()
