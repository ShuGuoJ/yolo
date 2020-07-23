import torch
from torch import optim, nn
from DarkNet import DarkNet
from MyDataset import MyDataset
from utils import *
from torch.utils.data import DataLoader
from visdom import Visdom

seed = 236594
torch.manual_seed(seed)

epoch = 60
lr = 1e-4
batchsz = 4
obj_name = ('sugarbeet', 'weed')
train_dataset = MyDataset('data/train/image', 'data/train/annotation', obj_name)
train_loader = DataLoader(train_dataset, batch_size=batchsz, shuffle=True, collate_fn=my_collate_fn)
test_dataset = MyDataset('data/val/image', 'data/val/annotation', obj_name)
test_loader = DataLoader(test_dataset, batch_size=4, collate_fn=my_collate_fn)
viz = Visdom()
viz.line([[0., 0.]], [0], win='train&&val', opts=dict(title="train&&val",
                                                      legend=['train', 'val']))


net = DarkNet()
optimizer = optim.Adam(net.parameters(), lr=lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, 30, 0.5)
net.to(device)

for i in range(epoch):
    total_loss = []
    net.train()
    for step, (input, (matching_gt_box, detector_mask, gt_boxes_grid)) in enumerate(train_loader):
        input, matching_gt_box, detector_mask, gt_boxes_grid = input.to(device), matching_gt_box.to(device), detector_mask.to(device), gt_boxes_grid.to(device)
        # print(matching_gt_box)
        # print(detector_mask.sum())
        # print(gt_boxes_grid)
        out = net(input)
        loss, subloss = yolo_loss(matching_gt_box, detector_mask, gt_boxes_grid, out)
        total_loss.append(loss.item())
        # gradient descent
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('epoch:{} batch:{} loss:{} object_loss:{} class_loss:{} coord_loss:{}'.format(i, step, loss.item(), subloss[0].item(), subloss[1].item(), subloss[2].item()))

    scheduler.step()
    net.eval()
    val_loss, object_loss, class_loss, coord_loss = 0., 0., 0., 0.
    for input, (matching_gt_box, detector_mask, gt_boxes_grid) in test_loader:
        input, matching_gt_box, detector_mask, gt_boxes_grid = \
        input.to(device), matching_gt_box.to(device), detector_mask.to(device), gt_boxes_grid.to(device)
        out = net(input)
        loss, subloss = yolo_loss(matching_gt_box, detector_mask, gt_boxes_grid, out)
        val_loss += loss.item() * input.shape[0]
        object_loss += subloss[0].item() * input.shape[0]
        class_loss += subloss[1].item() * input.shape[0]
        coord_loss += subloss[2].item() * input.shape[0]
    val_loss /= len(test_loader.dataset)
    object_loss /= len(test_loader.dataset)
    class_loss /= len(test_loader.dataset)
    coord_loss /= len(test_loader.dataset)
    print("epoch:{} loss:{} object_loss:{} class_loss:{} coord_loss:{}".format(i, loss, object_loss, class_loss, coord_loss))
    viz.line([[float(np.mean(total_loss)), val_loss]], [i], win='train&&val', update='append')
torch.save(net.state_dict(), 'model.pkl')


visualize_img(net, 'data/val/image/X3-130-1.png')