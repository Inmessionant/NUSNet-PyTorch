import glob
import os

import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchsummary import summary
from torchvision import transforms
import numpy as np
from UXNet_model import UXNet
from data_loader import RandomCrop
from data_loader import RescaleT
from data_loader import SalObjDataset
from data_loader import ToTensorLab
import torch.backends.cudnn as cudnn

# define loss function
bce_loss = nn.BCELoss(reduction='mean')



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

# final_fusion_loss is the sum of sup1 to sup6
def muti_bce_loss_fusion(final_fusion_loss, sup1, sup2, sup3, sup4, sup5, sup6, labels_v):
    final_fusion_loss = bce_loss(final_fusion_loss, labels_v)
    sup1 = bce_loss(sup1, labels_v)
    sup2 = bce_loss(sup2, labels_v)
    sup3 = bce_loss(sup3, labels_v)
    sup4 = bce_loss(sup4, labels_v)
    sup5 = bce_loss(sup5, labels_v)
    sup6 = bce_loss(sup6, labels_v)

    total_loss = final_fusion_loss + sup1 + sup2 + sup3 + sup4 + sup5 + sup6

    return final_fusion_loss, total_loss


def main():
    #  set the directory of training dataset
    model_name = 'UXNet'
    data_dir = os.path.join(os.getcwd(), 'train_data' + os.sep)
    tra_image_dir = os.path.join('DUTS-TR', 'DUTS-TR-Image' + os.sep)
    tra_label_dir = os.path.join('DUTS-TR', 'DUTS-TR-Mask' + os.sep)
    log_dir = os.path.join(os.getcwd(), 'log_dir', 'UXNet.pth' + os.sep)  # transformer training ,change thr file name
    image_ext = '.jpg'
    label_ext = '.png'

    # the directory of test_model
    # model_dir = os.path.join(os.getcwd(), 'test_model', model_name + os.sep)
    saved_model_dir = os.path.join(os.getcwd(), 'saved_models' + os.sep)

    epoch_num = 7000
    batch_size_train = 16

    tra_img_name_list = glob.glob(data_dir + tra_image_dir + '*' + image_ext)

    tra_lbl_name_list = []
    for img_path in tra_img_name_list:
        img_name = img_path.split(os.sep)[-1]

        aaa = img_name.split(".")
        bbb = aaa[0:-1]
        imidx = bbb[0]
        for i in range(1, len(bbb)):
            imidx = imidx + "." + bbb[i]

        tra_lbl_name_list.append(data_dir + tra_label_dir + imidx + label_ext)

    print("================================================================")
    print("train images numbers: ", len(tra_img_name_list))
    print("train labels numbers: ", len(tra_lbl_name_list))
    train_num = len(tra_img_name_list)

    if len(tra_img_name_list) != len(tra_lbl_name_list):
        print("The number of training images does not match the number of training labels, please check again!")
        exit()

    # define model
    setup_seed(2018)
    # define the net
    net = UXNet(3, 1)  # input channels and output channels

    # define optimizer
    optimizer = optim.Adam(net.parameters(), lr=1e-8, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)

    # 如果有保存的模型，则加载模型，并在其基础上继续训练
    if os.path.exists(log_dir):
        checkpoint = torch.load(log_dir)
        net.load_state_dict(checkpoint['model'])

        # send optimizer to cuda
        optimizer.load_state_dict(checkpoint['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()

        start_epoch = checkpoint['epoch']
        print("================================================================")
        print('加载 epoch {} 成功！'.format(start_epoch))
    else:
        start_epoch = 0
        print("================================================================")
        print('无保存模型，将从头开始训练！')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)

    # summary model
    summary(net, (3, 320, 320))

    salobj_dataset = SalObjDataset(
        img_name_list=tra_img_name_list,
        lbl_name_list=tra_lbl_name_list,
        transform=transforms.Compose([
            RescaleT(320),
            RandomCrop(288),
            ToTensorLab(flag=0)]))

    salobj_dataloader = DataLoader(salobj_dataset, batch_size=batch_size_train, shuffle=True, num_workers=4, pin_memory=True)

    # training parameter
    ite_num = 0
    running_loss = 0.0  # total_loss = final_fusion_loss +sup1 +sup2 + sup3 + sup4 +sup5 +sup6
    running_tar_loss = 0.0  # final_fusion_loss

    for epoch in range(start_epoch, epoch_num):
        net.train()

        for i, data in enumerate(salobj_dataloader):
            ite_num = ite_num + 1

            inputs, labels = data['image'], data['label']

            inputs = inputs.type(torch.FloatTensor)
            labels = labels.type(torch.FloatTensor)

            # wrap them in Variable
            if torch.cuda.is_available():
                inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(),
                                                                                            requires_grad=False)
            else:
                inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

            # y zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            final_fusion_loss, sup1, sup2, sup3, sup4, sup5, sup6 = net(inputs_v)
            final_fusion_loss_mblf, total_loss = muti_bce_loss_fusion(final_fusion_loss, sup1, sup2, sup3, sup4, sup5,
                                                                      sup6, labels_v)

            total_loss.backward()
            optimizer.step()

            # # print statistics
            running_loss += total_loss.item()
            running_tar_loss += final_fusion_loss_mblf.item()

            # del temporary outputs and loss
            del final_fusion_loss, sup1, sup2, sup3, sup4, sup5, sup6, final_fusion_loss_mblf, total_loss

            print("[ epoch: %3d/%3d, batch: %5d/%5d, iteration: %d ] total_loss: %3f, final_fusion_loss: %3f " % (
                epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, running_loss / ite_num,
                running_tar_loss / ite_num))

        # 支持断点训练，每1epoch保存一次
        state = {'model': net.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch+1}
        torch.save(state, saved_model_dir + model_name + "_epoch_%d_bce_itr_%d_train_%3f_final_fusion_loss_%3f.pth" % (
            epoch+1, ite_num, running_loss / ite_num, running_tar_loss / ite_num))


if __name__ == "__main__":
    main()
