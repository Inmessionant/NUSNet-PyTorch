import glob
import os
import random
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from UXNet_model.UXNet import *
from data_loader import RandomCrop
from data_loader import RescaleT
from data_loader import SalObjDataset
from data_loader import ToTensorLab


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


# final_fusion_loss is the sum of sup1 to sup6
def muti_bce_loss_fusion(final_fusion_loss, sup1, sup2, sup3, sup4, sup5, sup6, labels_v):
    final_fusion_loss = nn.BCELoss(reduction='mean')(final_fusion_loss, labels_v).cuda()
    sup1 = nn.BCELoss(reduction='mean')(sup1, labels_v).cuda()
    sup2 = nn.BCELoss(reduction='mean')(sup2, labels_v).cuda()
    sup3 = nn.BCELoss(reduction='mean')(sup3, labels_v).cuda()
    sup4 = nn.BCELoss(reduction='mean')(sup4, labels_v).cuda()
    sup5 = nn.BCELoss(reduction='mean')(sup5, labels_v).cuda()
    sup6 = nn.BCELoss(reduction='mean')(sup6, labels_v).cuda()
    total_loss = (final_fusion_loss + sup1 + sup2 + sup3 + sup4 + sup5 + sup6).cuda()

    return final_fusion_loss, total_loss


# change gpus，model_name，epoch_num，batch_size，resume, net, num_workers
def main():
    gpus = [0, 1]
    torch.cuda.set_device('cuda:{}'.format(gpus[0]))
    setup_seed(1222)
    model_name = 'UXNet'
    epoch_num = 10000
    batch_size_train = 32
    resume = False

    # Models : UXNet  UXNet4  UXNet5  UXNet6  UXNet7  UXNetCAM  UXNetSAM  UXNetCBAM  UXNet765CAM4SMALLSAM
    net = UXNet(3, 1).cuda()  # input channels and output channels
    net = nn.DataParallel(net, device_ids=gpus, output_device=gpus[0])

    data_dir = os.path.join(os.getcwd(), 'train_data' + os.sep)
    tra_image_dir = os.path.join('TR-Image' + os.sep)
    tra_label_dir = os.path.join('TR-Mask' + os.sep)
    saved_model_dir = os.path.join(os.getcwd(), 'saved_models' + os.sep)
    log_dir = os.path.join(os.getcwd(), 'saved_models', model_name + '.pth')

    if not os.path.exists(saved_model_dir):
        os.makedirs(saved_model_dir, exist_ok=True)

    image_ext = '.jpg'
    label_ext = '.png'

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

    salobj_dataset = SalObjDataset(img_name_list=tra_img_name_list, lbl_name_list=tra_lbl_name_list,
                                   transform=transforms.Compose([RescaleT(320), RandomCrop(288), ToTensorLab(flag=0)]))
    salobj_dataloader = DataLoader(salobj_dataset, batch_size=batch_size_train, shuffle=True, num_workers=16,
                                   pin_memory=True)

    optimizer = optim.Adam(net.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)

    start_epoch = 0
    # If there is a saved model, load the model and continue training based on it
    if resume:
        checkpoint = torch.load(log_dir)
        net.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']

    print(summary(net, (3, 320, 320)))

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

            inputs_v, labels_v = inputs.cuda(non_blocking=True), labels.cuda(non_blocking=True)

            # forward + backward + optimize
            final_fusion_loss, sup1, sup2, sup3, sup4, sup5, sup6 = net(inputs_v)
            final_fusion_loss_mblf, total_loss = muti_bce_loss_fusion(final_fusion_loss, sup1, sup2, sup3, sup4, sup5,
                                                                      sup6, labels_v)

            # y zero the parameter gradients
            optimizer.zero_grad()

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

        # The model is saved every 50 epoch
        if (epoch + 1) % 50 == 0:
            state = {'model': net.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch + 1}
            torch.save(state, saved_model_dir + model_name + ".pth")

    state = {'model': net.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch + 1}
    torch.save(state, saved_model_dir + model_name + ".pth")
    # torch.save(net.state_dict(), saved_model_dir + model_name + ".pth")
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
