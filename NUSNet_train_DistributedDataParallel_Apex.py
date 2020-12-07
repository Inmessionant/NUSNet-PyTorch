
import torch.optim as optim
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from NUSNet_model.NUSNet import *
from data_loader import RandomCrop
from data_loader import RescaleT
from data_loader import SalObjDataset
from data_loader import ToTensorLab
import argparse
from apex import amp
from utils.utils import *
import torch.distributed as dist


# change model_name，epoch_num，batch_size，resume, net, num_workers
def main():
    model_name = 'NUSNet'
    epoch_num = 1500
    batch_size_train = 32
    setup_seed(1222)
    resume = False

    data_dir = os.path.join(os.getcwd(), 'train_data' + os.sep)
    tra_image_dir = os.path.join('TR-Image' + os.sep)
    tra_label_dir = os.path.join('TR-Mask' + os.sep)
    saved_model_dir = os.path.join(os.getcwd(), 'saved_models' + os.sep)
    log_dir = os.path.join(os.getcwd(), 'saved_models', model_name + '.pth')

    if not os.path.exists(saved_model_dir):
        os.makedirs(saved_model_dir, exist_ok=True)

    image_ext = '.jpg'
    label_ext = '.png'

    net = NUSNet(3, 1)  # input channels and output channels
    net.to(device)

    # define optimizer
    optimizer = optim.Adam(net.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)

    start_epoch = 0

    # If there is a saved model, load the model and continue training based on it
    if resume:
        checkpoint = torch.load(log_dir, map_location=torch.device('cpu'))
        net.load_state_dict(checkpoint['model'], False)
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']

    net, optimizer = amp.initialize(net, optimizer, opt_level='O1', verbosity=0)

    dist.init_process_group(backend='nccl',  # 'distributed backend'
                            init_method='tcp://127.0.0.1:9999',  # distributed training init method
                            world_size=1,  # number of nodes for distributed training
                            rank=0)  # distributed training node rank

    net = torch.nn.parallel.DistributedDataParallel(net, find_unused_parameters=True)

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
    train_sampler = torch.utils.data.distributed.DistributedSampler(salobj_dataset)
    salobj_dataloader = torch.utils.data.DataLoader(salobj_dataset, batch_size=batch_size_train, sampler=train_sampler,
                                                    shuffle=False, num_workers=16, pin_memory=True)

    # summary model
    print(summary(net, (3, 320, 320)))

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

            inputs_v, labels_v = inputs.cuda(non_blocking=True), labels.cuda(non_blocking=True)

            # forward + backward + optimize
            final_fusion_loss, sup1, sup2, sup3, sup4, sup5, sup6 = net(inputs_v)
            final_fusion_loss_mblf, total_loss = muti_bce_loss_fusion(final_fusion_loss, sup1, sup2, sup3, sup4, sup5,
                                                                      sup6, labels_v)

            optimizer.zero_grad()

            with amp.scale_loss(total_loss, optimizer) as scaled_loss:
                scaled_loss.backward()

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

    torch.save(net.state_dict(), saved_model_dir + model_name + ".pth")
    torch.cuda.empty_cache()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True  # 固定随机性
    torch.backends.cudnn.benchmark = True  # 尺寸大小一样可以加速训练


# final_fusion_loss is the sum of sup1 to sup6
def muti_bce_loss_fusion(final_fusion_loss, sup1, sup2, sup3, sup4, sup5, sup6, labels_v):
    final_fusion_loss = torch.nn.BCEWithLogitsLoss(reduction='mean')(final_fusion_loss, labels_v).cuda()
    sup1 = torch.nn.BCEWithLogitsLoss(reduction='mean')(sup1, labels_v).cuda()
    sup2 = torch.nn.BCEWithLogitsLoss(reduction='mean')(sup2, labels_v).cuda()
    sup3 = torch.nn.BCEWithLogitsLoss(reduction='mean')(sup3, labels_v).cuda()
    sup4 = torch.nn.BCEWithLogitsLoss(reduction='mean')(sup4, labels_v).cuda()
    sup5 = torch.nn.BCEWithLogitsLoss(reduction='mean')(sup5, labels_v).cuda()
    sup6 = torch.nn.BCEWithLogitsLoss(reduction='mean')(sup6, labels_v).cuda()
    total_loss = (final_fusion_loss + sup1 + sup2 + sup3 + sup4 + sup5 + sup6).cuda()

    return final_fusion_loss, total_loss


# change device
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0, 1', help='device id (i.e. 0 or 0,1 or cpu)')
    args = parser.parse_args()
    device = torch_utils.select_device(args.device)
    main()
