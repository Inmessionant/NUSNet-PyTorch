import torch.optim as optim
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms
from NUSNet_model.NUSNet import *
from data_loader import *
import argparse
from apex import amp
from utils.utils import *
from utils.torch_utils import *
import torch.distributed as dist


# change model_name，epoch_num，batch_size，resume, net, num_workers
def main():
    epoch_num = 150000
    batch_size_train = 64
    model_name = 'NUSNet'
    init_seeds(2 + batch_size_train)
    resume = False

    net = NUSNet(3, 1).cuda()  # input channels and output channels
    optimizer = optim.Adam(net.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)

    tra_image_dir = os.path.abspath(str(Path('train_data/TR-Image')))
    tra_label_dir = os.path.abspath(str(Path('train_data/TR-Mask')))
    saved_model_dir = os.path.abspath(str(Path('saved_models')))
    log_dir = os.path.join(os.getcwd(), 'saved_models', model_name + '_Temp.pth')

    if not os.path.exists(saved_model_dir):
        os.makedirs(saved_model_dir, exist_ok=True)

    net, optimizer = amp.initialize(net, optimizer, opt_level='O1', verbosity=0)

    dist.init_process_group(backend='nccl',  # 'distributed backend'
                            init_method='tcp://127.0.0.1:9999',  # distributed training init method
                            world_size=1,  # number of nodes for distributed training
                            rank=0)  # distributed training node rank

    net = torch.nn.parallel.DistributedDataParallel(net, find_unused_parameters=True)

    start_epoch = 0

    # If there is a saved model, load the model and continue training based on it
    if resume:
        checkpoint = torch.load(log_dir, map_location=torch.device('cpu'))
        net.load_state_dict(checkpoint['model'], False)
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']

    img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.dng']

    images_files = sorted(glob.glob(os.path.join(tra_image_dir, '*.*')))
    labels_files = sorted(glob.glob(os.path.join(tra_label_dir, '*.*')))

    tra_img_name_list = [x for x in images_files if os.path.splitext(x)[-1].lower() in img_formats]
    tra_lbl_name_list = [x for x in labels_files if os.path.splitext(x)[-1].lower() in img_formats]
    
    print("================================================================")
    print("train images numbers: ", len(tra_img_name_list))
    print("train labels numbers: ", len(tra_lbl_name_list))

    assert len(tra_img_name_list) == len(tra_lbl_name_list), 'The number of training images: %g  , the number of training labels: %g .' % (len(tra_img_name_list), len(tra_lbl_name_list))

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
            inputs, labels = inputs.type(torch.FloatTensor), labels.type(torch.FloatTensor)
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
                epoch + 1, epoch_num, (i + 1) * batch_size_train, len(tra_img_name_list), ite_num, running_loss / ite_num,
                running_tar_loss / ite_num))

        # The model is saved every 50 epoch
        if (epoch + 1) % 50 == 0:
            state = {'model': net.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch + 1}
            torch.save(state, saved_model_dir + model_name + ".pth")

    torch.save(net.state_dict(), saved_model_dir + model_name + ".pth")
    torch.cuda.empty_cache()


# change device
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0, 1', help='device id (i.e. 0 or 0,1 or cpu)')
    args = parser.parse_args()
    device = torch_utils.select_device(args.device)
    main()
