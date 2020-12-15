import argparse
import logging

import torch.distributed as dist
import torch.optim as optim
from NUSNet_model.NUSNet import *
from torch.utils.data import DataLoader
from torchvision import transforms

from data_loader import *
from utils.torch_utils import *
from utils.utils import *

try:
    from apex import amp
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

logging.getLogger().setLevel(logging.INFO)


# change model_name，epoch_num，batch_size，resume, model, num_workers
def main():
    epoch_num = 3000
    batch_size_train = 64
    model_name = 'NUSNet'
    init_seeds(2 + batch_size_train)
    resume = False

    model = NUSNet(3, 1)  # input channels and output channels
    print("Using apex synced BN.")
    model = amp.parallel.convert_syncbn_model(model)
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)

    tra_image_dir = os.path.abspath(str(Path('train_data/TR-Image')))
    tra_label_dir = os.path.abspath(str(Path('train_data/TR-Mask')))
    saved_model_dir = os.path.join(os.getcwd(), 'saved_models' + os.sep)
    log_dir = os.path.join(os.getcwd(), 'saved_models', model_name + '_Temp.pth')

    if not os.path.exists(saved_model_dir):
        os.makedirs(saved_model_dir, exist_ok=True)

    model, optimizer = amp.initialize(model, optimizer, opt_level='O1', verbosity=0)

    dist.init_process_group(backend='nccl',  # 'distributed backend'
                            init_method='tcp://127.0.0.1:9999',  # distributed training init method
                            world_size=1,  # number of nodes for distributed training
                            rank=0)  # distributed training node rank

    model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)

    start_epoch = 0

    # If there is a saved model, load the model and continue training based on it
    if resume:
        checkpoint = torch.load(log_dir, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model'], False)
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']

    img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.dng']

    images_files = sorted(glob.glob(os.path.join(tra_image_dir, '*.*')))
    labels_files = sorted(glob.glob(os.path.join(tra_label_dir, '*.*')))

    tra_img_name_list = [x for x in images_files if os.path.splitext(x)[-1].lower() in img_formats]
    tra_lbl_name_list = [x for x in labels_files if os.path.splitext(x)[-1].lower() in img_formats]

    logging.info('================================================================')
    logging.info('train images numbers: %g' % len(tra_img_name_list))
    logging.info('train labels numbers: %g' % len(tra_lbl_name_list))

    assert len(tra_img_name_list) == len(
        tra_lbl_name_list), 'The number of training images: %g  , the number of training labels: %g .' % (
        len(tra_img_name_list), len(tra_lbl_name_list))

    salobj_dataset = SalObjDataset(img_name_list=tra_img_name_list, lbl_name_list=tra_lbl_name_list,
                                   transform=transforms.Compose([RescaleT(320), RandomCrop(288), ToTensorLab(flag=0)]))
    train_sampler = torch.utils.data.distributed.DistributedSampler(salobj_dataset)
    salobj_dataloader = torch.utils.data.DataLoader(salobj_dataset, batch_size=batch_size_train, sampler=train_sampler,
                                                    shuffle=False, num_workers=16, pin_memory=True)

    # summary model
    logging.info(summary(model, (3, 320, 320)))

    # training parameter
    ite_num = 0
    running_loss = 0.0  # total_loss = final_fusion_loss +sup1 +sup2 + sup3 + sup4 +sup5 +sup6
    running_tar_loss = 0.0  # final_fusion_loss

    for epoch in range(start_epoch, epoch_num):

        model.train()
        pbar = enumerate(salobj_dataloader)
        pbar = tqdm(pbar, total=len(salobj_dataloader))

        for i, data in pbar:
            ite_num = ite_num + 1

            inputs, labels = data['image'], data['label']
            inputs, labels = inputs.type(torch.FloatTensor), labels.type(torch.FloatTensor)
            inputs_v, labels_v = inputs.cuda(non_blocking=True), labels.cuda(non_blocking=True)

            # forward + backward + optimize
            final_fusion_loss, sup1, sup2, sup3, sup4, sup5, sup6 = model(inputs_v)
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

            s = ('%10s' + '%-15s' + '%10s' + '%-15s' + '%10s' + '%-10d' + '%20s' + '%-10.4f' + '%20s' + '%-10.4f') % (
                'Epoch: ',
                '%g/%g' % (epoch + 1, epoch_num),
                'Batch: ',
                '%g/%g' % ((i + 1) * batch_size_train, len(tra_img_name_list)),
                'Iteration: ',
                ite_num,
                'Total_loss: ',
                running_loss / ite_num,
                'Final_fusion_loss: ',
                running_tar_loss / ite_num)
            pbar.set_description(s)

        # The model is saved every 50 epoch
        if (epoch + 1) % 50 == 0:
            state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch + 1}
            torch.save(state, saved_model_dir + model_name + ".pth")

    torch.save(model.state_dict(), saved_model_dir + model_name + ".pth")
    torch.cuda.empty_cache()


# change device
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0, 1', help='device id (i.e. 0 or 0,1 or cpu)')
    args = parser.parse_args()
    device = torch_utils.select_device(args.device)
    main()
