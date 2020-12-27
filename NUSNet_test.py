import logging

from torch.utils.data import DataLoader
from torchvision import transforms

from NUSNet_model.NUSNet import *
from data_loader import *
from torch_utils import *

logging.getLogger().setLevel(logging.INFO)


# change gpus，model_name，pre_data_dir,  model, num_workers
def main():
    model_name = 'NUSNet'
    pre_data_dir = 'SOD'  # 'TUDS-TE'   'PASCAL'   'HKU'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Models : NUSNet  NUSNet4  NUSNet5  NUSNet6  NUSNet7  NUSNetCAM  NUSNetSAM  NUSNetCBAM
    # NUSNetNet765CAM4SMALLSAM
    model = NUSNet(3, 1)    # input channels and output channels
    model_info(model, verbose=True)

    model.to(device)
    # logging.info(summary(model, (3, 320, 320)))

    image_dir = os.path.join(os.getcwd(), 'test_data', pre_data_dir)
    prediction_dir = os.path.join(os.getcwd(), 'test_data', pre_data_dir + '_Results', model_name + os.sep)
    model_dir = os.path.join(os.getcwd(), 'saved_models', model_name + '.pth')
    img_name_list = glob.glob(image_dir + os.sep + '*')

    if not os.path.exists(prediction_dir):
        os.makedirs(prediction_dir, exist_ok=True)

    model.load_state_dict(torch.load(model_dir))

    # dataloader
    test_salobj_dataset = SalObjDataset(img_name_list=img_name_list, lbl_name_list=[],
                                        transform=transforms.Compose([RescaleT(320), ToTensorLab(flag=0)]))
    test_salobj_dataloader = DataLoader(test_salobj_dataset, batch_size=1, shuffle=False, num_workers=8,
                                        pin_memory=True)
    time_sum = 0
    logging.info('Start inference!')
    model.eval()

    # inference for each image
    for i_test, data_test in enumerate(test_salobj_dataloader):
        logging.info('testing: %s' % img_name_list[i_test].split(os.sep)[-1])

        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)
        inputs_test = inputs_test.to(device, non_blocking=True)

        with torch.no_grad():
            start = time_synchronized()
            sup1, sup2, sup3, sup4, sup5, sup6, sup7 = model(inputs_test)
            time_sum += time_synchronized() - start

            # normalization
            pred = sup1[:, 0, :, :]
            pred = normPRED(pred)

            save_output(img_name_list[i_test], pred, prediction_dir)

    logging.info('\n' + '%s is %f fps in the %s DataSet.' % (model_name, len(img_name_list) / time_sum, pre_data_dir))
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
