import glob
from torch.utils.data import DataLoader
from torchvision import transforms
from NUSNet_model.NUSNet import *
from utils.torch_utils import *
from data_loader import *


# change gpus，model_name，pre_data_dir,  net, num_workers
def main():
    model_name = 'NUSNet'
    pre_data_dir = 'SOD'  # 'TUDS-TE'   'PASCAL'   'HKU'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Models : NUSNetNet  NUSNetNet4  NUSNetNet5  NUSNetNet6  NUSNetNet7  NUSNetNetCAM  NUSNetNetSAM  NUSNetNetCBAM
    # NUSNetNet765CAM4SMALLSAM
    net = NUSNet(3, 1).to(device)  # input channels and output channels

    print(summary(net, (3, 320, 320)))

    image_dir = os.path.join(os.getcwd(), 'test_data', pre_data_dir)
    prediction_dir = os.path.join(os.getcwd(), 'test_data', pre_data_dir + '_Results', model_name + os.sep)
    model_dir = os.path.join(os.getcwd(), 'saved_models', model_name + '.pth')
    img_name_list = glob.glob(image_dir + os.sep + '*')

    if not os.path.exists(prediction_dir):
        os.makedirs(prediction_dir, exist_ok=True)

    net.load_state_dict(torch.load(model_dir))

    # dataloader
    test_salobj_dataset = SalObjDataset(img_name_list=img_name_list, lbl_name_list=[],
                                        transform=transforms.Compose([RescaleT(320), ToTensorLab(flag=0)]))
    test_salobj_dataloader = DataLoader(test_salobj_dataset, batch_size=1, shuffle=False, num_workers=0,
                                        pin_memory=True)
    time_sum = 0
    print('Start inference!')
    net.eval()

    # inference for each image
    for i_test, data_test in enumerate(test_salobj_dataloader):

        print("testing:", img_name_list[i_test].split(os.sep)[-1])

        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)
        inputs_test = inputs_test.to(device, non_blocking=True)

        with torch.no_grad():

            start = time_synchronized()
            sup1, sup2, sup3, sup4, sup5, sup6, sup7 = net(inputs_test)
            time_sum += time_synchronized() - start

            # normalization
            pred = sup1[:, 0, :, :]
            pred = normPRED(pred)

            save_output(img_name_list[i_test], pred, prediction_dir)

    print("FPS: %f" % (1.0 / (time_sum / len(img_name_list))))
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
