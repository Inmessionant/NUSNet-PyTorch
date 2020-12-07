import glob
import os
import time
import cv2
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from NUSNet_model.NUSNet import *
from data_loader import RescaleT
from data_loader import SalObjDataset
from data_loader import ToTensorLab


# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d - mi) / (ma - mi)

    return dn


def save_output(image_name, pred, d_dir):
    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()
    im = Image.fromarray(predict_np * 255).convert('RGB')
    img_name = image_name.split(os.sep)[-1]
    image = cv2.imread(image_name)
    imo = im.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)
    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]

    for i in range(1, len(bbb)):
        imidx = imidx + "." + bbb[i]

    imo.save(d_dir + imidx + '.png')


# change gpus，model_name，pre_data_dir,  net, num_workers
def main():
    gpus = [0]
    torch.cuda.set_device('cuda:{}'.format(gpus[0]))
    model_name = 'NUSNet'
    pre_data_dir = 'SOD'  # 'TUDS-TE'   'PASCAL'   'HKU'

    # Models : NUSNetNet  NUSNetNet4  NUSNetNet5  NUSNetNet6  NUSNetNet7  NUSNetNetCAM  NUSNetNetSAM  NUSNetNetCBAM
    # NUSNetNet765CAM4SMALLSAM
    net = NUSNet(3, 1).cuda()  # input channels and output channels
    net = nn.DataParallel(net, device_ids=gpus, output_device=gpus[0])
    print(summary(net, (3, 320, 320)))

    image_dir = os.path.join(os.getcwd(), 'test_data', pre_data_dir)
    prediction_dir = os.path.join(os.getcwd(), 'test_data', pre_data_dir + '_Results', model_name + os.sep)
    model_dir = os.path.join(os.getcwd(), 'saved_models', model_name + '.pth')
    img_name_list = glob.glob(image_dir + os.sep + '*')

    net.load_state_dict(torch.load(model_dir))

    # dataloader
    test_salobj_dataset = SalObjDataset(img_name_list=img_name_list, lbl_name_list=[],
                                        transform=transforms.Compose([RescaleT(320), ToTensorLab(flag=0)]))
    test_salobj_dataloader = DataLoader(test_salobj_dataset, batch_size=1, shuffle=False, num_workers=8,
                                        pin_memory=True)
    res = []

    print('Start inference!')
    net.eval()

    with torch.no_grad():
        # inference for each image
        for i_test, data_test in enumerate(test_salobj_dataloader):

            print("testing:", img_name_list[i_test].split(os.sep)[-1])

            inputs_test = data_test['image']
            inputs_test = inputs_test.type(torch.FloatTensor)

            inputs_test = inputs_test.cuda(non_blocking=True)

            torch.cuda.synchronize()
            start = time.time()
            sup1, sup2, sup3, sup4, sup5, sup6, sup7 = net(inputs_test)
            torch.cuda.synchronize()
            end = time.time()
            res.append(end - start)

            # normalization
            pred = sup1[:, 0, :, :]
            pred = normPRED(pred)

            # save results to test_results folder
            if not os.path.exists(prediction_dir):
                os.makedirs(prediction_dir, exist_ok=True)

            save_output(img_name_list[i_test], pred, prediction_dir)

        time_sum = 0

        for i in res:
            time_sum += i
        print("FPS: %f" % (1.0 / (time_sum / len(res))))

        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
