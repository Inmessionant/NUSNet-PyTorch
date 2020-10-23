import glob
import os
import time
import random
import numpy as np
import torch
import cv2
from PIL import Image
from torch import optim, nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchsummary import summary
from torchvision import transforms
from UXNet_model import UXNet
from data_loader import RescaleT
from data_loader import SalObjDataset
from data_loader import ToTensorLab


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True  # 固定随机性
    torch.backends.cudnn.benchmark = True  # 尺寸大小一样可以加速训练


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
    pb_np = np.array(imo)
    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]

    for i in range(1, len(bbb)):
        imidx = imidx + "." + bbb[i]

    imo.save(d_dir + imidx + '.png')


def main():
    setup_seed(1222)
    model_name = 'UXNet'
    gpus = [0, 1, 2, 3]
    torch.cuda.set_device('cuda:{}'.format(gpus[0]))
    net = UXNet(3, 1)
    net = net.cuda()
    net = nn.DataParallel(net)
    summary(net, (3, 320, 320))

    optimizer = optim.Adam(net.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)

    image_dir = os.path.join(os.getcwd(), 'test_data', 'test_images')
    prediction_dir = os.path.join(os.getcwd(), 'test_data', model_name + '_results' + os.sep)
    model_dir = os.path.join(os.getcwd(), 'test_model', model_name + '.pth')
    img_name_list = glob.glob(image_dir + os.sep + '*')

    checkpoint = torch.load(model_dir)
    net.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    # dataloader
    test_salobj_dataset = SalObjDataset(img_name_list=img_name_list, lbl_name_list=[],
                                        transform=transforms.Compose([RescaleT(320), ToTensorLab(flag=0)]))
    test_salobj_dataloader = DataLoader(test_salobj_dataset, batch_size=1, shuffle=False, num_workers=4,
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

            inputs_test = Variable(inputs_test.cuda(non_blocking=True), requires_grad=False)

            torch.cuda.synchronize()
            start = time.time()
            d1, d2, d3, d4, d5, d6, d7 = net(inputs_test)
            torch.cuda.synchronize()
            end = time.time()
            res.append(end - start)

            # normalization
            pred = d1[:, 0, :, :]
            pred = normPRED(pred)

            # save results to test_results folder
            if not os.path.exists(prediction_dir):
                os.makedirs(prediction_dir, exist_ok=True)
            save_output(img_name_list[i_test], pred, prediction_dir)

        torch.cuda.empty_cache()

        time_sum = 0
        for i in res:
            time_sum += i
        print("FPS: %f" % (1.0 / (time_sum / len(res))))


if __name__ == "__main__":
    main()
