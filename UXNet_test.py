import glob
import os
import time
import numpy as np
import torch
from PIL import Image
from skimage import io
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchsummary import summary
from torchvision import transforms

from UXNet_model import UXNet
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
    image = io.imread(image_name)
    imo = im.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)

    pb_np = np.array(imo)

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1, len(bbb)):
        imidx = imidx + "." + bbb[i]

    imo.save(d_dir + imidx + '.png')


def main():
    # set image path and model name
    model_name = 'UXNet'

    image_dir = os.path.join(os.getcwd(), 'test_data', 'test_images')
    prediction_dir = os.path.join(os.getcwd(), 'test_data', model_name + '_results' + os.sep)
    model_dir = os.path.join(os.getcwd(), 'test_model', model_name + '.pth')

    img_name_list = glob.glob(image_dir + os.sep + '*')

    # dataloader
    test_salobj_dataset = SalObjDataset(img_name_list=img_name_list,
                                        lbl_name_list=[],
                                        transform=transforms.Compose([RescaleT(320),
                                                                      ToTensorLab(flag=0)])
                                        )
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=4)

    net = UXNet(3, 1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)

    optimizer = optim.Adam(net.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)

    summary(net, (3, 320, 320))

    # net.load_state_dict(torch.load(model_dir))

    checkpoint = torch.load((model_dir))
    net.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']

    net.eval()

    print('Start inference!')
    res = []

    with torch.no_grad():
        # inference for each image

        for i_test, data_test in enumerate(test_salobj_dataloader):

            print("testing:", img_name_list[i_test].split(os.sep)[-1])

            inputs_test = data_test['image']
            inputs_test = inputs_test.type(torch.FloatTensor)

            if torch.cuda.is_available():
                inputs_test = Variable(inputs_test.cuda())
            else:
                inputs_test = Variable(inputs_test)

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
