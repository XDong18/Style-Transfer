import torch
from net import vgg19_features
from loss import content_loss, style_loss
import argparse
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import time
import cv2
import numpy as np
import random
from os.path import split


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

def gaussian_noise(shape):
    mean = 0
    var = 1
    noise = np.random.normal(mean, var ** 0.5, shape)
    noise -= noise.min()
    noise = noise / noise.max() * 255.
    noise = noise.astype(np.uint8)
    noise = Image.fromarray(noise)
    return noise


def save_img(epoch, output):
    args = arg_parser()
    un_trans = transforms.ToPILImage()
    img = output.cpu().detach()
    img.data.clamp_(0, 1)
    img = img.squeeze(0)
    un_trans = transforms.ToPILImage()
    img = un_trans(img)
    # img_array = np.array(img)
    # print(img_array.max())
    # img_array = img_array * np.array(STD).reshape(1, 1, -1)
    # img_array += np.array(MEAN).reshape(1, 1, -1)
    # img = Image.fromarray(img_array.astype(np.uint8))
    img.save('./output/{0}_{1}_output.jpg'.format(split(args.output)[-1].split('.')[0], epoch))

def pre_image(img):
    args = arg_parser()
    trans = transforms.Compose([transforms.Resize(args.size),transforms.ToTensor()])
    out = trans(img)
    out = out.unsqueeze(0)
    return out

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--content_weight', default=1e-3, type=float)
    parser.add_argument('--content')
    parser.add_argument('--style')
    parser.add_argument('--output', default='./output/output.jpg')
    parser.add_argument('--epoch', default=1000, type=int)
    parser.add_argument('--size', default=512, type=int)
    parser.add_argument('--lr', default=1e6, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--log', default=10, type=int)
    parser.add_argument('--save', default=False, type=bool)
    parser.add_argument('--save_fre', default=100, type=int)
    return parser.parse_args()


def train(model, img, art, photo, epoch_num, device, content_name_list, style_name_list):
    args = arg_parser()
    features = vgg19_features(model, content_name_list, style_name_list, device)
    optimizer = torch.optim.SGD([img.requires_grad_()], lr=args.lr, momentum=args.momentum)
    _, art_style = features.extract_features(art)
    art_style = [i_style.detach() for i_style in art_style]
    photo_content, _ = features.extract_features(photo)
    photo_content = [i_content.detach() for i_content in photo_content]
    for epoch in range(epoch_num):
        end_time = time.time()
        img_content, img_style = features.extract_features(img)
        C_loss = content_loss(img_content, photo_content)
        S_loss = style_loss(img_style, art_style)
        loss = C_loss * args.content_weight + S_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % args.log == 0:
            print('[{0}/{1}]\ttime:{time:.2f}\tloss:{loss:.4f}'.format(epoch, epoch_num,\
                    time=time.time()-end_time, loss=loss.item()*1e6))
            print(C_loss.item(), S_loss.item())
        
        if epoch % args.save_fre == 0:
            save_img(epoch, img)
    
    img.data.clamp_(0, 1)
    return img

def main():
    args = arg_parser()
    device = torch.device("cuda")
    vgg19 = models.vgg19()
    checkpoint = torch.load('checkpoint/vgg19-dcbb9e9d.pth')
    vgg19.load_state_dict(checkpoint)
    art = Image.open(args.style)
    photo = Image.open(args.content)
    trans = transforms.Compose([transforms.Resize((args.size, args.size)),transforms.ToTensor()])
    art = trans(art).to(device, torch.float)
    art = art.unsqueeze(0)
    photo = trans(photo).to(device, torch.float)
    photo = photo.unsqueeze(0)
    img = photo.clone() #TODO 
    # img = gaussian_noise((args.size, args.size, 3))
    # img = pre_image(img).to(device, torch.float)

    content_name_list = ['conv4_1']
    style_name_list = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
    # style_name_list = ['conv1_1']
    output = train(vgg19, img, art, photo, args.epoch, device, content_name_list, style_name_list)
    output = output.cpu().detach()
    output = output.squeeze(0)
    un_trans = transforms.ToPILImage()
    output = un_trans(output)
    output.save(args.output)

if __name__ == "__main__":
    main()
    





