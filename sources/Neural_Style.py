import warnings
warnings.filterwarnings('ignore')

from scipy.misc import imresize
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

import PIL
from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models


# cuda

#use_cuda = torch.cuda.is_available()
use_cuda = False
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

# images

content_image_path = "./deep-photo-styletransfer/examples/input/in1.png"
style_image_path = "./deep-photo-styletransfer/examples/style/tar1.png"
content_segment_image_path = "./deep-photo-styletransfer/examples/segmentation/in1.png"
style_segment_image_path = "./deep-photo-styletransfer/examples/segmentation/tar1.png"

imsize = 200
img_size = Image.open(content_image_path).size  # desired size of the output image

loader = transforms.Compose([
    transforms.Scale(imsize),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor

def float_loader(image):
    image = Variable(torch.from_numpy(image))
    image = image.unsqueeze(0)
    return image.type(dtype)

def image_loader(image_name):
    image = Image.open(image_name).resize((img_size))
    image = np.array(image)
    image = np.swapaxes(image, 0, 2)
    image = np.swapaxes(image, 1, 2)
    return image

style = float_loader(image_loader(style_image_path))
content = float_loader(image_loader(content_image_path))
content_seg = image_loader(content_segment_image_path)/255
style_seg = image_loader(style_segment_image_path)/255

color_vals = {
        'green' : (0.1, -0.1, 0.1),
        'white' : (-0.1, -0.1, -0.1),
        'black' : (0.1, 0.1, 0.1),
        'red': (-0.1, 0.1, 0.1),
        'blue': (0.1, 0.1, -0.1),
        'yellow': (-0.1, -0.1, 0.1),
        'lightblue': (0.1, -0.1, -0.1),
        'purple': (-0.1, 0.1, -0.1),
        'grey': (0.4, 0.4, 0.4, -0.4, -0.4,  -0.4)
}

def extract_mask(image, color_mode):
    mode = color_vals[color_mode]
    mask = np.ones([img_size[1], img_size[0]])

    for index, ind in enumerate(mode):
        i = index % 3
        if ind>0:
            mask *= (image[i,:,:] < ind)
        if ind<0:
            mask *= (image[i,:,:] > (1-ind))
    return mask


color_content_masks, color_style_masks = [], []

for color_code in color_vals.keys():
    content_mask_j = extract_mask(content_seg, color_code)
    seg_mask_j = extract_mask(style_seg, color_code)

    color_content_masks.append(float_loader(content_mask_j))
    color_style_masks.append(float_loader(seg_mask_j))

unloader = transforms.ToPILImage()  # reconvert into PIL image

def imshow(tensor):
    image = tensor.clone().cpu()  # we clone the tensor to not do changes on it
    image = image.view(3, imsize, imsize)  # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)


class TVLoss(nn.Module):
    def __init__(self, strength):
        super(TVLoss, self).__init__()
        self.strength = strength
        self.x_diff = torch.Tensor()
        self.y_diff = torch.Tensor()

    def updateOutput(self, input):
        self.output = input
        return self.output

    def updateGradInput(self, input, gradOutput):
        self.gradInput.resize_as_(input).zero_()
        C, H, W = input.size(0), input.size(1), input.size(2)
        self.x_diff.resize_(3, H-1, W-1)
        self.y_diff.resize_(3, H-1, W-1)

        self.x_diff.copy_(input[:, :-1, :-1])
        self.x_diff.add_(-1, input[:, :-1, 1:])
        self.y_diff.copy_(input[:, :-1, :-1])
        self.y_diff.add_(-1, input[:, 1:, :-1])

        self.gradInput[:, :-1, :-1].add_(self.x_diff).add(self.y_diff)
        self.gradInput[:, :-1, 1:].add_(-1, self.x_diff)
        self.gradInput[:, 1:, 1:].add_(-1, self.y_diff)

        self.gradInput.mul_(self.strength)
        self.gradInput.add_(gradOutput)
        return self.gradInput


# content loss
class ContentLoss(nn.Module):
    def __init__(self, target, weight):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        self.target = target.detach() * weight
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.weight = weight
        self.criterion = nn.MSELoss()

    def forward(self, input):
        self.loss = self.criterion.forward(input * self.weight, self.target)
        self.output = input
        return self.output

    def backward(self, retain_variables=True):
        self.loss.backward(retain_variables=retain_variables)
        return self.loss

# style loss


class GramMatrix(nn.Module):
    def forward(self, input):
        a, b, c, d = input.size()  # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)

        features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

        G = torch.mm(features, features.t())  # compute the gram product

        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(a * b * c * d)


class StyleLoss(nn.Module):
    def __init__(self, target, weight):
        super(StyleLoss, self).__init__()
        self.target = target.detach() * weight
        self.weight = weight
        self.gram = GramMatrix()
        self.criterion = nn.MSELoss()

    def forward(self, input):
        self.output = input.clone()
        self.G = self.gram.forward(input)
        self.G.mul_(self.weight)
        self.loss = self.criterion.forward(self.G, self.target)
        return self.output

    def backward(self, retain_variables=True):
        self.loss.backward(retain_variables=retain_variables)
        return self.loss

# load the cnn and build the model

class StyleLossWithSeg(nn.Module):
    def __init__(self, strength, target_grams, color_content_masks):
        super(StyleLossWithSeg, self).__init__()
        self.strength = strength
        self.target_grams = target_grams
        self.color_content_masks = color_content_masks
        self.color_codes = color_vals.keys()

        self.loss = 0
        self.gram = GramMatrix()
        self.crit = nn.MSELoss()

    def forward(self, input):
        self.output = input
        return self.output

    def backward(self, input, gradOutput):
        self.loss = 0
        self.gradInput = gradOutput.clone().zero_()

        for j in range(len(self.color_codes)):
            l_content_mask_ori = self.color_content_masks[j].clone()
            l_content_mask = l_content_mask_ori.repeat(1,1,1).expand_as(input)
            l_content_mean = l_content_mask_ori.mean()

            masked_input_features = l_content_mask * input
            masked_input_gram = self.gram.forward(masked_input_features).clone()
            if l_content_mean > 0:
                masked_input_gram /= input.nelement() * l_content_mean

            loss_j = self.crit.forward(maked_input_gram, self.target_grams[j])
            loss_j *= self.strength * l_content_mean
            self.loss += loss_j

            dG = self.crit.backward(masked_input_gram, self.target_grams[j])
            dG /= input.nelement()

            gradient = self.gram.backward(masked_input_features, dG)
            self.gradInput.add_(gradient)

        self.gradInput.mul_(self.strength)
        self.gradInput.add_(gradOutput)
        return self.gradInput


cnn = models.vgg16(pretrained=True).features

# move it to the GPU if possible:
if use_cuda:
    cnn = cnn

# desired depth layers to compute style/content losses :
content_layers = ['conv_4']
style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

# just in order to have an iterable access to or list of content/syle losses
content_losses = []
style_losses = []

model = nn.Sequential()  # the new Sequential module network
gram = GramMatrix()  # we need a gram module in order to compute style targets

# move these modules to the GPU if possible:
if use_cuda:
    model = model
    gram = gram

# weigth associated with content and style losses
content_weight = 1
style_weight = 1000

i = 1
for layer in list(cnn):
    if isinstance(layer, nn.Conv2d):
        name = "conv_" + str(i)
        model.add_module(name, layer)
        sap = nn.AvgPool2d((3,3), (1,1), (1,1))
        for k in range(len(color_vals.keys())):
            color_content_masks[k] = sap.forward(color_content_masks[k].repeat(1,1,1))[0].clone()
            color_style_masks[k] = sap.forward(color_style_masks[k].repeat(1,1,1))[0].clone()

        if name in content_layers:
            # add content loss:
            target = model.forward(content).clone()
            content_loss = ContentLoss(target, content_weight)
            model.add_module("content_loss_" + str(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            #gram
            target_feature = model.forward(style).clone()
            target_grams = []

            for k in range(len(color_vals.keys())):
                l_style_mask_ori = color_style_masks[k].clone()
                l_style_mask = l_style_mask_ori.repeat(1,1,1).expand(target_feature.size())
                l_style_mean = l_style_mask_ori.mean().mean().mean()
                l_style_mean = l_style_mean[0].data[0]

                masked_target_features = l_style_mask * target_feature
                masked_target_gram = gram.forward(masked_target_features).clone()
                if l_style_mean > 0:
                    masked_target_gram /= target_feature.nelement() * l_style_mean
                target_grams.append(masked_target_gram)

            loss_module = StyleLossWithSeg(style_weight, target_grams, color_content_masks)
            model.add_module('loss_module'+str(i),loss_module)
            style_losses.append(loss_module)

    if isinstance(layer, nn.ReLU):
        name = "relu_" + str(i)
        model.add_module(name, layer)

        i += 1

    if isinstance(layer, nn.MaxPool2d):
        name = "pool_" + str(i)
        model.add_module(name, layer)  # ***

        for k in range(len(color_vals.keys())):
            a1 =  (color_content_masks[k].size()[0] % 2)
            a2 =  (color_content_masks[k].size()[1] % 2)
            b1 =  (color_style_masks[k].size()[0] % 2)
            b2 =  (color_style_masks[k].size()[1] % 2)

            color_content_masks[k] = color_content_masks[k][a1::2, a1::2]
            color_style_masks[k] = color_style_masks[k][b1::2, b2::2]


print(model)

# input image

input = float_loader(image_loader(content_image_path))
input.data = torch.randn(input.data.size()).type(dtype)

# gradient descent

# this line to show that input is a parameter that requires a gradient
input = nn.Parameter(input.data)
optimizer = optim.LBFGS([input])

run = [0]
while run[0] <= 300:

    def closure():
        # correct the values of updated input image
        input.data.clamp_(0, 1)

        optimizer.zero_grad()
        model.forward(input)
        style_score = 0
        content_score = 0

        for sl in style_losses:
            style_score += sl.loss
        for cl in content_losses:
            content_score += cl.backward()

        run[0]+=1

        from torchvision.transforms import ToPILImage
        image = ToPILImage()(input.data[0,:,:,:])
        image.save('res_'+str(run[0]) + '.jpg')
        return style_score+content_score

    print run[0]
    optimizer.step(closure)

# a last correction...
input.data.clamp_(0, 1)

plt.subplot(224)
imshow(input.data)
plt.show()
