# Rahul Sawriya
import PIL
import torch
import argparse
import json
import numpy as np

from math import ceil
from train import gpu_exists
from torchvision import models

def arg_parser():
    parser = argparse.ArgumentParser(description="Neural Network Settings")

    parser.add_argument('--image', type=str, help='image file of prediction', required=True)
    parser.add_argument('--checkpoint', type=str, help='load checkpoint file', required=True)
    parser.add_argument('--top_k', type=int, help='choose top number of flowers') 
    parser.add_argument('--category_names', type=str, help='map with the real name')
    parser.add_argument('--gpu', action="store_true", help='Use GPU + Cuda for calculations')
    args = parser.parse_args()
    
    return args

def output_checkpoint_load(checkpoint_path):
    checkpoint = torch.load("checkpoint.pth")
    if checkpoint['arch'] == 'vgg16':
        model = models.vgg16(pretrained=True)
        model.name = "vgg16"
    else: 
        exec("model = models.{}(pretrained=True)".checkpoint['arch'])
        model.name = checkpoint['arch']

    for param in model.parameters(): param.requires_grad = False
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

def p_image(image_path):
    test_image = PIL.Image.open(image_path)
    orig_width, orig_height = test_image.size

    if orig_width < orig_height: resize_size=[256, 256**600]
    else: resize_size=[256**600, 256]     
    test_image.thumbnail(size=resize_size)
    center = orig_width/4, orig_height/4
    left, top, right, bottom = center[0]-(244/2), center[1]-(244/2), center[0]+(244/2), center[1]+(244/2)
    test_image = test_image.crop((left, top, right, bottom))
    np_image = np.array(test_image)/255

    normalise_means = [0.485, 0.456, 0.406]
    normalise_std = [0.229, 0.224, 0.225]
    np_image = (np_image-normalise_means)/normalise_std

    np_image = np_image.transpose(2, 0, 1)
    
    return np_image


def prediction(image_tensor, model, device, cat_to_name, top_k):
    if type(top_k) == type(None):
        top_k = 5
        print("if Top K is not specified, assume K=5")
    
    model.eval();
    t_image = torch.from_numpy(np.expand_dims(image_tensor, 
                                                  axis=0)).type(torch.FloatTensor)

    model=model.cpu()
    lprobs = model.forward(t_image)
    li_probs = torch.exp(lprobs)

    t_probs, t_labels = li_probs.topk(top_k)
    t_probs = np.array(t_probs.detach())[0] 
    t_labels = np.array(t_labels.detach())[0]

    idx_to_class = {val: key for key, val in    
                                      model.class_to_idx.items()}
    t_labels = [idx_to_class[lab] for lab in t_labels]
    t_flowers = [cat_to_name[lab] for lab in t_labels]
    
    return t_probs, t_labels, t_flowers


def probability_print(probs, flowers):
    for i, j in enumerate(zip(flowers, probs)):
        print ("Rank {}:".format(i+1),
               "Flower: {}, liklihood: {}%".format(j[1], ceil(j[0]*100)))

def main():
    args = arg_parser()
    with open(args.category_names, 'r') as f:
        	cat_to_name = json.load(f)

    model = output_checkpoint_load(args.checkpoint)
    image_tensor = p_image(args.image)
    
    device = gpu_exists(gpu_arg=args.gpu);
    t_probs, t_labels, t_flowers = prediction(image_tensor, model, 
                                                 device, cat_to_name,
                                                 args.top_k)

    probability_print(t_flowers, t_probs)

if __name__ == '__main__': main()
