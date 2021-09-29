#Rahul Sawriya
from torchvision import models
import PIL
import math
import torch
import argparse
import json
import numpy as np
from processor import pro_check

# print probability of flower
def probability(probs, flowers):
    for i, j in enumerate(zip(flowers, probs)):
        print ("Rank {}:".format(i+1), "Flower: {}, Probability: {}%".format(j[1], math.ceil(j[0]*100)))


def main():
    #command line arguments
    parser = argparse.ArgumentParser(description="Neural Network setup")
    parser.add_argument('--image', type=str, help='image file', required=True)
    parser.add_argument('--checkpoint', type=str, help='checkpoint file', required=True, default="checkpoint/checkpoint.pth")
    parser.add_argument('--top_k', type=int, help='top number of flowers', default=5) 
    parser.add_argument('--category_names', type=str, help='real names')
    parser.add_argument('--gpu', action="store_true", help='use processor gpu or cpu', default="gpu")
    args = parser.parse_args()
    with open(args.category_names, 'r') as f:
        	cat_to_name = json.load(f)

    checkpoint = torch.load(args.checkpoint)
    if checkpoint['arch'] == 'vgg16':
        model = models.vgg16(pretrained=True)
        model.name = "vgg16"
    else: 
        model = models.vgg13(pretrained=True)
        model.name = "vgg13"

    for param in model.parameters(): param.requires_grad = False
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])

    #====== image process start===
    t_image = PIL.Image.open(args.image)
    width, height = t_image.size

    if width < height: resize_image=[256, 256**600]
    else: resize_image=[256**600, 256]     
    t_image.thumbnail(size=resize_image)
    center = width/4, height/4
    left, top, right, bottom = center[0]-(244/2), center[1]-(244/2), center[0]+(244/2), center[1]+(244/2)
    t_image = t_image.crop((left, top, right, bottom))
    np_image = np.array(t_image)/255
    n_means, n_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    np_image = (np_image-n_means)/n_std
    np_image = np_image.transpose(2, 0, 1)
    #====== image process end ====
    image_tensor = np_image
    #checking which processor exists.
    device = pro_check(gpu_arg=args.gpu);
    print("Top k flower {}".format(args.top_k))

    # prediction of image start
    model.eval();
    t_image = torch.from_numpy(np.expand_dims(image_tensor, 
                                                  axis=0)).type(torch.FloatTensor)
    model=model.cpu()
    lprobs = model.forward(t_image)
    li_probs = torch.exp(lprobs)
    t_probs, t_labels = li_probs.topk(args.top_k)
    t_probs, t_labels = np.array(t_probs.detach())[0], np.array(t_labels.detach())[0]
    idx_to_class = {val: key for key, val in    
                                      model.class_to_idx.items()}
    t_labels = [idx_to_class[lab] for lab in t_labels]
    t_flowers = [cat_to_name[lab] for lab in t_labels]
    # prediction of image end
    probability(t_flowers, t_probs)

if __name__ == '__main__': 
    main()
