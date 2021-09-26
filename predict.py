# Rahul Sawriya
from math import ceil
from torchvision import models
import PIL
import torch
import argparse
import json
import numpy as np
from processor import pro_check

# print probability of flower
def probability_print(probs, flowers):
    for i, j in enumerate(zip(flowers, probs)):
        print ("Rank {}:".format(i+1), "Flower: {}, liklihood: {}%".format(j[1], ceil(j[0]*100)))


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

    #model = output_checkpoint_load(args.checkpoint)
    checkpoint = torch.load(args.checkpoint)
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

    #====== image process start===
    t_image = PIL.Image.open(args.image)
    orig_width, orig_height = t_image.size

    if orig_width < orig_height: resize_size=[256, 256**600]
    else: resize_size=[256**600, 256]     
    t_image.thumbnail(size=resize_size)
    center = orig_width/4, orig_height/4
    left, top, right, bottom = center[0]-(244/2), center[1]-(244/2), center[0]+(244/2), center[1]+(244/2)
    t_image = t_image.crop((left, top, right, bottom))
    np_image = np.array(t_image)/255
    normalise_means = [0.485, 0.456, 0.406]
    normalise_std = [0.229, 0.224, 0.225]
    np_image = (np_image-normalise_means)/normalise_std
    np_image = np_image.transpose(2, 0, 1)
    #====== image process end ====
    image_tensor = np_image
    #checking which processor exists.
    device = pro_check(gpu_arg=args.gpu);
    print("Top k {}".format(args.top_k))
    """t_probs, t_labels, t_flowers = prediction(image_tensor, model, 
                                                 device, cat_to_name,
                                                 args.top_k)"""
    # prediction of image start
    model.eval();
    t_image = torch.from_numpy(np.expand_dims(image_tensor, 
                                                  axis=0)).type(torch.FloatTensor)
    model=model.cpu()
    lprobs = model.forward(t_image)
    li_probs = torch.exp(lprobs)

    t_probs, t_labels = li_probs.topk(args.top_k)
    t_probs = np.array(t_probs.detach())[0] 
    t_labels = np.array(t_labels.detach())[0]

    idx_to_class = {val: key for key, val in    
                                      model.class_to_idx.items()}
    t_labels = [idx_to_class[lab] for lab in t_labels]
    t_flowers = [cat_to_name[lab] for lab in t_labels]
    # prediction of image end
    probability_print(t_flowers, t_probs)

if __name__ == '__main__': 
    main()
