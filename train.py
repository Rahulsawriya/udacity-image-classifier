#Rahul Sawriya

import argparse
import torch
from os.path import isdir
from torch import nn, optim
from collections import OrderedDict
from torchvision import models
from processor import pro_check
from train_func import data_transformer, record_loader, trainer_model

def main():
    
    #command line arguments
    parser = argparse.ArgumentParser(description="Neural Network Setup by Rahul")
    parser.add_argument('--arch',type=str, help='Choose model for training', default="vgg16")
    parser.add_argument('--save_dir', type=str, help='checkpoint directory')
    parser.add_argument('--learning_rate', type=float, help='learning rate', default=0.001)
    parser.add_argument('--hidden_units', type=int, help='hidden units', default=4096)
    parser.add_argument('--epochs', type=int, help='number of epochs for training', default=5)
    parser.add_argument('--gpu', action="store_true", help='use processor gpu or cpu', default="gpu")
    args = parser.parse_args()
     
    data_dir = 'flowers/'
    tr_dir, v_dir,te_dir = data_dir + 'train', data_dir + 'valid', data_dir + 'test'
    tr_data, v_data, te_data  = data_transformer(tr_dir), data_transformer(v_dir), data_transformer(te_dir)  
    tr_loader, v_loader, te_loader   = record_loader(tr_data), record_loader(v_data, train=False), record_loader(te_data, train=False)
    #This functionlity will download model from torchvision
    arch = args.arch
    if arch == "vgg16": 
        model = models.vgg16(pretrained=True)
        print("Using model vgg16.")
    elif arch == "vgg13":
        model = models.vgg13(pretrained=True)
        print("Using model vgg13.")
    else:
        print('select either vgg16 or vgg13 to build the model')
    model.name = arch
    for param in model.parameters():
        param.requires_grad = False
    #functionlity end
    in_features = model.classifier[0].in_features
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(in_features, args.hidden_units, bias=True)),
                          ('relu1', nn.ReLU()),
                          ('dropout1', nn.Dropout(p=0.5)),
                          ('fc2', nn.Linear(args.hidden_units, 102, bias=True)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    model.classifier = classifier
    #checking which processor exists.
    device = pro_check(gpu_arg=args.gpu);
    model.to(device);
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    
    print_all, steps = 30, 0
    print("Hidden_units {}".format(args.hidden_units))
    print("epochs {}".format(args.epochs))
    print("learning_rate {}".format(args.learning_rate))
    print("Training process initializing .....\n")
    trained_model = trainer_model(model, tr_loader, v_loader, 
                                  device, criterion, optimizer, args.epochs, 
                                  print_all, steps)
    
    print("\nTraining process is now completed!!")
    #This functionality will validate the model with test images
    correct, total = 0, 0
    with torch.no_grad():
        trained_model.eval()
        for data in te_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = trained_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print('Network test images accuracy in percentage: %d%%' % (100 * correct / total))
    #end
    #it will save the model into checkpoint folder
    if type(args.save_dir) == type(None):
        print("Model checkpoint directory not specified, model will not be saved.")
    else:
        if isdir(args.save_dir):
            trained_model.class_to_idx = tr_data.class_to_idx    
            checkpoint = {'arch': trained_model.name, 'classifier': trained_model.classifier, 'class_to_idx': trained_model.class_to_idx, 
                          'state_dict': trained_model.state_dict()}
            torch.save(checkpoint, args.save_dir+'checkpoint.pth')
        else:
            print("Model will not be saved, Directory not found.")
    #checkpoint end

if __name__ == '__main__':
    main()