#@author: Rahul Sawriya

import argparse
import torch
from os.path import isdir
from torch import nn
from collections import OrderedDict
from torch import optim
from torchvision import datasets, transforms, models
from processor import pro_check


def transformer(t_dir):
    if "train" in t_dir:
        nt_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
    elif "valid" in t_dir or "test" in t_dir:
        nt_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
    data = datasets.ImageFolder(t_dir, transform=nt_transforms)
    return data

# load data
def record_loader(data, train=True):
    re_loader = torch.utils.data.DataLoader(data, batch_size=50, shuffle=True) if train else torch.utils.data.DataLoader(data, batch_size=50)
    return re_loader

def network_trainer(model, tr_loader, v_loader, device, 
                  criterion, optimizer, epochs, print_every, steps):   
    print("Training process initializing .....\n")

    for e in range(epochs):
        running_loss = 0
        model.train() 
        for ii, (inputs, labels) in enumerate(tr_loader):
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
            if steps % print_every == 0:
                model.eval()
                with torch.no_grad():
                    #valid_loss, accuracy = validation(model, v_loader, criterion, device)
                    # validation start
                    test_loss = 0
                    accuracy = 0
                    for ii, (inputs, labels) in enumerate(v_loader):
                        inputs, labels = inputs.to(device), labels.to(device)
                        output = model.forward(inputs)
                        test_loss += criterion(output, labels).item()
                        ps = torch.exp(output)
                        equality = (labels.data == ps.max(dim=1)[1])
                        accuracy += equality.type(torch.FloatTensor).mean()
                        valid_loss, accuracy = test_loss, accuracy
                    # validation end
            
                print("Epoch: {}/{} | ".format(e+1, epochs),
                     "Training Loss: {:.4f} | ".format(running_loss/print_every),
                     "Validation Loss: {:.4f} | ".format(valid_loss/len(v_loader)),
                     "Validation Accuracy: {:.4f}".format(accuracy/len(v_loader)))
            
                running_loss = 0
                model.train()

    return model


def main():
    
    #command line parameters
    parser = argparse.ArgumentParser(description="Neural Network Setup by Rahul")
    parser.add_argument('--arch',type=str, help='Choose models as str from torchvision', default="vgg16")
    parser.add_argument('--save_dir', type=str, help='checkpoint directory')
    parser.add_argument('--learning_rate', type=float, help='learning rate', default=0.001)
    parser.add_argument('--hidden_units', type=int, help='hidden units for classifier', default=4096)
    parser.add_argument('--epochs', type=int, help='number of epochs for training', default=1)
    parser.add_argument('--gpu', action="store_true", help='use processor gpu or cpu', default="gpu")
    args = parser.parse_args()
     
    data_dir = 'flowers/'
    tr_dir, v_dir,te_dir = data_dir + 'train', data_dir + 'valid', data_dir + 'test'
    tr_data, v_data, te_data  = transformer(tr_dir), transformer(v_dir), transformer(te_dir)  
    tr_loader, v_loader, te_loader   = record_loader(tr_data), record_loader(v_data, train=False), record_loader(te_data, train=False)    
    #model = p_load_model(arch=args.arch)
    # p_load_model start
    arch = args.arch
    if arch == "vgg16": 
        model = models.vgg16(pretrained=True)
        model.name = "vgg16"
        print("Network arch specified as vgg16.")
    elif arch == "vgg13":
        model = models.vgg13(pretrained=True)
        model.name = "vgg13"
        print("Network arch specified as vgg13.")
    elif arch == "alexnet:
        model = models.alexnet(pretrained=True)
    else:
        print('select either vgg13 or vgg16 to build the model')
    for param in model.parameters():
        param.requires_grad = False
    # p_load_model_end
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
    
    print_every, steps = 30, 0
    print("Hidden_units {}".format(args.hidden_units))
    print("epochs {}".format(args.epochs))
    print("learning_rate {}".format(args.learning_rate))
    trained_model = network_trainer(model, tr_loader, v_loader, 
                                  device, criterion, optimizer, args.epochs, 
                                  print_every, steps)
    
    print("\nTraining process is now complete!!")
    #v_model(trained_model, te_loader, device)
    # function v_model start
    correct = 0
    total = 0
    with torch.no_grad():
        trained_model.eval()
        for data in te_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = trained_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print('Network test images accuracy: %d%%' % (100 * correct / total))
    # function v_model end
    #init_checkpoint(trained_model, args.save_dir, tr_data)
    # function init_checkpoint start
    if type(args.save_dir) == type(None):
        print("Model checkpoint directory not specified, model will not be saved.")
    else:
        if isdir(args.save_dir):
            trained_model.class_to_idx = tr_data.class_to_idx    
            checkpoint = {'arch': trained_model.name, 'classifier': trained_model.classifier, 'class_to_idx': trained_model.class_to_idx, 
                          'state_dict': trained_model.state_dict()}
            torch.save(checkpoint, 'checkpoint/checkpoint.pth')
        else:
            print("Model will not be saved, Directory not found.")
    # function init_checkpoint end

if __name__ == '__main__':
    main()