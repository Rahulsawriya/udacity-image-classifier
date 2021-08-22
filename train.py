#@author: Rahul Sawriya

import argparse
import torch
from collections import OrderedDict
from os.path import isdir
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models

def arg_parser():
    parser = argparse.ArgumentParser(description="Neural Network Setup by Rahul")
    parser.add_argument('--arch',type=str, help='Choose models as str from torchvision')
    parser.add_argument('--save_dir', type=str, help='define checkpoint directory')
    parser.add_argument('--learning_rate', type=float, help='learning rate')
    parser.add_argument('--hidden_units', type=int, help='hidden units for classifier')
    parser.add_argument('--epochs', type=int, help='epochs for training')
    parser.add_argument('--gpu', action="store_true", help='for gpu and cpu ')
    args = parser.parse_args()
    return args

def tr_transformer(tr_dir):
    #print(tr_dir)
    tr_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
    #print("kkk")
    tr_data = datasets.ImageFolder(tr_dir, transform=tr_transforms)
    return tr_data
       

def te_transformer(te_dir):
    te_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
    te_data = datasets.ImageFolder(te_dir, transform=te_transforms)
    return te_data

def gpu_exists(gpu_arg):
    if not gpu_arg:
        return torch.device("cpu")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if device == "cpu":
        print("CUDA is not on device, using CPU instead.")
    return device
    
def record_loader(data, train=True):
    if train: 
        re_loader = torch.utils.data.DataLoader(data, batch_size=50, shuffle=True)
    else: 
        re_loader = torch.utils.data.DataLoader(data, batch_size=50)
    return re_loader

def p_load_model(arch="vgg16"):
    if type(arch) == type(None): 
        model = models.vgg16(pretrained=True)
        model.name = "vgg16"
        print("Default Network arch specified as vgg16.")
    elif arch == "vgg13":
        model = models.vgg13(pretrained=True)
        model.name = "vgg13"
        print("Network arch specified as vgg13.")
    else:
        print('select either vgg13 or vgg16 to build the model')
    
    for param in model.parameters():
        param.requires_grad = False 
    return model

def init_classifier(model, hidden_units):
    if type(hidden_units) == type(None): 
        hidden_units = 4096
        print("Default Number of Hidden Units specificed as 4096.")
    
    in_features = model.classifier[0].in_features
    
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(in_features, hidden_units, bias=True)),
                          ('relu1', nn.ReLU()),
                          ('dropout1', nn.Dropout(p=0.5)),
                          ('fc2', nn.Linear(hidden_units, 102, bias=True)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    return classifier

def validation(model, te_loader, criterion, device):
    test_loss = 0
    accuracy = 0
    
    for ii, (inputs, labels) in enumerate(te_loader):
        
        inputs, labels = inputs.to(device), labels.to(device)
        
        output = model.forward(inputs)
        test_loss += criterion(output, labels).item()
        
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    return test_loss, accuracy

def network_trainer(model, tr_loader, v_loader, device, 
                  criterion, optimizer, epochs, print_every, steps):
    if type(epochs) == type(None):
        epochs = 5
        print("Default Number of Epochs specificed as 5.")    
 
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
                    valid_loss, accuracy = validation(model, v_loader, criterion, device)
            
                print("Epoch: {}/{} | ".format(e+1, epochs),
                     "Training Loss: {:.4f} | ".format(running_loss/print_every),
                     "Validation Loss: {:.4f} | ".format(valid_loss/len(v_loader)),
                     "Validation Accuracy: {:.4f}".format(accuracy/len(v_loader)))
            
                running_loss = 0
                model.train()

    return model

def v_model(Model, te_loader, Device):
    correct = 0
    total = 0
    with torch.no_grad():
        Model.eval()
        for data in te_loader:
            images, labels = data
            images, labels = images.to(Device), labels.to(Device)
            outputs = Model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print('Network test images accuracy: %d%%' % (100 * correct / total))

def init_checkpoint(Model, Save_Dir, Train_data):
       
    if type(Save_Dir) == type(None):
        print("Model checkpoint directory not specified, model will not be saved.")
    else:
        if isdir(Save_Dir):
            Model.class_to_idx = Train_data.class_to_idx
            
            checkpoint = {'arch': Model.name,
                          'classifier': Model.classifier,
                          'class_to_idx': Model.class_to_idx,
                          'state_dict': Model.state_dict()}
            
            torch.save(checkpoint, 'checkpoint.pth')

        else: 
            print("Model will not be saved, Directory not found.")

def main():
     
    args = arg_parser()
    data_dir = 'flowers'
    tr_dir, v_dir,te_dir = data_dir + '/train', data_dir + '/valid', data_dir + '/test'
    tr_data, v_data, te_data  = te_transformer(tr_dir), tr_transformer(v_dir), tr_transformer(te_dir)  
    tr_loader, v_loader, te_loader   = record_loader(tr_data), record_loader(v_data, train=False), record_loader(te_data, train=False)    
    model = p_load_model(arch=args.arch)
    model.classifier = init_classifier(model, 
                                         hidden_units=args.hidden_units)

    device = gpu_exists(gpu_arg=args.gpu);
    model.to(device);

    if type(args.learning_rate) == type(None):
        learning_rate = 0.001
        print("Default Learning rate specificed as 0.001")
    else: learning_rate = args.learning_rate
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    print_every, steps = 30, 0
    trained_model = network_trainer(model, tr_loader, v_loader, 
                                  device, criterion, optimizer, args.epochs, 
                                  print_every, steps)
    
    print("\nTraining process is now complete!!")
    v_model(trained_model, te_loader, device)
    init_checkpoint(trained_model, args.save_dir, tr_data)

if __name__ == '__main__':
    main()