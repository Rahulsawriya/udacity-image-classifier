#This function will transform the data
import torch
from torchvision import datasets, transforms
def data_transformer(t_dir):
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

