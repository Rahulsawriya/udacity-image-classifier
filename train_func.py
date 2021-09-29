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

# load data and creating the dataset
def record_loader(data, train=True):
    re_loader = torch.utils.data.DataLoader(data, batch_size=50, shuffle=True) if train else torch.utils.data.DataLoader(data, batch_size=50)
    return re_loader

# This function will train and validate the model
def trainer_model(model, tr_loader, v_loader, device, 
                  criterion, optimizer, epochs, print_all, steps):   
    

    for i in range(epochs):
        rloss = 0
        model.train() 
        for aa, (inputs, labels) in enumerate(tr_loader):
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            rloss += loss.item()
        
            if steps % print_all == 0:
                model.eval()
                with torch.no_grad():
                    # This functionality will validate the model
                    tloss, accuracy = 0, 0
                    for bb, (inputs, labels) in enumerate(v_loader):
                        inputs, labels = inputs.to(device), labels.to(device)
                        output = model.forward(inputs)
                        tloss += criterion(output, labels).item()
                        ps = torch.exp(output)
                        similarity = (labels.data == ps.max(dim=1)[1])
                        accuracy += similarity.type(torch.FloatTensor).mean()
                        v_loss, accuracy = tloss, accuracy
                    # validation end
            
                print("Epoch: {}/{} || ".format(i+1, epochs),
                     "Training Loss: {:.5f} || ".format(rloss/print_all),
                     "Validation Loss: {:.5f} || ".format(v_loss/len(v_loader)),
                     "Validation Accuracy: {:.5f}".format(accuracy/len(v_loader)))
            
                rloss = 0
                model.train()

    return model

