import os
import torch
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn
from tqdm import tqdm




class ConvBlock(nn.Module):
    def __init__(self,c_in_channels, c_out_channels, c_kernel_size, c_padding,
                 b_num_features,
                 m_kernal=None) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(in_channels=c_in_channels, out_channels=c_out_channels, kernel_size=c_kernel_size, padding=c_padding)
        self.r1 = nn.ReLU(inplace=True)
        self.b1 = nn.BatchNorm2d(b_num_features)
        self.m1 = nn.MaxPool2d(kernel_size=m_kernal)
    def forward(self,x):
        x = self.c1(x)
        x = self.r1(x)
        x = self.b1(x)
        x = self.m1(x)
        return x
    
class CatDogCNN(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        self.conv_layer_1 = ConvBlock(3,64,3,1,64,2)
        self.conv_layer_2 = ConvBlock(64,512,3,1,512,2)
        self.conv_layer_3 = ConvBlock(512,512,3,1,512,2)
        self.flatten = nn.Flatten()
        self.linear_layer_1 =nn.Linear(512*3*3,out_features=num_classes)
    def forward(self, x: torch.Tensor):
        x = self.conv_layer_1(x)
        x = self.conv_layer_2(x)
        x = self.conv_layer_3(x)
        x = self.conv_layer_3(x)
        x = self.conv_layer_3(x)
        x = self.conv_layer_3(x)
        x = self.flatten(x)
        x = self.linear_layer_1(x)
        return x



def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer):
    model.train()
    train_loss, train_acc = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item() 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        correct_predictions = (y_pred_class == y).sum().item()
        total_predictions = len(y_pred) 
        accuracy = correct_predictions / total_predictions
        train_acc += accuracy
        # train_acc += (y_pred_class == y).sum().item()/len(y_pred)
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module):
    model.eval() 
    test_loss, test_acc = 0, 0
    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            test_pred_logits = model(X)
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc

def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          epochs: int = 5):
    results = {"train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }
    
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer)
        test_loss, test_acc = test_step(model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn)
        
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
    return results



if __name__ == "__main__":
    torch.manual_seed(42) 
    torch.cuda.manual_seed(42)
    IMAGE_WIDTH = 255
    IMAGE_HEIGHT = 255
    IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)

    train_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.TrivialAugmentWide(),  # this one is a data augmentation technique
        transforms.ToTensor()])
    test_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor()])
    train_folder = 'data/training_set/training_set/'
    test_folder = 'data/test_set/test_set/'
    train_data_aug = ImageFolder(train_folder, transform = train_transform)
    test_data_aug = ImageFolder(test_folder, transform = test_transform)

    torch.manual_seed(42)
    train_dataloader_aug = torch.utils.data.DataLoader(train_data_aug, batch_size=8, shuffle=True, num_workers=2)
    test_dataloader_aug = torch.utils.data.DataLoader(test_data_aug, batch_size=8, shuffle=False, num_workers=2)
    model = CatDogCNN(2)
    device = "cpu"
    model.to(device=device)

    NUM_EPOCHS = 25

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)

    from timeit import default_timer as timer 
    start_time = timer()

    model_results = train(model=model,
                        train_dataloader=train_dataloader_aug,
                        test_dataloader=test_dataloader_aug,
                        optimizer=optimizer,
                        loss_fn=loss_fn,
                        epochs=NUM_EPOCHS)
    end_time = timer()
    print(f"Total training time: {end_time-start_time:.3f} seconds")

    torch.save(model.state_dict(),"model/cat_dog_full_cnn_25_epoch.pth")
    print(model_results)

    print(f"Total training time: {end_time-start_time:.3f} seconds")
    print("DONE")