import torch
import torch.nn as nn
import torchvision.models as models

class Deeplabv3_GRU_CombineChannels(nn.Module):
    def __init__(self,num_classes,backbone="resnet50"):
        super(Deeplabv3_GRU_CombineChannels, self).__init__()

        if backbone.find("resnet50")>=0:
            self.dl = models.segmentation.deeplabv3_resnet50(pretrained=False, progress=True)
        elif backbone.find("resnet101")>=0:
            self.dl = models.segmentation.deeplabv3_resnet101(pretrained=False, progress=True)
        else:
            print("backbone for Deeplap not recognized ...\n")
            exit(-1)
        self.dl.classifier[4] = torch.nn.Conv2d(256, num_classes, 1)
        out_channels = self.dl.classifier[0].project[0].out_channels
        self.project_gru = nn.Sequential(
            Project_GRU(out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))
        self.dl.classifier[0].project=self.project_gru

    def forward(self, x):
        x = self.dl(x)['out']
        return x

class Project_GRU(nn.Module):
    def __init__(self,out_channels):
        super(Project_GRU, self).__init__()
        self.out_channels = out_channels
        self.gru = nn.GRU(input_size=out_channels, hidden_size = out_channels) #the input is a pixel in the combined features with C=2048 "Seqlenth"

    def forward(self, x):
        feature_maps= [] #feature map for each image (batch,C,H,W)
        for image in x:#for each image in the batch do
            image_shape = image.shape  # (C,H,W)
            image = image.view(5, self.out_channels, *image_shape[1:])  # (5,256, H, W)
            image = image.view(5, self.out_channels, -1)  # (5,256, HW)
            image = image.permute(0, 2, 1)  # (5,HW,256) (seq_len=C, batch=HW, input_size=256)

            ouput, h = self.gru(image) # h =(1, batch=HW, hidden_size=out_channels)

            # (1,HW,out_channels)=>(HW,out_channels)=>(out_channels,HW)=>(out_channels,H, W)
            h = h.squeeze().transpose(1,0).view(-1,*image_shape[1:])

            feature_maps.append(h)
        x = torch.stack(feature_maps)

        return x