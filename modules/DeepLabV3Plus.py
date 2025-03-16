import torch
import modules.xception as xception
import torch.nn as nn

# Basic element of the Xception model
class basic_cnn_layer(torch.nn.Module):
    def __init__(self, 
                 input_channel:int,
                 out_channels:int,
                 kernel_size:int,
                 stride:int=1,
                 padding:int=0,
                 dilation:int=1):
        """
        initiate basic cnn layer class
        conv -> batch normalization -> relu

        Parameters:
            input_channel (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the kernel.
            stride (int): Stride of the convolution.
            padding (int): Padding of the convolution.
            dilation (int): Dilation of the convolution.
        """
        super(basic_cnn_layer, self).__init__()
        
        self.conv = torch.nn.Conv2d(in_channels= input_channel,
                                    out_channels = out_channels, 
                                    kernel_size = kernel_size,
                                    stride = stride,
                                    padding = padding ,dilation=dilation)
        
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU()
    
    def forward(self, 
                x:torch.Tensor)->torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
class seperable_cnn_layer(torch.nn.Module):
    def __init__(self, 
                 input_channel:int,
                 out_channels:int,
                 kernel_size:int,
                 stride:int=1,
                 padding:int=0,
                 dilation:int=1):
        super(seperable_cnn_layer, self).__init__()
        """
        make a seperable xception layer

        Parameters:
            input_channel (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the kernel.
            stride (int): Stride of the convolution.
            padding (int): Padding of the convolution.
            dilation (int): Dilation of the convolution.
        """        
        self.depthwise_conv = torch.nn.Conv2d(in_channels= input_channel,
                                            out_channels = input_channel, 
                                            kernel_size = kernel_size,
                                            stride = stride,
                                            padding = padding ,dilation=dilation,
                                            groups = input_channel)
        
        self.pointwise_conv = torch.nn.Conv2d(in_channels= input_channel,
                                            out_channels = out_channels, 
                                            kernel_size = 1,
                                            stride = 1,
                                            padding = 0)
        
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU()
    
    def forward(self, 
                x:torch.Tensor)->torch.Tensor:
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

# ASPP module
class ASPP_pooling(torch.nn.Module):
    def __init__(self,
                 input_channel:int=2048,
                 out_channel:int=256):
        """
        make a Atrous Spatial Pyramid Pooling, 
        it is used to Combines multiple separate layers

        Parameters:
            input_channel (int): Number of input channels.
            out_channels (int): Number of output channels.
        """  
        super(ASPP_pooling, self).__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d((1,1))
        self.conv1 = torch.nn.Conv2d(in_channels = input_channel,
                                     out_channels = out_channel,
                                     kernel_size = 1,
                                     stride = 1,
                                     padding = 0)
    
    def forward(self, 
                x:torch.Tensor)->torch.Tensor:
        size = x.shape[-2:]
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = torch.nn.functional.interpolate(x, size, mode='bilinear', align_corners=True)
        return x

class ASPP(torch.nn.Module):
    def __init__(self,rate,input_channel=2048,out_channel=256):
        """
        make a Atrous Spatial Pyramid,
        it has 5 different rate of dilated convolution 

        Parameters:
            rate (list): List of dilation rates of each aspp convolution.
            input_channel (int): Number of input channels.
            out_channels (int): Number of output channels.
        """  
        super(ASPP, self).__init__()
        self.rate = rate

        self.aspp_conv1 = seperable_cnn_layer(input_channel, out_channel, kernel_size=1, 
                                              stride=1, padding=0, dilation=1)
        self.aspp_conv2 = seperable_cnn_layer(input_channel, out_channel, kernel_size=3, 
                                              stride=1, padding=self.rate[0], dilation=self.rate[0])  
        self.aspp_conv3 = seperable_cnn_layer(input_channel, out_channel, kernel_size=3, 
                                              stride=1, padding=self.rate[1], dilation=self.rate[1])
        self.aspp_conv4 = seperable_cnn_layer(input_channel, out_channel, kernel_size=3, 
                                              stride=1, padding=self.rate[2], dilation=self.rate[2])  
        self.aspp_conv5 = seperable_cnn_layer(input_channel, out_channel, kernel_size=3, 
                                              stride=1, padding=self.rate[3], dilation=self.rate[3])  
        self.img_interpolate = ASPP_pooling(input_channel, out_channel)  

        self.pointwise_conv = basic_cnn_layer(out_channel*6, out_channel,kernel_size=3, 
                                              stride=1, padding=1, dilation=1) 
        
    def forward(self, 
                x:torch.Tensor)->torch.Tensor:
        slice1 = self.aspp_conv1(x)
        slice2 = self.aspp_conv2(x)
        slice3 = self.aspp_conv3(x)
        slice4 = self.aspp_conv4(x)
        slice5 = self.aspp_conv5(x)
        img_inter = self.img_interpolate(x)

        x = torch.cat([slice1,slice2,slice3,slice4,slice5,img_inter],dim=1)

        x = self.pointwise_conv(x)

        return x
    
class Decoder(torch.nn.Module):
    def __init__(self, 
                 in_channels:int,
                 out_channel:int,
                 num_classes:int,
                 rate:list):
        """
        make a decoder layer

        Parameters:
            in_channels (int): Number of input channels.
            out_channel (int): Number of output channels.
            num_classes (int): Number of classes.
            rate (list): List of dilation rates of each aspp convolution.
        """  
        super(Decoder, self).__init__()
        self.project = basic_cnn_layer(in_channels, 48, 1, 1, 0)

        self.aspp = ASPP(rate,in_channels, out_channel)
        self.cls_conv = basic_cnn_layer(256, 256, 3, 1, 1)

        self.classifier = torch.nn.Sequential(
            basic_cnn_layer(304, 256, 3, 1, 1),
            torch.nn.Conv2d(256, num_classes, 1)
        )

    def forward(self, 
                x:torch.Tensor)->torch.Tensor:
        x_1 = self.project(x)
        x_2 = self.aspp(x)
        x_2 = self.cls_conv(x_2)
        x = torch.cat([x_1, x_2], dim=1)
        x = self.classifier(x)
        
        output_feature = torch.nn.functional.interpolate(x, (513,513), mode='bilinear', align_corners=True)
        return output_feature

class DeepLabV3Plus(torch.nn.Module):
    def __init__(self, 
                 num_classes:int,
                 rate:list):
        """
        make a DeepLabV3Plus senetic segmentation model

        Parameters:
            num_classes (int): Number of classes.
            rate (list): List of dilation rates of each aspp convolution.
        """ 
        super(DeepLabV3Plus, self).__init__()
        self.backbone = xception.xception(num_classes = 1000, pretrained = False)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        

        self.decoder = Decoder(2048, 256, num_classes, rate)
    
    def forward(self, 
                x:torch.Tensor)->torch.Tensor:
        x = self.backbone(x)
        x = self.decoder(x)
        return x
        
