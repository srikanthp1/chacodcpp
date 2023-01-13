# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class Resnet(nn.Module):
#     def __init__(self):
#         pass


import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models 
from torchvision.models.feature_extraction import create_feature_extractor 

def python_operator(layer1,layer2,layer3,layer4):
    """Pythonoperatorexampleforupsample-aggregate:paramlayer1:outputfromlayer1with size[1,128,104,104]:paramlayer2:outputfromlayer2withsize[1,128,52, 52]:paramlayer3:outputfromlayer3withsize[1,256,26, 26]:paramlayer4:outputfromlayer4withsize[1,512,13, 13]:return: upscaled output with size [1, 960, 104, 104]"""
    output=[layer1] 
    for l in [layer2, layer3,layer4]:
        output.append(F.interpolate(l,(104,104)))
        return torch.cat(output,dim=1)

class Aggus(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, layer1, layer2, layer3, layer4):
        output=[layer1] 
        for l in [layer2, layer3,layer4]:
            output.append(F.interpolate(l,(104,104)))
        return torch.cat(output, dim=1)
        
class ExtremeResNet(nn.Module):
    
    def __init__(self):
        super().__init__()
        #loadthecustomeroperatorusingtheDLLfile
        self.resnet18 = models.resnet18()
        layer_keys=["layer1","layer2","layer3","layer4"]
        # createthe featureextractor
        self.feature_extractor = create_feature_extractor(self.resnet18, layer_keys)
        self.downsample =nn.Conv2d(in_channels=960,out_channels=512,kernel_size=7,stride=8)
        self.aggus = Aggus()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, 10)
        if True:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self,x):
        feats=self.feature_extractor(x)

        output = self.aggus(*[v for k, v in feats.items()])# downsample
        output=self.downsample(output)#concatenatewithlayer4output
        output=torch.cat((feats["layer4"],output),dim=1)
        output = self.avgpool(output)
        output = torch.flatten(output, 1)
        output = self.fc(output)
        return output
        
if __name__=="__main__":
    eresnet = ExtremeResNet()# create random input
    image=torch.randint(255,(1,3,416,416),dtype=torch.float)
    model_out=eresnet(image)# [1, 1024, 13, 13] is the expected output
    print(model_out.shape)

    X = torch.randn(1, 3, 416, 416)
    # X1 = torch.randn(1, 128, 104, 104)
    # X2 = torch.randn(1, 256, 52, 52)
    # X3 = torch.randn(1, 512, 26, 26)
    inputs = (X)#, X1, X2, X3)

    f = './model.onnx'
    torch.onnx.export(ExtremeResNet(), inputs, f,
                       opset_version=11,
                    #    example_outputs=None,
                       input_names=["X"], output_names=["Y"])
                    #    custom_opsets={"mydomain": 2})