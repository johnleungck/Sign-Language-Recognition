import cv2
import time
from PIL import Image
import numpy as np
import torch
import torchvision
from torch import nn

class slimnet(nn.Module):
    def __init__(self):
        super(slimnet, self).__init__()
        
        self.quant = torch.ao.quantization.QuantStub()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1), #output_shape=(32,32,32)
            nn.BatchNorm2d(32),
            nn.ReLU(), # activation
            nn.Dropout(0.2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1), #output_shape=(32,32,32)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1), #output_shape=(32,32,32)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.MaxPool2d(2,2) #output_shape=(64,16,16)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=50, kernel_size=3, stride=1, padding=1), #output_shape=(50,16,16)
            nn.BatchNorm2d(50),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(in_channels=50, out_channels=64, kernel_size=3, stride=1, padding=1), #output_shape=(64,16,16)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1), #output_shape=(64,16,16)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(2,2) #output_shape=(64,8,8)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1), #output_shape=(64,8,8)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1), #output_shape=(64,8,8)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.2),    
            nn.MaxPool2d(2,2) #output_shape=(128,4,4)
        )
        
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1), #output_shape=(64,4,4)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1), #output_shape=(64,4,4)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.2)
        ) 

        self.layer6 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1), #output_shape=(64,4,4)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1), #output_shape=(64,4,4)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1), #output_shape=(64,4,4)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(2,2), #output_shape=(64,2,2)

            nn.Conv2d(in_channels=64, out_channels=100, kernel_size=1, stride=1, padding=0), #output_shape=(100,2,2)
            nn.BatchNorm2d(100),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(2,2), #output_shape=(100,1,1)
        )       

        self.flatten = nn.Flatten()

        self.layer7 = nn.Sequential(
            nn.Linear(100, 24),
        )

        self.dequant = torch.ao.quantization.DeQuantStub()
    
    def forward(self, x):
        # Convolution 1
        out = self.quant(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.flatten(out)
        out = self.layer7(out)
        out = self.dequant(out)
       
        return out
    
    def fuse_model(self, is_qat=False):
        fuse_modules = torch.ao.quantization.fuse_modules_qat if is_qat else torch.ao.quantization.fuse_modules
        for m in self.modules():
            if type(m) == nn.Sequential:
                fuse_modules(m, [['0','1','2']], inplace=True)


model = slimnet()
model.eval
model.qconfig = torch.ao.quantization.get_default_qat_qconfig('x86')
model_pre = torch.ao.quantization.prepare_qat(model.train())
model = torch.ao.quantization.convert(model_pre.eval(), inplace=False)
print(model)

model.load_state_dict(torch.load('./slimnetqatparam.pth'))

#model = torch.load('./slimnetqat.pth')

cap = cv2.VideoCapture(0)

y = 0
while (y >= 0):
    ret, img= cap.read()
    img=img[:,80:560]
    preimg = torch.from_numpy(img).permute(2,0,1)
    grey = torchvision.transforms.Grayscale()
    preimg = grey(preimg)
    trans = torchvision.transforms.Resize((32,32))
    preimg = trans(preimg)
    preimg = preimg.to(torch.float32).unsqueeze(0)
    #preimg = preimg.expand(-1,3,-1,-1)
    #print(preimg.shape)
    results = model(preimg).data
    print(results)
    _, pred = torch.max(results ,1)
    print(pred)
    cv2.putText(img, str(pred), (10,50), cv2.FONT_HERSHEY_COMPLEX,1, (0,255,0),2)

    p1 = torch.Tensor.numpy(preimg[0][0])
    p1 = np.uint8(p1)
    img2 = Image.fromarray(p1).convert('RGB')
    img2 = np.array(img2)[:,:,::-1].copy()

    cv2.imshow('Webcam', img)
    cv2.imshow('gray28', img2)

    if cv2.waitKey(1) == ord('q'):
        break

    time.sleep(0.001)
    y+=1

cap.release()
cv2.destroyAllWindows()