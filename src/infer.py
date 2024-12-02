

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import cv2
import glob

import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from torch.utils.data import Dataset
from tqdm import tqdm_notebook as tqdm
from matplotlib import pyplot as plt
import tifffile
import random
from collections import defaultdict
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tqdm import tqdm as ttm
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import warnings
from random import randint
warnings.filterwarnings("ignore")
import numpy as np
from skimage import io

import pandas as pd

import collections
from pprint import pprint
import numpy as np
import pandas as pd
from skimage import measure
from skimage.measure import label,regionprops
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from ipywidgets import interact
#from scipy.ndimage import rotate,zoomb
from skimage.registration import phase_cross_correlation
from skimage.registration._phase_cross_correlation import _upsampled_dft
from scipy.ndimage import fourier_shift
from tqdm  import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage import data, util
from skimage.measure import label,regionprops
from recoloss import CrossEntropyLabelSmooth,TripletLoss

import random
import statistics
from sklearn.model_selection import KFold
from unetext3Dn_con7s import UNet3D

import argparse

# 创建解析器
parser = argparse.ArgumentParser(description="Training script for the model")

# 添加参数
parser.add_argument('--data_dir', type=str, required=True, help="Path to the training data directory")
parser.add_argument('--out_dir', type=str, required=True, help="Path to the output data directory")
parser.add_argument('--pretrained_weights1', type=str, help="Path to the pretrained model weights file")
parser.add_argument('--pretrained_weights2', type=str, help="Path to the pretrained model weights file")
parser.add_argument('--pretrained_weights3', type=str, help="Path to the pretrained model weights file")

parser.add_argument('--overlapxy', type=int, default=16, help="overlap pixels xy panel")
parser.add_argument('--overlapz', type=int, default=4, help="overlap pixels z panel")
parser.add_argument('--resolution_z', type=int, default=10, help="ratio of resolution z/xy")
parser.add_argument('--patch_size_xy', type=int, default=256, help="patch size xy")
parser.add_argument('--patch_size_z', type=int, default=31, help="patch size z")
parser.add_argument('--low', type=int, default=40, help="test real cells in first frame")



# 解析参数
args = parser.parse_args()

if not os.path.exists('./ctc2/'):
    os.mkdir('./ctc2/')
if not os.path.exists('./ctc2x/'):
    os.mkdir('./ctc2x/')
def fill(n,x,y,z,v=1,s=8):
    rr,cc=draw.ellipse(int(x),int(y), s, s)
    ir=(rr>0)*(rr<n.shape[0])
    ic=(cc>0)*(cc<n.shape[1])
    ii=ir*ic
    rr=rr[ii]
    cc=cc[ii]
    z1=int(max(0,z-max(1,s//10+1)-1))
    z2=int(min(100,z+max(1,s//10+1)+2))

    if z1==z2:
        n[rr,cc,z1]=v
    else:
        n[rr,cc,z1:z2]=v
    return n
def bc(img,th=0):
    x=img!=th
    x=x.sum(axis=0)
    x=(x>0)
    x1=list(x.squeeze()).index(1)
    x2=list(x[::-1].squeeze()).index(1)
    y=img!=th
    y=y.sum(axis=1)
    y=(y>0)
    y1=list(y).index(1)
    y2=list(y[::-1]).index(1)
    return img[y1:img.shape[0]-y2,x1:img.shape[1]-x2]
def ar(v):
    av=v.clone()
    av=torch.zeros([26,v.shape[0],v.shape[1],v.shape[2]])
    zv=torch.zeros([v.shape[0]+2,v.shape[1]+2,v.shape[2]+2])
    zv[1:-1,1:-1,1:-1]=v
    n=0
    
    for x in range(3):
        for y in  range(3):
            for z in range(3):
                if not x==1 and y==1 and z==1:
                    av[n]=zv[x:x+v.shape[0],y:y+v.shape[1],z:z+v.shape[2]]
                    n=n+1
    return av
def tim(a):
    l=[]
    a.seek(0)
    l.append(np.array(a)[:,:,np.newaxis])
    n=0
    while(1):
       n+=1
       try:
           a.seek(n)
       except:break
       l.append(np.array(a)[:,:,np.newaxis])
    b=np.concatenate(l,2)
    return b

        


l=os.listdir(args.data_dir+'/')
l=[i for i in l if 'tif' in i]

vD={}
for i in tqdm(l):
    if 'tif' in i:         
        
        num=int(i.split('.')[0][1:])
        vD[num]=args.data_dir+'/'+i

kh=len(vD)-1

class VDataset(Dataset):

    def __init__(self, data,le, transform=None):
        
        
        #self.path = path
        self.data =data
        self.transform = transform
       
    def __len__(self):
        
        return kh

    def __getitem__(self, i):
        
   
        
        j=i
        if 1:
            j=j
            g=vD[j]
    
            o=1
            g2=vD[j+o]
      
            
            if j+o+1==kh+1:
                g3=vD[kh]
            else:
                g3=vD[j+o+1]        
          

        #j=random.choice(x_train)
      
        #print(g)
        
        img=tim(Image.open(g))#[i1:i2,i3:i4,i5:]
        img2=tim(Image.open(g2))#[i1:i2,i3:i4,i5:]
        img3=tim(Image.open(g3))#[i1:i2,i3:i4,i5:]
      
 
  
    
        b = torch.from_numpy((img.astype(float)))
        
  
       
        img3 = torch.from_numpy(img3.astype(float))
       

        c=torch.from_numpy(img2.astype(float))     
        #size1=torch.from_numpy(t31)
        #size2=torch.from_numpy(t32)
        #print(b.shape,timg.shape,k)
        return {'image': b,'im2':c,'img3':img3} 




class EXNet(nn.Module):
    """
    Implementations based on the Unet3D paper: https://arxiv.org/pdf/1706.00120.pdf
    """

    def __init__(self, in_channels, n_classes, base_n_filter=8):
        super(EXNet, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.base_n_filter = base_n_filter

        self.lrelu = nn.LeakyReLU()
        

        self.l1=nn.Linear(self.in_channels*5, self.in_channels)
        self.l2=nn.Linear(self.in_channels*5, self.in_channels)
        self.l3=nn.Linear(self.in_channels*5, self.in_channels)
        self.l4=nn.Linear(3*5, self.in_channels)
        self.l5=nn.Linear(6*5, self.in_channels)
        self.l6=nn.Linear(6*5, self.in_channels)
        self.bnx=nn.BatchNorm1d(self.in_channels)
        self.bny=nn.BatchNorm1d(self.in_channels)
        self.bn1=nn.BatchNorm1d(self.in_channels)
        self.bn2=nn.BatchNorm1d(self.in_channels)
        self.bn3=nn.BatchNorm1d(self.in_channels)
        self.bn4=nn.BatchNorm1d(self.in_channels)
        self.bn5=nn.BatchNorm1d(self.in_channels)
        self.bn6=nn.BatchNorm1d(self.in_channels)
        self.bn7=nn.BatchNorm1d(self.in_channels*2)
        self.bnc=nn.BatchNorm2d(8)
        self.fc=nn.Linear(self.in_channels*6, self.in_channels*2)
        self.fc2=nn.Linear(self.in_channels*2, self.in_channels)
        self.out=nn.Linear(self.in_channels, self.n_classes)
        self.c1 = nn.Conv2d(1, 8, kernel_size=(2,5), stride=1, padding=(0,2),
                                     bias=False)
        self.c2 = nn.Conv2d(1, 8, kernel_size=(2,3), stride=1, padding=(0,1),
                                     bias=False)
        self.cx=nn.Linear(self.in_channels*8*2, self.in_channels)
        '''self.conv3d_c1_1 = nn.Conv3d(self.in_channels, self.base_n_filter, kernel_size=3, stride=1, padding=1,
                                     bias=False)'''
        self.outc=nn.Linear(self.in_channels+self.n_classes, 2)
        self.d1= nn.Dropout(p=0.1)
        self.d2= nn.Dropout(p=0.1)
        self.s=nn.Sigmoid()
    def forward(self, x,y,p,s,zs):
        #  Level 1 context pathway
      
        x01=torch.cat([x,y[:,0].unsqueeze(1)],1)

        x01=self.cx(torch.cat([self.lrelu(self.bnc(self.c1(x01.unsqueeze(1)))),self.lrelu(self.bnc(self.c2(x01.unsqueeze(1))))],1).reshape(x01.shape[0],-1))
  
        x02=torch.cat([x,y[:,1].unsqueeze(1)],1)

        x02=self.cx(torch.cat([self.lrelu(self.bnc(self.c1(x02.unsqueeze(1)))),self.lrelu(self.bnc(self.c2(x02.unsqueeze(1))))],1).reshape(x02.shape[0],-1))
        x03=torch.cat([x,y[:,2].unsqueeze(1)],1)

        x03=self.cx(torch.cat([self.lrelu(self.bnc(self.c1(x03.unsqueeze(1)))),self.lrelu(self.bnc(self.c2(x03.unsqueeze(1))))],1).reshape(x03.shape[0],-1))
        x04=torch.cat([x,y[:,3].unsqueeze(1)],1)

        x04=self.cx(torch.cat([self.lrelu(self.bnc(self.c1(x04.unsqueeze(1)))),self.lrelu(self.bnc(self.c2(x04.unsqueeze(1))))],1).reshape(x04.shape[0],-1))
        x05=torch.cat([x,y[:,4].unsqueeze(1)],1)

        x05=self.cx(torch.cat([self.lrelu(self.bnc(self.c1(x05.unsqueeze(1)))),self.lrelu(self.bnc(self.c2(x05.unsqueeze(1))))],1).reshape(x05.shape[0],-1))
   
        x1=torch.cat([x01,x02,x03,x04,x05],1).reshape(x.shape[0],-1)
        x2=((y-x)).reshape(x.shape[0],-1)
        #x3=(self.bnx(x)*self.bny(y)).reshape(x.shape[0],-1)
        xx=torch.sqrt((x*x).sum(2))
        yy=torch.sqrt((y*y).sum(2))
        xy=x*y
        xy2=(xx*yy).unsqueeze(-1)#.repeat(1,1,xy.shape[-1])
        #print(xy.shape,xy2.shape)
        x3=xy/xy2
        x3=x3.reshape(x.shape[0],-1)
        
        x4=p.reshape(x.shape[0],-1)
        x5=s.reshape(x.shape[0],-1)
        x6=zs.reshape(x.shape[0],-1)
        x1=self.lrelu(self.bn1(self.l1(x1)))
        x2=self.lrelu(self.bn2((self.l2(x2))))
        x3=self.lrelu(self.bn3(self.l3(x3)))
        x4=self.lrelu(self.bn4((self.l4(x4))))
        x5=self.lrelu(self.bn5((self.l5(x5))))
        x6=self.lrelu(self.bn6((self.l6(x6))))
        nx=torch.cat([x1,x2,x3,x4,x5,x6],1)
        #x1=self.lrelu(self.l1(x))
        ui=self.bn7(self.lrelu(self.fc(((nx))) ))
        nx=self.lrelu(self.fc2(ui))
        f=(self.out(nx))   
        #fk=self.outc(torch.cat([nx,f],1))   
        return f#,fk



SMOOTH = 1e-6

batch_size =1
os.environ['CUDA_VISIBLE_DEVICES']='0'
device = torch.device("cuda:0")

from skimage import draw

tm=2
tm1=1
DK=np.zeros([tm*2+1,tm*2+1,3])
rr,cc=draw.ellipse(tm,tm, tm,tm)
DK[rr,cc,:]=1
DK=torch.from_numpy(DK).float().to(device).reshape([1,1,tm*2+1,tm*2+1,3])
DK1=DK[:,:,:,:,:2*tm1+1]
DK1[DK1<1]=1
#G =UNet(1,1)
U =UNet3D(2,6)
EX=EXNet(64,8)
EN=EXNet(64,6)

#U2 =UNet(1,2)
#model.fc = torch.nn.Linear(2048, 6)

#G.to(device)
U.to(device)
EX.to(device)
EN.to(device)
# U.load_state_dict(torch.load('../model/U-ext+cer2-149.0-0.7062.pth'))
# EX.load_state_dict(torch.load('../model/EX+cer2-149.0-0.7062.pth'))
# EN.load_state_dict(torch.load('../model/EN+cer2-149.0-0.7062.pth'))

# U.load_state_dict(torch.load('../xmodel/U-ext+sc2n-an-89.0-1.3038.pth'))
# EX.load_state_dict(torch.load('../xmodel/EX+sc2n-an-89.0-1.3038.pth'))
# EN.load_state_dict(torch.load('../xmodel/EN+sc2n-an-89.0-1.3038.pth'))
U.load_state_dict(torch.load(args.pretrained_weights1))
EX.load_state_dict(torch.load(args.pretrained_weights2))
EN.load_state_dict(torch.load(args.pretrained_weights3))

# U.load_state_dict(torch.load('../xmodel/U-ext+sc2n-345.0-13.0442.pth'))
# EX.load_state_dict(torch.load('../xmodel/EX+sc2n-345.0-13.0442.pth'))
# EN.load_state_dict(torch.load('../xmodel/EN+sc2n-345.0-13.0442.pth'))

# U.load_state_dict(torch.load('X:/xmodel/U-ext+sc2n27-171.0-2.0427.pth'))
# EX.load_state_dict(torch.load('X:/xmodel/EX+sc2n27-171.0-2.0427.pth'))
# EN.load_state_dict(torch.load('X:/xmodel/EN+sc2n27-171.0-2.0427.pth'))
# U.load_state_dict(torch.load('X:/xmodel/U-ext+sc2n27-269.0-14.6831.pth'))
# EX.load_state_dict(torch.load('X:/xmodel/EX+sc2n27-269.0-14.6831.pth'))
# EN.load_state_dict(torch.load('X:/xmodel/EN+sc2n27-269.0-14.6831.pth'))
#EXP=torch.load('D:/track-data/model//EXP+-25.0-1.9528.pth')


print('    Total params: %.2fM' % (sum(p.numel() for p in U.parameters())/1000000.0))


val_dataset = VDataset(
 vD, le=1000)
data_loader_val = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0) 
val_loss=1000
va=10000
win=3
lg=32
ko=3
pl=16
pk=(1,1,1)
def kfbx(x):
    
    kn=torch.zeros([x.shape[0],x.shape[1]+4,x.shape[2]+4,x.shape[3]+4])
    kq=torch.zeros([x.shape[0],45,x.shape[1],x.shape[2],x.shape[3]])
    kn[:,2:-2,2:-2,2:-2]+=x.cpu()
    num=0
    for xc in [-2,-1,0,1,2]:
        for y in [-2,-1,0,1,2]:
            for z in [-2,-1,0,1,2]:
               if np.abs(xc)+max(np.abs(y),np.abs(z))<=2:
                    kq[:,num]=kn[:,xc+2:xc+2+x.shape[1],y+2:y+2+x.shape[2],z+2:z+2+x.shape[3]]
                    num+=1
                    #print(num)
    return kq
def kfbu(x):   
    kn=torch.zeros([x.shape[0],x.shape[1]+4,x.shape[2]+4,x.shape[3]+4])
    kq=torch.zeros([x.shape[0],25,x.shape[1],x.shape[2],x.shape[3]])
    kn[:,2:-2,2:-2,2:-2]+=x.cpu()
    num=0
    for xc in [-2,-1,0,1,2]:
        for y in [-2,-1,0,1,2]:
            for z in [-2,-1,0,1,2]:
               if np.abs(xc)+np.abs(y)+np.abs(z)<=2:
                    kq[:,num]=kn[:,xc+2:xc+2+x.shape[1],y+2:y+2+x.shape[2],z+2:z+2+x.shape[3]]
                    num+=1
                    
    return kq
def kfl(x):
    
    kn=torch.zeros([x.shape[0],x.shape[1]+2,x.shape[2]+2,x.shape[3]+2])
    kq=torch.zeros([x.shape[0],27,x.shape[1],x.shape[2],x.shape[3]])
    kn[:,1:-1,1:-1,1:-1]+=x.cpu()
    num=0
    for xc in [-1,0,1]:
        for y in [-1,0,1]:
            for z in [-1,0,1]:
                if np.abs(xc)+np.abs(y)+np.abs(z)<=1:
                    kq[:,num]=kn[:,xc+1:xc+1+x.shape[1],y+1:y+1+x.shape[2],z+1:z+1+x.shape[3]]
                    num+=1
    return kq
def kflb(x):
    
    kn=torch.zeros([x.shape[0],x.shape[1]+2,x.shape[2]+2,x.shape[3]+2])
    kq=torch.zeros([x.shape[0],27,x.shape[1],x.shape[2],x.shape[3]])
    kn[:,1:-1,1:-1,1:-1]+=x.cpu()
    num=0
    for xc in [-1,0,1]:
        for y in [-1,0,1]:
            for z in [-1,0,1]:
                if np.abs(xc)+max(np.abs(y),np.abs(z))<=1:
                    kq[:,num]=kn[:,xc+1:xc+1+x.shape[1],y+1:y+1+x.shape[2],z+1:z+1+x.shape[3]]
                    num+=1
    return kq
def kfs2(x):
    
    kn=torch.zeros([x.shape[0],x.shape[1]+4,x.shape[2]+4,x.shape[3]])
    kq=torch.zeros([x.shape[0],25,x.shape[1],x.shape[2],x.shape[3]])
    kn[:,2:-2,2:-2,:]+=x.cpu()
    num=0
    for xc in [-2,-1,0,1,2]:
        for y in [-2,-1,0,1,2]:
               
                    kq[:,num]=kn[:,xc+2:xc+2+x.shape[1],y+2:y+2+x.shape[2],:]
                    num+=1
    return kq
def kfs(x):
    
    kn=torch.zeros([x.shape[0],x.shape[1]+2,x.shape[2]+2,x.shape[3]])
    kq=torch.zeros([x.shape[0],9,x.shape[1],x.shape[2],x.shape[3]])
    kn[:,1:-1,1:-1,:]+=x.cpu()
    num=0
    for xc in [-1,0,1]:
        for y in [-1,0,1]:
        
                    kq[:,num]=kn[:,xc+1:xc+1+x.shape[1],y+1:y+1+x.shape[2],:]
                    num+=1
    return kq


tk0 = tqdm(data_loader_val, desc="Iteration")
   
U.eval()
EN.eval()
EX.eval()

DK=np.zeros([tm*2+1,tm*2+1,9])
rr,cc=draw.ellipse(tm,tm, tm,tm)
DK[rr,cc,:]=1
tm=2
tm1=1
DK=torch.from_numpy(DK).float().to(device).reshape([1,1,tm*2+1,tm*2+1,9])
DK1=np.array([[0,0,1,1,1,0,0],[0,1,1,2,1,1,0],[1,1,2,3,2,1,1],[1,2,3,4,3,2,1],[1,1,2,3,2,1,1],[0,1,1,2,1,1,0],[0,0,1,1,1,0,0]])
DK1=torch.from_numpy(DK1).float().to(device).reshape([1,1,7,7,1])#.repeat(1,1,1,1,3)
def kfb(x):
    
    kn=torch.zeros([x.shape[0],x.shape[1]+4,x.shape[2]+4,x.shape[3]+4])
    kq=torch.zeros([x.shape[0],125,x.shape[1],x.shape[2],x.shape[3]])
    kn[:,2:-2,2:-2,2:-2]+=x.cpu()
    num=0
    for xc in [-2,-1,0,1,2]:
        for y in [-2,-1,0,1,2]:
            for z in [-2,-1,0,1,2]:
               
                    kq[:,num]=kn[:,xc+2:xc+2+x.shape[1],y+2:y+2+x.shape[2],z+2:z+2+x.shape[3]]
                    num+=1
    return kq
def ud(x):
    x1=torch.zeros_like(x)
    x2=torch.zeros_like(x)
    x1[:,:,:,:,:-1]=x[:,:,:,:,1:]
    x2[:,:,:,:,1:]=x[:,:,:,:,:-1]
    x0=x==0
    xn=(x0*torch.cat([x1,x2],1).max(1)[0]-1)
    
    xn[xn<0]=0
    return x+xn
def vsup256(U,x,la):
    with torch.no_grad():
        Uout,uo,foz,zs1,size1 =  U(x)
        Uout=Uout[:,:2]
        #Uout1,uo1,_,_,_ =  U(torch.cat([x[:,1:2],x[:,:1]],1))
    # nf4=foz[:,:,:,::]
    # nf4=nf4*nf4
    # output1 = torch.sqrt(fixed_conv_d1(nf4))
    # output5 = torch.sqrt(fixed_conv_d5(nf4))
    # output=(output5-output1)[0]
    Uf=foz.transpose(0,1)
    #print(uo.shape)
    ux=(uo[:,4:].max(1)[0]).float().unsqueeze(1)#F.conv3d((uo[:,:].max(1)[1]>3).float().unsqueeze(1), DK1, padding=(3,3,0))
    
    #ux=F.conv3d((ux).float(), DDK7, padding=(3,3,0))#[:,:,:,:,0]
    #print(ux.shape,'1')
    
    uk=(Uout.argmax(1)==1)#.squeeze().cpu().numpy()
    #kc=(torch.abs(output)<0.03)
    #kc=F.conv3d((kc.unsqueeze(1)>0).float(), DK1, padding=(3,3,0))
    #uk=uk*(kc==0)
    #ux=ux*uk
    uc=((kflb(ux.squeeze(1)).cuda().max(1)[0]==ux))*(uk)#*(size1[:,0]>1)
    # ar=F.conv3d((uc>0).float(), DK1, padding=(3,3,0))
    # ar=ud(ar)
 
    # uc+=((kflb(ux.squeeze(1)).cuda().max(1)[0]==ux))*(uk)#*(ar==0)
    uk=uk.squeeze().cpu().numpy()
    uc=uc.squeeze(1)
    uu=label(uk)
    # uf=np.unique(uu[uc.cpu()[0]>0])
    # uf=uf[uf>0]
    # for i in regionprops(uu):
    #     if i.label not in uf:
    #         x1,y1,z1,x2,y2,z2=i.bbox
    #         if i.area>50 :# and x1>0 and y1>0 and z1>0 and x2<uu.shape[0]-1 and y2<uu.shape[1]-1 and z2<30:
    #             x,y,z=i.centroid
    #             if uu[int(x),int(y),int(z)]==i.label:
    #                 uc[:,int(x),int(y),int(z)]=1
    #print(uc.shape,uc.sum())    #uc=((kflb(uo[:,:].max(1)[0]).cuda().max(1)[0]==uo[:,4]))*(kfs(Uout.argmax(1)==0).cuda().sum(1)==0)
    uc[:,:2]=0
    uc[:,-2:]=0
    uc[:,:,:2]=0
    uc[:,:,-2:]=0
    uc[:,:,:,:2]=0
    uc[:,:,:,-2:]=0
    ar=F.conv3d((uc.unsqueeze(1)>0).float(), DK1, padding=(3,3,0))
    
    #ar=F.conv3d((ar>0).float(), DK1, padding=(3,3,0))
    #ar=F.conv3d((ar>0).float(), DK1, padding=(3,3,0))
    for i in range(1):
            ar=ud(ar)
             
    u,x,y,z=torch.where(uc.to(device, dtype=torch.float)>0)

    #print(u.shape,lx,lx1)
    f1=Uf[:,u,x,y,z].transpose(0,1)
    s1=size1[u,0,x,y,z]
    zs1=F.sigmoid(zs1[u,:,x,y,z])
    #s11=size1[u,0,x,y,z+1]
    #s12=size1[u,0,x,y,z-1]
    #s1=torch.cat([s1.unsqueeze(1),s11.unsqueeze(1),s12.unsqueeze(1)],1).max(1)[0]
    
    lb1=la[u.cpu(),x.cpu(),y.cpu(),z.cpu()]
   
    p1=torch.cat([u.unsqueeze(1),x.unsqueeze(1),y.unsqueeze(1),z.unsqueeze(1)],1)
    #p2=torch.cat([u2.unsqueeze(1),x2.unsqueeze(1),y2.unsqueeze(1),z2.unsqueeze(1)],1)
  
    #Uout=torch.cat([Uout[:,:,:,:240],Uoutf[:,:,:,240-(320-256):]],3)
    return p1,f1,s1,zs1,lb1,ar,Uout
def cp2(x,ba,bb,bc):
    aa=((x[:,1])<=ba.max())*((x[:,1])>=ba.min())
    ab=((x[:,2])<=bb.max())*((x[:,2])>=bb.min())
    ac=((x[:,3])<=bc.max())*((x[:,3])>=bc.min())
    return aa*ab*ac
def cp3(x,ba):
    u,x,y,z=x[:,0],x[:,1],x[:,2],x[:,3]
    na=ba[x,y,z]>0
    return na
def cp2(x,ba):
    u,x,y,z=x[:,0],x[:,1],x[:,2],x[:,3]
    na=ba[x,y]>0
    return na
def bvsup256(U,x,l,xz1=0,xz2=2000000,yz1=0,yz2=200000):

    u,c,pl1,pl2,pl3=x.shape
    pol=args.patch_size_xy
    polz=args.patch_size_z
    pl=pol-args.overlapxy#256-128
    plz=polz-args.overlapz
    xz2=min(xz2,pl1)
    yz2=min(yz2,pl2)
    xm,ym,zm=(xz2-xz1)//pl+int((xz2-xz1)%pl>0),(yz2-yz1)//pl+int((yz2-yz1)%pl>0),(pl3)//plz+int(pl3%plz>0)
    p,f,s,zs,lb=[],[],[],[],[]
    plist=[]
    num=0
    uout=torch.zeros([pl1,pl2,pl3])
    ku=torch.zeros([pl1,pl2,pl3])
    kup=torch.zeros([pl1,pl2,pl3])
    #print(x[:,:,:,:256,:].shape,x[:,:,:,:256,:].max(),x[:,:,:,:256,:].min(),x[:,:,:,-256:,:].shape)
    for x1 in range(xm):
        for y1 in range(ym):
            for z1 in range(zm):
                num+=1
                ku[ku>1]=1
                
                v1,v2,v3,v4,v5,v6=x1*pl+xz1,x1*pl+pol+xz1,y1*pl+yz1,y1*pl+yz1+pol,z1*plz,z1*plz+polz
                if v2>=xz2:
                    v2=xz2
                    v1=v2-pol
                if v4>=yz2:
                    v4=yz2
                    v3=v4-pol
                if v6>=pl3:
                    v6=pl3
                    v5=v6-polz  
                    if v5<0:v5=0                             
                #print(v1,v2,v3,v4,v5,v6)
                with torch.no_grad():
                    p1,f1,s1,zs1,lb1,ar,uo=  vsup256(U,x[:,:,v1:v2,v3:v4,v5:v6],l[:,v1:v2,v3:v4,v5:v6])  
                uout[v1+2:v2-2,v3+2:v4-2,v5:v6]+=(uo.argmax(1).squeeze().cpu()[2:-2,2:-2]==1).float()
                    
                ar[ar>1]=1
                ar=ar.squeeze().cpu()
                #kup[v1:v2,v3:v4,v5:v6]+=ar
                #ku[v1:v2,v3:v4,v5:v6]+=ar
                #kux=(ku>1).float()*kup
                p1[:,1]+=v1
                p1[:,2]+=v3
                p1[:,3]+=v5
                if num>1:
                    
              
                  
                   
                    kn=cp3(p1.cpu(),ku==1)
                    
           
            
                    fn=~kn
                    p1=p1[fn]
                    f1=f1[fn]
                    s1=s1[fn]
                    zs1=zs1[fn]
                    lb1=lb1[fn]
                    
             
                    p=torch.cat([p,p1],0)
                    f=torch.cat([f,f1],0)
                    s=torch.cat([s,s1],0)
                    zs=torch.cat([zs,zs1],0)
                    lb=torch.cat([lb,lb1],0)
                else:
                    p=p1
                    f=f1
                    s=s1
                    zs=zs1
                    lb=lb1    
                ku[v1:v2,v3:v4,v5:v6]+=ar
    #print(p.shape[0],torch.unique(l).shape[0],torch.unique(lb).shape[0])
    #Uout=torch.cat([Uout[:,:,:,:240],Uoutf[:,:,:,240-(320-256):]],3)
    return p,f,zs,s,lb,uout


img3=tim(Image.open(vD[np.sort(list(vD.keys()))[-1]])).astype(float)
img4=tim(Image.open(vD[np.sort(list(vD.keys()))[-2]])).astype(float)
if img3.max()>255:
    def pa(x):
        return torch.log1p(x)
else:
        def pa(x):
            return x
def PA(x):
    #x=x*5
    #x[x<50]=0
    pa(x)
    return x
labels=torch.zeros(img3.shape).unsqueeze(0)
with torch.no_grad():
    p1,f1,zs1,s1,lb1,uo =  bvsup256(U,(PA(torch.cat([torch.from_numpy(img3).to(device, dtype=torch.float).unsqueeze(0).unsqueeze(0),torch.from_numpy(img4).to(device, dtype=torch.float).unsqueeze(0).unsqueeze(0)],1))),labels)
def box(x):
   
    x1=(x>1)
    l1=list(x1.sum((0,1))>1)
    z1=l1.index(1)
    z2=len(l1)-l1[::-1].index(1)
    l1=list(x1.sum((1,2))>1)
    xx1=l1.index(1)
    xx2=len(l1)-l1[::-1].index(1)
    l1=list(x1.sum((0,2))>1)
    y1=l1.index(1)
    y2=len(l1)-l1[::-1].index(1)

    return xx1,xx2,y1,y2,z1,z2#, mi+(ma-mi)/10
x1,x2,y1,y2,z1,z2=box(uo[:,:,1:-1])
aq=img3[uo>0].min()
BA=uo
BO=torch.zeros_like(BA)
BO[x1:x2,y1:y2,z1:z2]=1
w=torch.ones(1,1, 39,39,11)
nb=torch.nn.functional.conv3d(BA.unsqueeze(0), w, stride=1, padding=(19,19,5)).squeeze()
T=[]

def PA0(x):
    # x[x<15]=0
    # x[x<30]=(x[x<30]-15)*2
    # x[x<0]=0
    # x=x-x.min()
    pa(x)
    return x
def PA(x):
    #a=(BA*x.cpu().squeeze()).mean()*2+10
    c=((1-BO)*x.cpu().squeeze())
    a=(c[c>0]).mean()#*2+10
    # print(a)
    # a=30
    x[x<a]=0
    # x[x<15]=0
    # x[x<30]=(x[x<30]-15)*2
    # x[x<0]=0
    # x=x-x.min()
    pa(x)
    return x
def fill2(n,x,y,v=1,s=8):
    rr,cc=draw.ellipse(int(x),int(y), s, s)
    ir=(rr>0)*(rr<n.shape[0])
    ic=(cc>0)*(cc<n.shape[1])
    ii=ir*ic
    rr=rr[ii]
    cc=cc[ii]
   
    n[rr,cc]=v
    
    return n
def erosion(input_image, kernel_size=3):

    eroded_image = -F.max_pool3d(-input_image, kernel_size, stride=1, padding=(kernel_size // 2,kernel_size // 2,kernel_size // 2))
    
    return eroded_image
def cut(x):
    x0=x.max(-1)
    x1=x0.sum(1)
    x2=x0.sum(0)
    x3=x.sum(axis=(0,1))
    l=[]
    #print(x1.shape,x2.shape,x3.shape)
    if x1.shape[0]>4:
        for i in range(2,x1.shape[0]-2):
            if x1[i]+1<x1[i+1] and x1[i]+1<x1[i-1] and x1[i+1]<x1[i+2] and x1[i-1]<x1[i-2]:
                x[i]=0
            # if  x1[i]+1<=x1[i+1:i+3].mean() and 1+x1[i]<=x1[i-3:i].mean():
            #              x[i]=0
    if x2.shape[0]>4:
        for i in range(2,x2.shape[0]-2):
            if x2[i]+1<x2[i+1] and x2[i]+1<x2[i-1] and x2[i+1]<x2[i+2] and x2[i-1]<x2[i-2]:
                x[:,i]=0
            # if x2[i]+1<=x2[i+1:i+3].mean() and 1+x2[i]<=x2[i-3:i].mean():
            #     x[:,i]=0
    if x3.shape[0]>2:
        for i in range(1,x3.shape[0]-1):
            if x3[i]<x3[i+1] and x3[i]<x3[i-1]:
                x[:,:,i]=0
            # if x3[i]<x3[i+1:i+3].mean() and x3[i]<x3[i-3:i].mean():
            #     x[:,:,i]=0
    return x  
ZS=args.resolution_z
def ccut(x,p1,p2):
    u,v,w=p1
    uu,vv,ww=p2
    w=w//ZS
    ww=ww//ZS
    x=x[min(uu,u):max(uu,u),min(vv,v):max(vv,v),max(0,min(ww,w))-1:min(x.shape[2],max(ww,w))]
    x0=x.max(-1)
    x4=x[:,:,x.shape[2]//2]
    x14=x4.sum(1)
    x24=x4.sum(0)    
    x1=x0.sum(1)
    x2=x0.sum(0)
    x3=x.sum(axis=(0,1))
    l=0
    #print(x1.shape,x2.shape,x3.shape)
    #if x1.shape[0]>3:
    for i in range(1,x1.shape[0]-2):
        if x1[i]<x1[0] and x1[i]<x1[-1]:
            l+=1
        if x14[i]<x14[0] and x14[i]<x14[-1]:
            l+=1
        # if x1[i]<x1[i+1] and x1[i]>x1[i+2]:
        #     l+=1
        # if  x1[i]+1<=x1[i+1:i+3].mean() and 1+x1[i]<=x1[i-3:i].mean():
        #              l+=1
    for i in range(1,x2.shape[0]-1):
        if x2[i]<x2[0] and x2[i]<x2[-1]:
            l+=1
        if x24[i]<x24[0] and x24[i]<x24[-1]:
                l+=1
        # if x2[i]<x2[i+1] and x2[i]>x2[i+2]:
        #         l+=1
        # if x2[i]+1<=x2[i+1:i+3].mean() and 1+x2[i]<=x2[i-3:i].mean():
        #     l+=1
    if x3.shape[0]>2:
        for i in range(1,x3.shape[0]-1):
            if x3[i]<x3[i+1] and x3[i]<x3[i-1]:
                l+=1
            if x3[i]<x3[0] and x3[i]<x3[-1]:
                l+=1
            # if x3[i]<x3[i+1:i+3].mean() and x3[i]<x3[i-3:i].mean():
            #     x[:,:,i]=0
    return l   
edge=[]
FP=[]
FN=[]
SIS=[]
FPD=[]
FND=[]
val_loss=0
kk=0
se=0
iou=0
n=0
DP={}
TP={}
De=0
DD={}
l=[]
l2=[]
ll=[]
k=[]
ni=0
cuo={}
T=[]
nid=1
KD={}
import time
for step, batch in enumerate(tk0):
    ti0=time.time()
    #break
    inputs = batch["image"].unsqueeze(1)
    #print(inputs.max().item(),inputs.min().item())
    #continue
    in2=batch["im2"].unsqueeze(1)
   
    inputs = inputs.to(device, dtype=torch.float)

    in2=in2.to(device, dtype=torch.float)
   
    in3= batch["img3"].unsqueeze(1).to(device, dtype=torch.float)
    #in4= batch["img4"].unsqueeze(1)[:,:,win2[0]:win2[0]+256,win2[1]:win2[1]+256,win2[2]:win2[2]+pl].to(device, dtype=torch.float)
    labels=torch.zeros_like(inputs)[:,0]
    la2=labels
    with torch.no_grad():
                if step==0:
                    #Uout,uo,fo,zs1,size1 =  U(PA(torch.cat([inputs,in2],1)))
                    p1,f1,zs1,s1,lb1,uo =  bvsup256(U,(PA0(torch.cat([inputs,in2],1))),labels)#,xz1=x1,xz2=x2,yz1=y1,yz2=y2)
                    uu=(cp3(p1,nb.cuda())>0)#cp2(p1,BA.max(-1)[0].cuda())
                    p1=p1[uu]
                    f1=f1[uu]
                    zs1=zs1[uu]
                    s1=s1[uu]
                    
                    
                else:
                    #print(kel)
                    #dfghjk
                    p1=zp2#[kel]
                    f1=zf2#[kel]
                    zs1=zzsk#[kel]
                    s1=zzs2#[kel]
                    lb1=labels[p1[:,0],p1[:,1],p1[:,2],p1[:,3]]
                    uo=zu2
                    #weqwe
                #Uout2,uo2,fo2,zs2,size2 =  U(PA(torch.cat([inputs,in2],1)))
                p2,f2,zs2,s2,lb2,u2 =  bvsup256(U,(PA(torch.cat([in2,in3],1))),la2)#,xz1=x1,xz2=x2,yz1=y1,yz2=y2)
                uu=(cp3(p2,nb.cuda())>0)#(p2[:,1]>x1)*(p2[:,1]<x2)*(p2[:,2]>y1)*(p2[:,2]<y2)#cp2(p1,BA.max(-1)[0].cuda())
                p2=p2[uu]
                f2=f2[uu]
                zs2=zs2[uu]
                s2=s2[uu]              
                # if step==138:
                #     p21,f21,zs21,s21,lb21,u21 =  bvsup256(U,(PA(torch.cat([in2,in2],1))),la2,xz1=x1,xz2=x2,yz1=y1,yz2=y2)
                #     p2=torch.cat([p2,p21],0)
                #     f2=torch.cat([f2,f21],0)
                #     zs2=torch.cat([zs2,zs21],0)
                #     s2=torch.cat([s2,s21],0)
                #     l2=torch.cat([lb2,lb21],0)
                #     u2=u2+u21
                #if step==:sdfghj
    zp2=p2.clone()
    zf2=f2.clone()
    zzsk=zs2.clone()
    zzs2=s2.clone()
    zu2=u2.clone()

 
    p1=p1[:,1:]
    p2=p2[:,1:]
    p1[:,2]=p1[:,2]*ZS
    p2[:,2]=p2[:,2]*ZS          
 
    #lb2=lb2[pq]
    #s2=s2[pq]     
    qf,ql,px,gf,gl,py=f1,lb1,p1,f2,lb2,p2
    
   
    if 1:# qf.shape[0]>5 and gf.shape[0]>5:

        ###########################################################
     
     
##############################################################
        if step>0:
            dx=dx2
            dxex=dx2ex
            dx.pop(100000)
    
        else:

            m, n = px.shape[0], px.shape[0]
            px1=px.clone()
            px1[:,2]=px1[:,2]
            distmat = torch.pow(px1.float(), 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(px1.float(), 2).sum(dim=1, keepdim=True).expand(n, m).t()
            distmat.addmm_(1, -2, px1.float(),px1.float().t())
            qx,q=distmat.topk(min(6,p1.shape[0]),largest=False)
            qx=qx[:,1:]
            q=q[:,1:]
            if q.shape[1]<5:
                while q.shape[1]<5:
                    q=torch.cat([q,q[:,-1].unsqueeze(-1)],1)
                    qx=torch.cat([qx,qx[:,-1].unsqueeze(-1)],1)
        
       
            ey=[]
            ep=[]
            es=[]
            ezs=[]
            #epx=px*px
       
            for jk in range(5):
                ey.append(qf[q[:,jk].unsqueeze(1)])
                t=px[q[:,jk]]-px
                ep.append(torch.abs(t).unsqueeze(1))
            epy=torch.cat(ep,1).float()
            ey=torch.cat(ey,1)
            epyy=torch.sqrt((epy*epy).sum(-1))
        
            for jk in range(5):
                ts1=s1[q[:,jk]]
                es.append(torch.cat([(ts1-s1).unsqueeze(1),(ts1/(s1+1)).unsqueeze(1),
                                     epyy[:,jk].unsqueeze(1),(ts1-epyy[:,jk]).unsqueeze(1),((ts1-epyy[:,jk])>0).float().unsqueeze(1),(ts1/(epyy[:,jk]+0.0001)).unsqueeze(1)],1).unsqueeze(1))
                tzs1=zs1[q[:,jk]]
                ezs.append(torch.cat([zs1.unsqueeze(1),tzs1.unsqueeze(1),torch.cat([((zs1[:,0]>0.5)*(tzs1[:,1]>0.5)).unsqueeze(1).float(),((zs1[:,0])*(tzs1[:,1])).unsqueeze(1)],1).unsqueeze(1)],2))
       
            esp=torch.cat(es,1)
            ezsp=torch.cat(ezs,1)
            with torch.no_grad():
                score=F.sigmoid(EN(qf.unsqueeze(1),ey,epy.float(),esp,ezsp))
       
            ss=(score[:,:-1]>0.5).sum(1)  
            ss1=(score>0.9)
            ss2=(score>0.8)
            t1=[]
            tx=[]
            dx={}
            dxm={}
            DD={}
            for i in range(px.shape[0]):
                if i not in t1 :
                    tm=[i]
                    tmm=[ql[i].item()]
                    if ss[i]>0:
                        ts=ss1[i,:]>ss1[i,-1]
                        ts1=ss2[i]
                        for  j in range(ts.shape[0]-1):
                            if ts[j] or (ts1[j] and epyy[i,j]<max(10,0.6*s1[i])):
                                t1.append(q[i,j].item())
                                tm.append(q[i,j].item())
                                tmm.append(ql[q[i,j]].item())
                    dx[i]=tm
                    dxm[i]=tmm
                    tx.append(i)
            
#########################################################


        # eroded_image = erosion(u2.unsqueeze(0))
        # qq=eroded_image[0][:,:,:]>0
        # qq=qq.cpu().numpy()
        # # c1=label(qq)
        # # for i in regionprops(c1):
        # #     #if i.label==20:aasd
        # #     tx1,ty1,tz1,tx2,ty2,tz2=i.bbox
        # #     if (tx2-tx1>20 or ty2-ty1>20) or tz2-tz1>8 :
        # #         qq[tx1:tx2,ty1:ty2,tz1:tz2][i.image]=cut(i.image.copy())[i.image]    
        # c1=label(qq)
        c2=label((u2>0).cpu().numpy())
        #plt.imshow(c1[:,:,12:16].max(-1))
        ce={}
        cef={}
        ce2={}
        cef2={}
        for i in range(p2.shape[0]):
            x,y,z=p2[i,:].cpu().numpy().astype(int)
            z=z//ZS
            # ta=c1[x,y,z]
            # #c2[c1==ta]=ta
            # ce[ta]=ce.get(ta,[])+[i]
            # cef[i]=ta

     
            ta=c2[int(x),int(y),int(z)]
            #c2[c1==ta]=ta
            ce2[ta]=ce2.get(ta,[])+[i]
            cef2[i]=ta
        m, n = py.shape[0], py.shape[0]
        py1=py.clone()
        py1[:,2]=py1[:,2]
        distmat = torch.pow(py1.float(), 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(py1.float(), 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, py1.float(),py1.float().t())
        qx,q=distmat.topk(min(6,py.shape[0]),largest=False)
        qx=qx[:,1:]
        q=q[:,1:]
                       
        if q.shape[1]<5:
            while q.shape[1]<5:
                q=torch.cat([q,q[:,-1].unsqueeze(-1)],1)
                qx=torch.cat([qx,qx[:,-1].unsqueeze(-1)],1)  
   
 
            
            
        ey=[]
        ep=[]
        es=[]
        ezs=[]
        #epx=px*px
   
        for jk in range(5):
            ey.append(gf[q[:,jk].unsqueeze(1)])
            t=py[q[:,jk]]-py
            ep.append(torch.abs(t).unsqueeze(1))
        epy=torch.cat(ep,1).float()
        ey=torch.cat(ey,1)
        epyy=torch.sqrt((epy*epy).sum(-1))
        epy2=epy.clone()
        epy2[:,:,2]=(epy2[:,:,2]-ZS).clip(0,200)
        epyy2=torch.sqrt((epy2*epy2).sum(-1))
        for jk in range(5):
            ts1=s2[q[:,jk]]
            es.append(torch.cat([(ts1-s2).unsqueeze(1),(ts1/(s2+1)).unsqueeze(1),
                                 epyy[:,jk].unsqueeze(1),(ts1-epyy[:,jk]).unsqueeze(1),((ts1-epyy[:,jk])>0).float().unsqueeze(1),(ts1/(epyy[:,jk]+0.0001)).unsqueeze(1)],1).unsqueeze(1))
            tzs1=zs2[q[:,jk]]
            ezs.append(torch.cat([zs2.unsqueeze(1),tzs1.unsqueeze(1),torch.cat([((zs2[:,0]>0.5)*(tzs1[:,1]>0.5)).unsqueeze(1).float(),((zs2[:,0])*(tzs1[:,1])).unsqueeze(1)],1).unsqueeze(1)],2))
   
        esp=torch.cat(es,1)
        ezsp=torch.cat(ezs,1)
        with torch.no_grad():
            score=F.sigmoid(EN(gf.unsqueeze(1),ey,epy.float(),esp,ezsp))
       
        ss=(score[:,:-1]>0.5).sum(1)  
       
        ss=(score[:,:-1]>0.5).sum(1)  
        ss1=(score>0.9)
        ss2=score#(score>0.6)
        t1=[]
        tx=[]
        dx2={}
        dx2m={}
        DD2={}
        dxf={}
        for i in range(py.shape[0]):
            #if i not in t1:
                tm=[i]
                dxf[i]=i
                if (cef2.get(i,0))>0:
                    tk=ce2[cef2[i]]
                    cc=cef2
                    cq=c2
                    
                    # if len(tk)>100 and (cef.get(i,0))>0:
                    #     tk=ce[cef[i]]  
                    #     cc=cef
                    #     cq=c1
                    if 1:
                            ts=ss1[i,:]>ss1[i,-1]
                            ts1=ss2[i,:]>ss2[i,-1]
                            for  j in range(ts.shape[0]-1):
                                if q[i,j].item() in tk:
                                    if ccut(cq==cc[i],py[i],py[q[i,j].item()])==0:
                                        if ts[j]  or (ts1[j] and epyy2[i,j]<max(6,min(30,1*s2[i]))):
                                            t1.append(q[i,j].item())
                                            tm.append(q[i,j].item())
                                            dxf[q[i,j].item()]=i
                #dx2[i]=tm
              
                for ji in tm:
                    dx2[ji]=dx2.get(ji,[])+tm
                tx.append(i)

        dxx={}
        for i in dx2:
            for j in dx2[i]:
                dxx[i]=dxx.get(i,[])+dx2[j]
                dxx[j]=dxx.get(j,[])+dx2[i]
        dx2ex={}
        for i in dxx:
            dx2ex[i]=list(np.unique(dxx[i]))
        ndx={}
        nl=[]
                    
                        
        for i in dx2ex:
            if i not in nl:
                nll=list(np.unique(dx2ex[i]))
                ndx[i]=nll
                nl+=nll
        fdx={}
        for i in ndx:
            for j in ndx[i]:
                fdx[j]=i
        ks2=score.clone()
        kq2=q.clone()
#######################################################
        #yl.append((ya==0).unsqueeze(1))
        px1=px.clone()
        px1[:,2]=px1[:,2]
        py1=py.clone()
        py1[:,2]=py1[:,2]                            
        m, n = qf.shape[0], gf.shape[0]
        distmat = torch.pow(px1.float(), 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(py1.float(), 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, px1.float(),py1.float().t())
        qx,q=distmat.topk(min(15,px.shape[0],py.shape[0]),largest=False)
     
        if q.shape[1]<5:
            while q.shape[1]<5:
                q=torch.cat([q,q[:,-1].unsqueeze(-1)],1)
                qx=torch.cat([qx,qx[:,-1].unsqueeze(-1)],1)          
                       
        else:
            nq=[]
            for i in q:
                l=[]
                no=[]
                for j in i:
                    if j not in no:
                        l.append(j.unsqueeze(0))
                        no+=dx2ex[j.item()]
                    if len(l)==5:
                        break
                while len(l)<5:
                  l.append(l[-1])  
                nq.append(torch.concat(l).unsqueeze(0))
            q=torch.concat(nq)   
   
        ey=[]
        ep=[]
        es=[]
        ezs=[]
        #epx=px*px
   
        for jk in range(5):
            ey.append(gf[q[:,jk].unsqueeze(1)])
            t=py[q[:,jk]]-px
            ep.append(torch.abs(t).unsqueeze(1))
        epy=torch.cat(ep,1).float()
        ey=torch.cat(ey,1)
        epyy=torch.sqrt((epy*epy).sum(-1))
        efy=torch.sqrt((epy*epy)[:,:,:2].sum(-1))
    
        for jk in range(5):
            ts1=s2[q[:,jk]]
            es.append(torch.cat([(ts1-s1).unsqueeze(1),(ts1/(s1+1)).unsqueeze(1),
                                 epyy[:,jk].unsqueeze(1),(ts1-epyy[:,jk]).unsqueeze(1),((ts1-epyy[:,jk])>0).float().unsqueeze(1),(ts1/(epyy[:,jk]+0.0001)).unsqueeze(1)],1).unsqueeze(1))
            tzs1=zs2[q[:,jk]]
            ezs.append(torch.cat([zs1.unsqueeze(1),tzs1.unsqueeze(1),torch.cat([((zs1[:,0]>0.5)*(tzs1[:,1]>0.5)).unsqueeze(1).float(),((zs1[:,0])*(tzs1[:,1])).unsqueeze(1)],1).unsqueeze(1)],2))
   
        esp=torch.cat(es,1)
        ezsp=torch.cat(ezs,1)
     
        with torch.no_grad():
            score=(EX(qf.unsqueeze(1),ey,epy.float(),esp,ezsp))
            sp=score[:,-2:]
            #sp2=score[:,-12:-2]
            score=F.sigmoid(score[:,:-2])
            kscore=score
        
        t1=[]
        tx=[]
        fp=0
        fn=0
        fpd=0
        fnd=0
        IS=0
        c=F.softmax(sp)
        cc=0
        ss1=(score>0.5)
        r={}
        r1s={}
        r2={}
        r2s={}
        r1l={}
        r2l={}
        r1f={}
        r2f={}
        rk={}
        r1su={}
        nn=0
        scorei=score
        
        for di in dx:
            nn+=1
    
            rt=[]
            rs=[]
            res=[]
            g=[]
            
            su=np.max([s1[i].item() for i in dx[di]])
           
            for i in dx[di]:
                    ts=ss1[i,:]
                  
                    
                    g.append(c[i][1].item())
                    if 1:#ts.sum()<2:
                        ss=scorei[i,:4].topk(4)[1]
                        zz=dx2ex[q[i,ss[0]].item()]
                        rt.append(q[i,ss[0]].item())
                        rs.append(scorei[i,ss[0]].item())
                        res.append(epyy[i,ss[0]].item())
                        #st=score[i].argmax().item()
                        for st in ss[1:]:
                            if st==5:continue#break
                            if   q[i,st].item() not in zz :#and q[i,st].item() not in rg:
                                rt.append(q[i,st].item())
                                rs.append(scorei[i,st].item())
                                res.append(epyy[i,st].item())
                                zz=zz+dx2ex[q[i,st].item()]
                                       
            if len(rs)>0:
                rk=rs.index(np.max(rs))
                dj={}
                djn={}
                for ie in range(len(rt)):
                    dj[rt[ie]]=max(dj.get(rt[ie],0),rs[ie])
                    djn[rt[ie]]=res[ie]
                nrt=list(dj.keys())
                nrs=[dj[ie] for ie in nrt]   
                nres=[djn[ie] for ie in nrt] 
                
                #print(g)
                if 1:
                                dx2[100000]=[]
                                rtx=[]
                                rts=[]
                                zr=[]        
                                zrs=[]
                                zrl=[]
                                rtl=[]
                                zr2=[100000,100000]
                                zr2s=[0,0]
                                zr2l=[0,0]
                                zr2.append(100000)
                                zr2s.append(0)
                                
                                zr2l.append(0)
                                nrt=nrt+zr2
                                nrs=nrs+zr2s
                                nres=nres+zr2l
                                ss=torch.from_numpy(np.array(nrs)).topk(min(5,len(nrs)))[1]
                                zz=[]
                       
                                rtx.append(nrt[ss[0]])
                                rts.append(nrs[ss[0]])
                                rtl.append(nres[ss[0]])
                                
                                zz=zz+dx2[nrt[ss[0]]]                       
                                #st=score[i].argmax().item()
                                for st in ss[1:]:
                                    if nrt[st]==100000:break
                                    if  nrt[st] not in zz and nres[st]<max(su,5)*6:
                                        rtx.append(nrt[st])
                                        rts.append(nrs[st])
                                        rtl.append(nres[st])
                                        zz=zz+dx2[nrt[st]]
                
                             
                                r[di]=rtx[:4]#+rtx1
                                r1s[di]=rts[:4]
                                r1l[di]=rtl[:4]
                                if r1f.get(di,0)<np.max(g):
                                    r1f[di]=np.max(g)
                                else:
                                    r1f[di]=0
                                r1su[di]=su
    #if step==8:asfsaf
  
    for i in r:
                       
      
                    r[i]=[[r[i][x],r1s[i][x]] for x in range(len(r[i])) if (x==0 and r1s[i][x]>0.1)  or (r1f[i]>0.01 and r1s[i][x]>0.1) or ( r1f[i]>0.001 and r1s[i][x]>0.2 )]
                   
    #           r[i]=[[r[i][x],r1s[i][x]] for x in range(min(4,len(r[i]))) if r1s[i][x]>0.1]
    
    if step==0:mid={}
    zr={}
    r1=r.copy()
    for i in r:
            if i in mid or  len(mid)==0:#continue
                for  j in r[i]:
            
                    if zr.get(fdx[j[0]],0)<j[1]+0*r1f[i]/5:
                        zr[fdx[j[0]]]=j[1]+0*r1f[i]/5
  
    for i in r1.copy():
        for  j in r1[i].copy():
        
            if zr.get(fdx[j[0]],0)>j[1]+0*r1f[i]/5:
             
                r[i].remove(j)
  
    r0=r.copy()
    
    for i in r0:
        if len(r0[i])>0:
            #zi=np.max([j[1] for j in r0[i]])
            
            r[i]=[j[0]  for j in r0[i] ] 
      
            
    #if step==108:asdad
    if step==0:
        Kim={}
        mid={}
        c=label(uo>0)
        pic=np.zeros(c.shape)
        for j in range(px.shape[0]):
            jj=px[j]
            ta=c[jj[0],jj[1],jj[2]//ZS]
            pic[c==ta]=nid+j
        uk=str(step)
        while len(uk)<3:uk='0'+uk
        Kim[step]=pic.copy()
 
        fontScale = 0.5
        color = (255, 0, 0) 
        thickness = 1
        #cv2.namedWindow("frame", 0)
        pp=pic.max(-1)
        font = cv2.FONT_HERSHEY_SIMPLEX 
        tq=pp>0
        tq=tq.astype(int)*128
        tq=tq[:,:,np.newaxis]
        tq=tq.repeat(3,2) 
        mid={}
        ni=0
        for i in px:
            if ni in r:
                x,y,z=i[0],i[1],i[2]//ZS
                tid=pic[x,y,z]
                mid[ni]=tid
                
                cv2.putText(tq,str(int(tid)),(int(y),int(x)),font,fontScale, color, thickness)
            ni+=1
        nid+=ni   
        cv2.imwrite('./ctc2/'+uk+'.png',tq)
        
  
    
    midn=mid.copy()
  
    cx=label(uo>0,connectivity=3)
    
    ct=[]
    if step>1:
        m, n = px.shape[0], px.shape[0]
        px1=px.clone()
        px1[:,2]=px1[:,2]//ZS
        distmat = torch.pow(px1.float(), 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(px1.float(), 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, px1.float(),px1.float().t())
        qx,zaq=distmat.topk(min(15,p1.shape[0]),largest=False)
        qx=qx[:,1:]
        zaq=zaq[:,1:]
       
        # nq=[]
        # for i in zaq:
        #     l=[]
        #     no=[]
        #     for j in i:
        #         if j not in no:
        #             l.append(j.unsqueeze(0))
        #             no+=dxex[j.item()]
        #         if len(l)==5:
        #             break
        #     while len(l)<5:
        #       l.append(l[-1])  
        #     nq.append(torch.concat(l).unsqueeze(0))
        # zaq=torch.concat(nq)
        #if step==13:dfsf376410
        zover=[]
        cx=label(uo>0,connectivity=3)
        for j in r.copy():
            if j in nkr  or len(r.get(j,[]))==0:continue
            ppx=px[j].unsqueeze(0)
            #if not (ppx[:,0]<310)&(ppx[:,0]>280)&(ppx[:,1]<1085)&(ppx[:,1]>1066)&(ppx[:,2]>60*5)&(ppx[:,2]<70*5):continue
            if j not in mid and len(r[j])>0:
                u=[(i in zover)   for i in dx[j]]
                #print(u)
             
                tu=[fks2[j][list(fkq2[j]).index(i)].item() if i in fkq2[j]  else 0 for i in zaq[j]]
                if np.sum(u)==0:
                    zover.append(j)
                    
                    ff=0
                    #print(zaq[j][0].item() in midn,tu[0]==np.max(tu) )
                    if zaq[j][0].item() in midn and len(r.get(zaq[j][0].item(),[]))>0 and tu[0]==np.max(tu) and j not in dxex.get(zaq[j][0].item() ,[]):
                   
                        ra=zaq[j][0].item()
                        if ra not in jr:continue
                        fb=py1[r[j]][0]
                        fa=py1[r[ra]][0]
                        tx3,ty3,tz3=fb
                        tx4,ty4,tz4=fa
                        tx1,ty1,tz1=px[j]
                        tx2,ty2,tz2=px[ra]
                        tx0,ty0,tz0,s0=jr[ra]
                        txa=tx1/2+tx2/2
                        tya=ty1/2+ty2/2
                        tza=tz1/2+tz2/2
                        txa2=tx3/2+tx4/2
                        tya2=ty3/2+ty4/2
                        tza2=tz3/2+tz4/2
                        flag1=(s0>s1[j]) and (s0>s1[ra])
                        ttl1=(torch.sqrt((tx1-tx0)*(tx1-tx0)+(ty1-ty0)*(ty1-ty0)+0*(tz1-tz0)*(tz1-tz0)))
                        ttl2=(torch.sqrt((tx2-tx0)*(tx2-tx0)+(ty2-ty0)*(ty2-ty0)+0*(tz2-tz0)*(tz2-tz0)))
                        ttl3=(torch.sqrt((tx3-tx0)*(tx3-tx0)+(ty3-ty0)*(ty3-ty0)+0*(tz3-tz0)*(tz3-tz0)))
                        ttl4=(torch.sqrt((tx4-tx0)*(tx4-tx0)+(ty4-ty0)*(ty4-ty0)+0*(tz4-tz0)*(tz4-tz0)))
                        ttla=(torch.sqrt((txa-tx0)*(txa-tx0)+(tya-ty0)*(tya-ty0)+0*(tza-tz0)*(tza-tz0)))
                        ttla2=(torch.sqrt((txa2-tx0)*(txa2-tx0)+(tya2-ty0)*(tya2-ty0)+(tza2-tz0)*(tza2-tz0)))
                        flag2=(ttla<=ttl1 and ttla<=ttl2 and ttla<30 and torch.abs(ttl1-ttl2)<30) or ( ttl3>=ttl1 and ttl4>=ttl2 and  ttla<30)
                        if flag2:
                        
                            nid=nid+1
                            KD[mid[ra]]=[nid,nid+1]
                            nf=mid[ra]
                            mid[ra]=nid
                            mid[j]=nid+1
                            nid=nid+2
                            ff=1
                        
                    elif zaq[j][1].item() in midn and len(r.get(zaq[j][1].item(),[]))>0 and  j not in dxex.get(zaq[j][1].item() ,[]):
                        ra=zaq[j][1].item()
                        if ra not in jr:continue
                        fb=py1[r[j]]
                        fa=py1[r[ra]]
                        tx3,ty3,tz3=fb[0]
                        tx4,ty4,tz4=fa[0]
                        tx1,ty1,tz1=px[j]
                        tx2,ty2,tz2=px[ra]
                        tx0,ty0,tz0,s0=jr[ra]
                        txa=tx1/2+tx2/2
                        tya=ty1/2+ty2/2
                        tza=tz1/2+tz2/2
                        txa2=tx3/2+tx4/2
                        tya2=ty3/2+ty4/2
                        tza2=tz3/2+tz4/2
                        flag1=(s0>s1[j]) and (s0>s1[ra])
                        ttl1=(torch.sqrt((tx1-tx0)*(tx1-tx0)+(ty1-ty0)*(ty1-ty0)+0*(tz1-tz0)*(tz1-tz0)))
                        ttl2=(torch.sqrt((tx2-tx0)*(tx2-tx0)+(ty2-ty0)*(ty2-ty0)+0*(tz2-tz0)*(tz2-tz0)))
                        ttl3=(torch.sqrt((tx3-tx0)*(tx3-tx0)+(ty3-ty0)*(ty3-ty0)+0*(tz3-tz0)*(tz3-tz0)))
                        ttl4=(torch.sqrt((tx4-tx0)*(tx4-tx0)+(ty4-ty0)*(ty4-ty0)+0*(tz4-tz0)*(tz4-tz0)))
                        ttla=(torch.sqrt((txa-tx0)*(txa-tx0)+(tya-ty0)*(tya-ty0)+0*(tza-tz0)*(tza-tz0)))
                        ttla2=(torch.sqrt((txa2-tx0)*(txa2-tx0)+(tya2-ty0)*(tya2-ty0)+(tza2-tz0)*(tza2-tz0)))
                        flag2=(ttla<=ttl1 and ttla<=ttl2 and ttla<30 and torch.abs(ttl1-ttl2)<30) or ( ttl3>=ttl1 and ttl4>=ttl2 and  ttla<30)
                        
                        if flag2:
                     
                            nid=nid+1
                            KD[mid[ra]]=[nid,nid+1]
                            nf=mid[ra]
                            mid[ra]=nid
                            mid[j]=nid+1
                            nid=nid+2
                            ff=2
                    elif zaq[j][2].item() in midn and len(r.get(zaq[j][2].item(),[]))>0 and  j not in dxex.get(zaq[j][2].item() ,[]):
                        ra=zaq[j][2].item()
                        if ra not in jr:continue
                        fb=py1[r[j]]
                        fa=py1[r[ra]]
                        tx3,ty3,tz3=fb[0]
                        tx4,ty4,tz4=fa[0]
                        tx1,ty1,tz1=px[j]
                        tx2,ty2,tz2=px[ra]
                        tx0,ty0,tz0,s0=jr[ra]
                        txa=tx1/2+tx2/2
                        tya=ty1/2+ty2/2
                        tza=tz1/2+tz2/2
                        txa2=tx3/2+tx4/2
                        tya2=ty3/2+ty4/2
                        tza2=tz3/2+tz4/2
                        flag1=(s0>s1[j]) and (s0>s1[ra])
                        ttl1=(torch.sqrt((tx1-tx0)*(tx1-tx0)+(ty1-ty0)*(ty1-ty0)+0*(tz1-tz0)*(tz1-tz0)))
                        ttl2=(torch.sqrt((tx2-tx0)*(tx2-tx0)+(ty2-ty0)*(ty2-ty0)+0*(tz2-tz0)*(tz2-tz0)))
                        ttl3=(torch.sqrt((tx3-tx0)*(tx3-tx0)+(ty3-ty0)*(ty3-ty0)+0*(tz3-tz0)*(tz3-tz0)))
                        ttl4=(torch.sqrt((tx4-tx0)*(tx4-tx0)+(ty4-ty0)*(ty4-ty0)+0*(tz4-tz0)*(tz4-tz0)))
                        ttla=(torch.sqrt((txa-tx0)*(txa-tx0)+(tya-ty0)*(tya-ty0)+0*(tza-tz0)*(tza-tz0)))
                        ttla2=(torch.sqrt((txa2-tx0)*(txa2-tx0)+(tya2-ty0)*(tya2-ty0)+(tza2-tz0)*(tza2-tz0)))
                        flag2=(ttla<=ttl1 and ttla<=ttl2 and ttla<30 and torch.abs(ttl1-ttl2)<30) or ( ttl3>=ttl1 and ttl4>=ttl2 and  ttla<30)
                        if flag2:
                     
    
                            nid=nid+1
                            KD[mid[ra]]=[nid,nid+1]
                            nf=mid[ra]
                            mid[ra]=nid
                            mid[j]=nid+1
                            nid=nid+2
                            ff=3
                    if ff>0:
                        #if mid[j]==3896:sasfa
                        fontScale =1
                        color = (255, 0, 0) 
                        thickness = 2
                        jj=px[j]
                        ta=cx[jj[0],jj[1],jj[2]//ZS]
                        q1=np.zeros(pic.shape)
                        q2=np.zeros(pic.shape)
                        q1=fill(q1,jj[0],jj[1],jj[2]//ZS,v=1,s=min(50,max(18,s1[j].item())//1))
                        q2=fill(q2,jj[0],jj[1],jj[2]//ZS,v=1,s=1)
                        tt=cx==ta
                        q1=q1*tt*(pic==0)+q2
                        pic[q1>0]=mid[j]
                        tq[:,:,0][(pic==mid[j]).max(-1)]=128
                        tq[:,:,1][(pic==mid[j]).max(-1)]=128
                        tq[:,:,2][(pic==mid[j]).max(-1)]=128
                        cv2.putText(tq,str(int(mid[j])),(int(jj[1].item()),int(jj[0].item())),font,fontScale, (0,0,255), thickness)
                        jj1=px[ra]
                        #ta=pic[jj1[0],jj1[1],jj1[2]//10]
                        q1=np.zeros(pic.shape)
                       
                        q1=fill(q1,jj1[0],jj1[1],jj1[2]//ZS,v=1,s=1)
                        q=(pic==midn[ra])+q1
                        pic[q>0]=mid[ra]
                        cv2.putText(tq,str(int(mid[ra])),(int(jj1[1].item()),int(jj1[0].item())),font,fontScale, (0,0,255), thickness)
                        
    if step>0:      
                    fks2=ks2
                    fkq2=kq2
                    jr={}
                    for i in r:
                        if len(r[i])>0:
                            i=int(i)
                            jr[r[i][0]]=[px[i][0].item(),px[i][1].item(),px[i][2].item(),s1[i].item()]
                            if len(r[i])>1:
                                jr[r[i][1]]=[px[i][0].item(),px[i][1].item(),px[i][2].item(),s1[i].item()]
                    nkr=[]
                    for i in r:
                        i=int(i)
                        if len(r[i])>0 and i in mid:
                            nkr+=dx2ex[r[i][0]]
                    uk=str(step)
                    while len(uk)<3:uk='0'+uk
                    Kim[step]=pic.copy()
                    # image=np.zeros([pic.shape[2],pic.shape[0],pic.shape[1]])
                    # for j in range(pic.shape[2]):
                    #     image[j]=pic[:,:,j]
                    # image=image[np.newaxis][:,:,np.newaxis][:,:,:,:,:,np.newaxis]
                    # tifffile.imwrite('../Fluo-N3DH-CE/02_RES/mask'+uk+'.tif',
                    #                  image.astype('int16'),
                    #                  shape=image.shape,
                    #                  imagej=True,
                    #                  resolution=(1/1, 1/1),
                    #                  metadata={'spacing': 1,
                    #                            'unit': 'um',
                    #                            'axes': 'TZCYXS',
                    #                            })  

                    a=np.concatenate([tqx,tq],0)
                     
                    cv2.imwrite('./ctc2/'+uk+'.png',a)
    
    pic=np.zeros(cx.shape)
    cx=label(u2>0,connectivity=3)
    ntl=[]
    midu=np.sort(list(mid.keys()))#[::-1]
    for j in midu:
        if j in r:
            dj=r[j]
            if len(dj)==1 :   
                ui=[o for o in dj if o not in ntl]
                if len(ui)>0:
                    ui=ui[0]
                else:
                    continue
                jj=py[ui]
                if ui in ntl:continue
                ntl+=(dx2ex[ui])
                ta=cx[jj[0],jj[1],jj[2]//ZS]
                if 1:#ta in ct:
                    q1=np.zeros(pic.shape)
                    q2=np.zeros(pic.shape)
                    q1=fill(q1,jj[0],jj[1],jj[2]//ZS,v=1,s=min(50,max(18,s2[dj[0]].item())//1))
                    q2=fill(q2,jj[0],jj[1],jj[2]//ZS,v=1,s=11)
                    tt=cx==ta
                    q1=q1*tt*(pic==0)+q2
                    pic[q1>0]=mid[j]
                    #if 185 in pic:adad
                else:
                    pic[cx==ta]=mid[j]
                ct.append(ta)
            elif len(dj)>1:
                ui=[o for o in dj if o not in ntl]
                tl=[]
                if len(ui)==0:continue
                tas=[cx[py[qi][0],py[qi][1],py[qi][2]//ZS] for qi in ui]
                for qi in ui:
                    jj=py[qi]
                    ta=cx[jj[0],jj[1],jj[2]//ZS]
                    ntl+=dx2ex[qi]
                    #ntl+=dx2ex[dj[1]]
               
                    if tas.count(ta)>1:
                   
                        q1=np.zeros(pic.shape)
                        q2=np.zeros(pic.shape)
                        q1=fill(q1,jj[0],jj[1],jj[2]//ZS,v=1,s=min(50,max(18,s2[dj[0]].item())//1))
                        q2=fill(q2,jj[0],jj[1],jj[2]//ZS,v=1,s=11)
                        tt=cx==ta
                        q1=q1*tt*(pic==0)+q2
                        pic[q1>0]=nid+1
                        nid+=1
                    
                        tl.append(nid)
                    
               
                       
                    else:
                        
                        if 1:#ta in ct:
                            q1=np.zeros(pic.shape)
                            q2=np.zeros(pic.shape)
                            q1=fill(q1,jj[0],jj[1],jj[2]//ZS,v=1,s=min(50,max(18,s2[dj[0]].item())//1))
                            q2=fill(q2,jj[0],jj[1],jj[2]//ZS,v=1,s=11)
                            tt=cx==ta
                            q1=q1*tt*(pic==0)+q2
                            pic[q1>0]=nid+1
                        else:
                            pic[cx==ta]=nid+1
                        nid+=1
               
                        tl.append(nid)
               
                    KD[mid[j]]=tl                
                 
                     
            
            





    fontScale = 0.6
    color = (255, 0, 0) 
    thickness = 1
    #cv2.namedWindow("frame", 0)
    pp=pic.max(-1)
    font = cv2.FONT_HERSHEY_SIMPLEX 
    tq=pp>0
    tq=tq.astype(int)*128
    tq=tq[:,:,np.newaxis]
    tq=tq.repeat(3,2)
    pp=pic[:,:,15:].max(-1)
    
    tq1=pp>0
    tq1=tq1.astype(int)*128
    tq1=tq1[:,:,np.newaxis]
    tq1=tq1.repeat(3,2)
    pp=pic[:,:,:15].max(-1)
    
    tq2=pp>0
    tq2=tq2.astype(int)*128
    tq2=tq2[:,:,np.newaxis]    
    tq2=tq2.repeat(3,2)
    tqx=in2.max(-1)[0].squeeze().cpu().numpy()
    tqx=tqx.astype(int)
    tqx=tqx[:,:,np.newaxis]
    tqx=tqx.repeat(3,2)
    tqx1=in2[:,:,:,:,15:].max(-1)[0].squeeze().cpu().numpy()
    tqx1=tqx1.astype(int)
    tqx1=tqx1[:,:,np.newaxis]
    tqx1=tqx1.repeat(3,2)
    tqx2=in2[:,:,:,:,:15].max(-1)[0].squeeze().cpu().numpy()
    tqx2=tqx2.astype(int)
    tqx2=tqx2[:,:,np.newaxis]
    tqx2=tqx2.repeat(3,2)
    mid={}
    uk=str(step+1)
    while len(uk)<3:uk='0'+uk
    ni=0
    ct=[]
    for ix in r:
        for iix in r[ix]:
            #iix=iix[0]
            i=py[iix].cpu().numpy()
            x,y,z=i[0],i[1],i[2]//ZS
            tid=pic[x,y,z]
            if  tid ==0:continue
            mid[iix]=tid
            
            ct.append(tid)
            
            
            
            cv2.putText(tq,str(int(tid)),(int(y),int(x)),font,fontScale, color, thickness)
    
    a=np.concatenate([tqx,tq],0)
    b=np.concatenate([tqx1,tq1],0)
    c=np.concatenate([tqx2,tq2],0)
    cv2.imwrite('./ctc2/'+uk+'.png',np.concatenate([a,b,c],1)) 
    
    
    ad=[]
    nmid={}
    for i in mid:
        if mid[i] not in ad:
            nmid[i]=mid[i]
            ad.append(mid[i])
    ll=np.unique(pic)
    for i in ll[1:]:
        if i not in ad:
            pic[pic==i]=0
    mid=nmid
    uk=str(step+1)
    while len(uk)<3:uk='0'+uk
    Kim[step+1]=pic.copy()
    # image=np.zeros([pic.shape[2],pic.shape[0],pic.shape[1]])
    # for j in range(pic.shape[2]):
    #     image[j]=pic[:,:,j]
    # image=image[np.newaxis][:,:,np.newaxis][:,:,:,:,:,np.newaxis].astype('uint16')
    # tifffile.imwrite('../Fluo-N3DH-CE/02_RES/mask'+uk+'.tif',
    #                  image,
    #                  shape=image.shape,
    #           )     
    
    #1138,1241



KK=KD.copy() 

path=args.out_dir
pa=np.sort(os.listdir(path))
S={}
E={}
B={}
now=[]
for i in tqdm(range(len(Kim))):
    
    if 1:
        i1=np.unique(Kim[i])
        #i1=tim(Image.open(path+pa[i]))#[i1:i2,i3:i4,i5:]
        for j in i1[1:]:
            if i==0:
               S[j]=0
               now.append(j)
            elif j not in now:
                S[j]=i
                now.append(j)
        for j in now.copy():
            if j not in i1:
                E[j]=i-1
                now.remove(j)
           

for j in now:
    E[j]=kh
dx=pd.DataFrame()
l1,l2,l3=[],[],[]
l4=[]
fkd={}
for i in KD:
    for j in KD[i]:
        fkd[j]=i
for i in range(len(S)):
    jj=list(S.keys())[i]
    l1.append(jj)
    l2.append(S[jj])
    l3.append(E[jj])
    l4.append(int(fkd.get(jj,0)))




dx['id']=l1
dx['s']=l2
dx['e']=l3
dx['d']=-dx.s+dx.e
dx['b']=l4

nl=dx.groupby('id')['d'].mean().to_dict()
ns=dx.groupby('id')['s'].mean().to_dict()
ne=dx.groupby('id')['e'].mean().to_dict()





xdd={}
nxdd=[]
fno=[]
FKD={}
fxdd={}


nfkd={}
for i in np.sort(list(fkd.keys()))[::-1]:
    if i in l1:
        j=fkd[i]
        tl=[]
        while(j not in l1):
            tl.append(j)
            
            j=fkd[j]
            
        nfkd[i]=j
        for k in tl:
            FKD[k]=j
nkd={}
nf={}
for i in nfkd:
    nkd[nfkd[i]]=nkd.get(nfkd[i],[])+[i]

so=[ns[nkd[i][0]] for i in nkd]
i1=np.unique(Kim[0])
qfno=[]
for ni in range(1):
    i1=np.unique(Kim[ni])
    for i in i1:
        if i>0:
            nee=0
            ll=[i]
            sl=KD.get(i,[])
            while len(sl)>0:
                ll+=sl
                ssl=[]
                for j in sl:
                    ssl+=KD.get(j,[])
                nee=[ne.get(k,0) for k in sl]
                sl=ssl
                nee=np.max(nee)
            if nee<args.low:
                qfno+=ll
            
# for i in i1:
#     if i not in nkd and i>0:
#         fno.append(i)
#     else:
#         if i>0:
#             f=0
#             for j in nkd[i]:
#                 if j not in nkd and nl[j]<3:
#                     fno.append(j)
#                     f+=1
#                 else:
#                     ff=0
#                     if j in nkd:
#                         for k in nkd[j]:
#                             if k not in nkd and nl[k]<3:
#                                 fno.append(k)
#                                 ff+=1
#                         if ff>=len(nkd[j]):
#                             fno.append(j)
#                             f+=1
#             if f>=len(nkd[i]):
#                 fno.append(i)
for i in nkd:
    
    if len(nkd[i])>=0:
        ff=0
        if len(nkd[i])>2:
            print(i,nkd[i])
            ff=1
        for j in nkd[i].copy():
            if ns.get(j,0)<kh-5 and nl.get(j,-1)<=0 and (ne.get(j,0)<kh and len(nkd.get(j,[]))<1):
             
                nkd[i].remove(j)
                fno.append(j)
            elif  (ne.get(j,0)<args.low and len(nkd.get(j,[]))<1):
             
                nkd[i].remove(j)
                fno.append(j)
        if len(nkd[i])==1:
            FKD[nkd[i][0]]=i
            nf[nkd[i][0]]=i
for i in nf:
   
    nkd[nf[i]]=nkd.get(i,[])  
    if i in nkd:
        nkd.pop(i)   
nns=[]
for i in FKD.copy():
    if i in nns:continue
    j=FKD[i]
    ss=[]
    while(j in FKD):
        ss.append(j)
        j=FKD[j]
    
    nns+=ss
    for iq in ss+[i]:
        FKD[iq]=j
    FKD[i]=j
    
no=[]#dx.loc[(dx.d<=2)&(dx.e<139)&(dx.s<132)&(dx.s>0)].id.to_list()

KD1=nkd#KD.copy()

pa=np.sort(os.listdir(path))
S={}
E={}
B={}
now=[]
nkim={}
for i in tqdm(range(len(Kim))):
    if 1:
        i1=Kim[i].copy()
        # i1=tim(Image.open(path+pa[i]))#[i1:i2,i3:i4,i5:]
        ii=np.unique(i1)[1:]
        for j in qfno:
            if j in ii:
               
                    i1[i1==j]=0
        for j in fno:
            if j in ii:
                if fkd.get(j,-1)>=0:
                     cc=fkd[j]
                     if cc not in fno and cc not in FKD :
                         #if np.sum([ij in FKD or ij in fno for ij in KD[cc]])==len(KD[cc]):
                            i1[i1==j]=cc
                         # else:
                         #     i1[i1==j]=0
                     else:
                        i1[i1==j]=0
                else:
                    i1[i1==j]=0
        for j in ii:
            if j in FKD:
         
                #print(j,FKD[j])
                i1[i1==j]=FKD[j]
        ii=np.unique(i1)[1:]
        for j in ii:
            jk= nkd.get(j,[]) 
            for jj in jk:
                if jj in ii:
                    i1[i1==j]=0
        ii=np.unique(i1)[1:]
        for j in ii:
            if i==0:
               S[j]=0
               now.append(j)
            elif j not in now:
                S[j]=i
                now.append(j)
        for j in now.copy():
            if j not in ii:
                E[j]=i-1
                now.remove(j)
    
        nkim[i]=i1
        image=np.zeros([i1.shape[2],i1.shape[0],i1.shape[1]])
        for j in range(i1.shape[2]):
            image[j]=i1[:,:,j]
        uk=str(i)
        while len(uk)<3:uk='0'+uk
        
        # image=np.zeros([pic.shape[2],pic.shape[0],pic.shape[1]])
        # for j in range(pic.shape[2]):
        #     image[j]=pic[:,:,j]
        # image=image[np.newaxis][:,:,np.newaxis][:,:,:,:,:,np.newaxis].astype('uint16')
        # tifffile.imwrite('../Fluo-N3DH-CE/02_RES/mask'+uk+'.tif',
        image=image[np.newaxis][:,:,np.newaxis][:,:,:,:,:,np.newaxis].astype('uint16')
        tifffile.imwrite(path+'mask'+uk+'.tif',
                          image,
                          shape=image.shape,
                    )  
        

    
        fontScale = 1
        color = (255, 0, 0) 
        thickness = 2
        pic=i1.copy()
        #cv2.namedWindow("frame", 0)
        pp=pic.max(-1)
        font = cv2.FONT_HERSHEY_SIMPLEX 
        tq=pp>0
        tq=tq.astype(int)*128
        tq=tq[:,:,np.newaxis]
        tq=tq.repeat(3,2)
        pp=pic[:,:,15:].max(-1)
        
        tq1=pp>0
        tq1=tq1.astype(int)*128
        tq1=tq1[:,:,np.newaxis]
        tq1=tq1.repeat(3,2)
        pp=pic[:,:,:15].max(-1)
        
        tq2=pp>0
        tq2=tq2.astype(int)*128
        tq2=tq2[:,:,np.newaxis]    
        tq2=tq2.repeat(3,2)
        
        mid={}
       
        ni=0
        ct=[]
        for ix in regionprops(pic.astype(int)):
            x,y,z=ix.centroid
            tid=ix.label
            if tid>0:
                #print(tid)

                
                
                
                cv2.putText(tq,str(int(tid)),(int(y),int(x)),font,fontScale, color, thickness)
        
        a=tq
        b=tq1
        c=tq2
        cv2.imwrite('./ctc2x/'+uk+'.png',np.concatenate([a,b,c],1)) 

for j in now:
    E[j]=kh
fkd={}
for i in KD1:
    for j in KD1[i]:
        fkd[j]=i
    
xdd={}
nxdd=[]
fno=[]
FKD={}
fxdd={}
l1=list(S.keys())

nfkd={}
for i in np.sort(list(fkd.keys()))[::-1]:
    if i in l1:
        j=fkd[i]
        tl=[]
        while(j not in l1):
            tl.append(j)
            
            j=fkd[j]
            
        nfkd[i]=j
        for k in tl:
            FKD[k]=j
# nkd={}
# nKD={}
# for i in KD1:
#     for j in KD1[i]:
#         nKD[j]=i








l1,l2,l3,l4=[],[],[],[]
for j in S:
            l1.append(int(j))
            l2.append(int(S[j]))
            l3.append(int(E[j]))
            l4.append(int(nfkd.get(j,0)))


rd=pd.DataFrame()
rd['n']=l1
rd['s']=l2
rd['e']=l3
rd['b']=l4





rd.to_csv(path+'/res_track.txt',index=False,header=False,sep=' ')


