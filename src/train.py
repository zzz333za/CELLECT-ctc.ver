
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 14:25:14 2022

@author: zzz333-pc
"""



import os
import cv2
import glob

import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from torch.utils.data import Dataset
from tqdm import tqdm_notebook as tqdm
from matplotlib import pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
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
from scipy.ndimage import rotate,zoom
from skimage.registration import phase_cross_correlation
from skimage.registration._phase_cross_correlation import _upsampled_dft
from scipy.ndimage import fourier_shift
from tqdm  import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage import data, util
from skimage.measure import label,regionprops
from recoloss import CrossEntropyLabelSmooth,TripletLoss
#import lovasz_losses as L
import random
import time
from sklearn.model_selection import KFold
from unetext3Dn_con7s import UNet3D
import torch.distributed as dist

from torch.utils.data.distributed import DistributedSampler
import argparse

# 创建解析器
parser = argparse.ArgumentParser(description="Training script for the model")

# 添加参数
parser.add_argument('--data_dir', type=str, required=True, help="Path to the training data directory")
parser.add_argument('--out_dir', type=str, required=True, help="Path to the model directory")

parser.add_argument('--resolution_z', type=int, default=10, help="ratio of resolution z/xy")
parser.add_argument('--patch_size_xy', type=int, default=256, help="patch size xy")
parser.add_argument('--patch_size_z', type=int, default=31, help="patch size z")
parser.add_argument('--noise', type=int, default=100, help="noise level")



# 解析参数
args = parser.parse_args()
ZS=args.resolution_z
def rget(t):
    for i in np.unique(t):
        if i>0:
            t[t==i]=((t==i).sum(axis=(1,2))>0).sum()/2
    return t
def fill(n,x,y,z,v=1,s=4):
    rr,cc=draw.ellipse(int(x),int(y), s, s)
    ir=(rr>0)*(rr<n.shape[0])
    ic=(cc>0)*(cc<n.shape[1])
    ii=ir*ic
    rr=rr[ii]
    cc=cc[ii]
    z1=int(max(0,z-3))
    z2=int(min(100,z+4))

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
dpath=args.data_dir
op='01'
l=os.listdir(dpath+'/'+op+'/')
l=[i for i in l if 'tif' in i]

D={}

for i in tqdm(l):
    if 'tif' in i:         
        
        num=int(i.split('.')[0][1:])
        D[num]=dpath+'/'+op+'/'+i
        
kh1=len(D)

d1=pd.read_table(dpath+'/'+op+'_GT/TRA/man_track.txt',sep=' ',header=None).values#,skiprows=3)
gd1={}
gj1={}
for i in d1:
    gj1[i[1]]=gj1.get(i[1],[])+[{i[0]:i[-1]}]
    gd1[i[0]]=i[-1]

op='02'
l=os.listdir(dpath+'/'+op+'/')
l=[i for i in l if 'tif' in i]

vD={}
for i in tqdm(l):
    if 'tif' in i:         
        
        num=int(i.split('.')[0][1:])
        vD[num]=dpath+'/'+op+'/'+i
kh2=len(vD)
d1=pd.read_table(dpath+'/'+op+'_GT/TRA/man_track.txt',sep=' ',header=None).values#,skiprows=3)
gd2={}
gj2={}
for i in d1:
    gd2[i[0]]=i[-1] 
    gj2[i[1]]=gj2.get(i[1],[])+[{i[0]:i[-1]}]       
      
     
    
oa=range(0,512,2)
ob=range(0,712,2)    
def small(x):
    #x=x[oa]
    #x=x[:,ob]
    '''x=torch.from_numpy(x.astype(float))
    x=torch.max_pool3d(x.unsqueeze(0), [2,2,1])
    x=x[0].numpy()'''
    return x
def cen(xq):
    x=np.zeros(xq.shape)
    c=regionprops(xq)
    for a in c:
        a1,a2,a3=a.centroid
        a1=int(a1)
        a2=int(a2)
        a3=int(a3)
        x[a1,a2,a3]=a.label
        #x=fill(x,a1,a2,a3)
    return x


dpath=args.data_dir
KK={}
K1={}
K2={}
K3={}
K4={}
kh1=len(os.listdir(dpath+'/01_ST/SEG/'))-3
kh2=len(os.listdir(dpath+'/02_ST/SEG/'))-3
print(kh1,kh2)
for i in tqdm(range(max(kh1,kh2))):
    if i<=kh1:
        KK[D[i]]=tim(Image.open(D[i]))
        K3[D[i]]=rget(tim(Image.open(dpath+'/01_ST/SEG/man_seg'+D[i].split('/')[-1][1:])))
        K1[D[i]]=tim(Image.open(dpath+'/01_GT/TRA/man_track'+D[i].split('/')[-1][1:]))
        K2[D[i]]=tim(Image.open(dpath+'/01_ST/SEG/man_seg'+D[i].split('/')[-1][1:]))
    if i <=kh2:    
        KK[vD[i]]=tim(Image.open(vD[i]))
        K1[vD[i]]=tim(Image.open(dpath+'/02_GT/TRA/man_track'+vD[i].split('/')[-1][1:]))
        K2[vD[i]]=tim(Image.open(dpath+'/02_ST/SEG/man_seg'+vD[i].split('/')[-1][1:]))
        
        K3[vD[i]]=rget(tim(Image.open(dpath+'/02_ST/SEG/man_seg'+vD[i].split('/')[-1][1:])))
img3= KK[D[0]]          
if img3.max()>255:
    def pa(x):
        return torch.log1p(x)
else:
        def pa(x):
            return x
    
    
    
class IntracranialDataset(Dataset):

    def __init__(self, data,le, transform=None):
        
        #self.path = path
        self.data =data
        self.transform = transform
       
    def __len__(self):
        
        return kh1+kh2+20#len(self.data)+185-3

    def __getitem__(self, i):
        
   
        
        j=i
        flag=0
        if i<=kh1-4:
            g=D[j]
    
            o=1
            g2=D[j+o]
      
            g3=D[j+o+1]
            gj=gj1.get(j+o,[])
            o='01'
        elif i<=kh1+kh2-8 :
            j=j-kh1+4
            g=vD[j]
    
            o=1
            g2=vD[j+o]
      
            g3=vD[j+o+1]
            gj=gj2.get(j+o,[])  
            o='02'
        elif i<500:
            o=random.choice(['01','02'])
            if o=='01':
             j=random.choice(list(range(kh1-30,kh1-5)))
             g=D[j]
     
          
             g2=D[j+1]
       
             g3=D[j+1+1]
             gj=gj1.get(j+1,[]) 
            else:
                j=random.choice(list(range(kh2-30,kh2-5)))
                g=vD[j]
        
                g2=vD[j+1]
          
                g3=vD[j+1+1]
                gj=gj2.get(j+1,[])  
           
            
            
        #j=random.choice(x_train)
      
        #print(g)
        
        img=KK[g].copy()#tim(Image.open(g))#[i1:i2,i3:i4,i5:]
        img2=KK[g2].copy()#tim(Image.open(g2))#[i1:i2,i3:i4,i5:]
        img3=KK[g3].copy()#tim(Image.open(g3))#[i1:i2,i3:i4,i5:]

 
        if random.randint(0,2)==1:
            noise=np.random.normal(loc=0,scale=random.randint(5,args.noise),size=img.shape)
            noise2=np.random.normal(loc=0,scale=random.randint(5,args.noise),size=img.shape)
            noise3=np.random.normal(loc=0,scale=random.randint(5,args.noise),size=img.shape)
 
            img=(img+noise).clip(img.min(),img.max())
            img3=(img3+noise2.clip(img3.min(),img3.max()))
            img2=(img2+noise3).clip(img2.min(),img2.max())           
    

        if flag==0:
            t=K1[g]
            t6=K2[g]
            t16=K2[g2]
            t15=K1[g2]
            t=cen(t)
            t15=cen(t15)
       
        t1=t15.copy()
        t7=t16.copy()
        t4=t15.copy()
        t5=t15.copy()
        t4=t4*0
        t5=t5*0
        for ix in gj:
            i=list(ix.keys())[0]
           
            t1[t1==i]=ix[i]
            t7[t7==i]=ix[i]
            t4[t6==ix[i]]=1
            t5[t16==i]=2
        #t2=np.load('../extradata/mskcc_confocal/mskcc-confocal/ls'+op+'/'+str(j)+'-k3--3d-1-imaris.npy')
        #t3=np.load('../extradata/mskcc_confocal/mskcc-confocal/ls'+op+'/'+str(j)+'-k4-3d-1-imaris.npy')
        
        
        t31=K3[g]#rget(t6.copy())
        t32=K3[g2]#rget(t16.copy())
        pa,pb,pc=args.patch_size_xy,args.patch_size_xy,args.patch_size_z
        xa,xb,xc=random.randint(0,t.shape[0]-pa-0),random.randint(0,t.shape[1]-pb-0 ),random.randint(0,max(0,t.shape[2]-pc) )
       
        
        #print(xa,xb,xc,img.shape)
   
        #print(1)
        b = torch.from_numpy((img.astype(float)))
        
  
        timg = torch.from_numpy(t.astype(float))
        img3 = torch.from_numpy(img3.astype(float))
        c=torch.from_numpy(img2.astype(float))
        d= torch.from_numpy(t1.astype(float))
        
        e1=torch.from_numpy(t4.astype(float))
        e2=torch.from_numpy(t5.astype(float))
        #e3=torch.from_numpy(t14)
        #e4=torch.from_numpy(t15)        
        #t8[t8>0]+=100000
        #t9[t9>0]+=100000
        
        p1=torch.from_numpy(t6.astype(float))
        p2=torch.from_numpy(t7.astype(float))

        size1=torch.from_numpy(t31.astype(float))
        size2=torch.from_numpy(t32.astype(float))
        #print(b.shape,timg.shape,k)
        return {'image': b[xa:xa+pa,xb:xb+pb,xc:xc+pc], 'labels': p1[xa:xa+pa,xb:xb+pb,xc:xc+pc],'im2':c[xa:xa+pa,xb:xb+pb,xc:xc+pc],'la2':p2[xa:xa+pa,xb:xb+pb,xc:xc+pc],
                'p1':timg[xa:xa+pa,xb:xb+pb,xc:xc+pc],'p2':d[xa:xa+pa,xb:xb+pb,xc:xc+pc],'s1':e1[xa:xa+pa,xb:xb+pb,xc:xc+pc],
                's2':e2[xa:xa+pa,xb:xb+pb,xc:xc+pc],
                'size1':size1[xa:xa+pa,xb:xb+pb,xc:xc+pc],'size2':size2[xa:xa+pa,xb:xb+pb,xc:xc+pc],'img3':img3[xa:xa+pa,xb:xb+pb,xc:xc+pc]}         
    
class VDataset(Dataset):

    def __init__(self, data,le, transform=None):
        
        #self.path = path
        self.data =data
        self.transform = transform
       
    def __len__(self):
        
        return 10#len(self.data)-188

    def __getitem__(self, i):
        
   
        
        j=i+kh1+kh2-20
        if j<kh1-2:
            g=D[j]
      
            o=1
            g2=D[j+o]
      
            g3=D[j+o+1]
            gj=gj1.get(j+o,[])
            o='01'
        elif j==kh1+kh2-4:
            j=1
            g=D[j]
      
            o=1
            g2=D[j+o]
      
            g3=D[j+o+1]
            gj=gj1.get(j+o,[])
            o='01'
        else:
            j=j-kh1
            g=vD[j]
      
            o=1
            g2=vD[j+o]
      
            g3=vD[j+o+1]
            gj=gj2.get(j+o,[])  
            o='02'
        #j=random.choice(x_train)
      
        #print(g)
        
        img=tim(Image.open(g))#[i1:i2,i3:i4,i5:]
        img2=tim(Image.open(g2))#[i1:i2,i3:i4,i5:]
        img3=tim(Image.open(g3))#[i1:i2,i3:i4,i5:]
        
      
        t=tim(Image.open(dpath+'/'+o+'_GT/TRA/man_track'+g.split('/')[-1][1:]))
        t6=tim(Image.open(dpath+'/'+o+'_ST/SEG/man_seg'+g.split('/')[-1][1:]))
        t16=tim(Image.open(dpath+'/'+o+'_ST/SEG/man_seg'+g2.split('/')[-1][1:]))
        t15=tim(Image.open(dpath+'/'+o+'_GT/TRA/man_track'+g2.split('/')[-1][1:]))
        t=cen(t)
        t15=cen(t15)
        t1=t15.copy()
        t7=t16.copy()
        t4=t15.copy()
        t5=t15.copy()
        t4[t4>0]=0
        t5[t5>0]=0
        for ix in gj:
            i=list(ix.keys())[0]
          
            t1[t1==i]=ix[i]
            t7[t7==i]=ix[i]
            t4[t6==ix[i]]=1
            t5[t16==i]=2
        #t2=np.load('../extradata/mskcc_confocal/mskcc-confocal/ls'+op+'/'+str(j)+'-k3--3d-1-imaris.npy')
        #t3=np.load('../extradata/mskcc_confocal/mskcc-confocal/ls'+op+'/'+str(j)+'-k4-3d-1-imaris.npy')
        
        img=small(img)
        img2=small(img2)
        img3=small(img3)
        t=small(t)
        t6=small(t6)
        t16=small(t16)
        t1=small(t1)
        t4=small(t4)
        t5=small(t5)
        t7=small(t7)        
        t31=rget(t6.copy())
        t32=rget(t16.copy())
       
        #print(xa,xb,xc,img.shape)
      
        #print(1)
        b = torch.from_numpy((img.astype(float)))
        
      
        timg = torch.from_numpy(t.astype(float))
        img3 = torch.from_numpy(img3.astype(float))
        c=torch.from_numpy(img2.astype(float))
        d= torch.from_numpy(t1.astype(float))
        
        e1=torch.from_numpy(t4.astype(float))
        e2=torch.from_numpy(t5.astype(float))
        #e3=torch.from_numpy(t14)
        #e4=torch.from_numpy(t15)        
        #t8[t8>0]+=100000
        #t9[t9>0]+=100000
        
        p1=torch.from_numpy(t6.astype(float))
        p2=torch.from_numpy(t7.astype(float))
      
        size1=torch.from_numpy(t31.astype(float))
        size2=torch.from_numpy(t32.astype(float))
        #print(b.shape,timg.shape,k)
        return {'image': b, 'labels': p1,'im2':c,'la2':p2,'p1':timg,'p2':d,'s1':e1,'s2':e2,'img3':img3,
                'size1':size1,'size2':size2} 



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

n_epochs = 200
batch_size =1
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
#device = torch.device("cuda:0")#
ids=[0]
#dist.init_process_group(backend="nccl")
device = torch.device("cuda:0")
device1 = torch.device("cpu")


from skimage import draw

tm=2
tm1=1
DK=np.zeros([tm*2+1,tm*2+1,9])
rr,cc=draw.ellipse(tm,tm, tm,tm)
DK[rr,cc,:]=1
DK=torch.from_numpy(DK).float().to(device).reshape([1,1,tm*2+1,tm*2+1,9])
DK1=np.array([[0,0,0,1,0,0,0],[0,0,1,2,1,0,0],[0,1,2,3,2,1,0],[1,2,3,4,3,2,1],[0,1,2,3,2,1,0],[0,0,1,2,1,0,0],[0,0,0,1,0,0,0]])
DK1=torch.from_numpy(DK1).float().to(device).reshape([1,1,7,7,1])#.repeat(1,1,1,1,3)

#G =UNet(1,1)
U =UNet3D(2,6)
EX=EXNet(64,8)
EN=EXNet(64,6)
#U= torch.nn.parallel.DataParallel(U)
#EX= torch.nn.parallel.DataParallel(EX)
#EN= torch.nn.parallel.DataParallel(EN)
#U2 =UNet(1,2)
#model.fc = torch.nn.Linear(2048, 6)
U.to(device)
EX.to(device)
EN.to(device)

plist = [{'params': U.parameters()}]
oU = optim.Adam(plist, lr=2e-4)
plist = [{'params': EX.parameters(), 'lr': 2e-4}]
oEX = optim.Adam(plist)#EXP=torch.load('../track-data/model//EXP+-25.0-1.9528.pth')
plist = [{'params': EN.parameters(), 'lr': 2e-4}]
oEN = optim.Adam(plist)#EXP=torch.load('../track-data/model//EXP+-25.0-1.9528.pth')

weights = torch.tensor([1.0,4.0]).cuda()
criterion = torch.nn.CrossEntropyLoss(reduction='none',weight=weights)
weights2 = torch.tensor([1.0, 4.0,1.0]).cuda()
criterion3 = torch.nn.CrossEntropyLoss(reduction='none',weight=weights2)
weights5 = torch.tensor([1.0, 2,4,8,16]).cuda()
criterion5 = torch.nn.CrossEntropyLoss(reduction='none',weight=weights5)
weights6 = torch.tensor([1.0,1]).cuda()
criterion6 = torch.nn.CrossEntropyLoss(reduction='none',weight=weights6)
#criterion2 = torch.nn.CrossEntropyLoss(reduction='none',weight=weights2)
criterion2 = torch.nn.CrossEntropyLoss(reduction='none')
rkloss=torch.nn.MarginRankingLoss(margin=0.3)
Tloss =  TripletLoss(0.3)
l1loss=torch.nn.L1Loss(reduction='none')
mse=torch.nn.MSELoss()
#criterion1 = torch.nn.CrossEntropyLoss(ignore_index=0,reduction='none')
bce=torch.nn.BCELoss(reduction='none')

#plist = [{'params': G.parameters(), 'lr': 2e-3}]
#oG = optim.Adam(plist, lr=2e-3)



sU=torch.optim.lr_scheduler.ReduceLROnPlateau(oU,factor=0.5,patience=3)
sEX=torch.optim.lr_scheduler.ReduceLROnPlateau(oEX,factor=0.5,patience=3)
sEN=torch.optim.lr_scheduler.ReduceLROnPlateau(oEN,factor=0.5,patience=3)


print('    Total params: %.2fM' % (sum(p.numel() for p in U.parameters())/1000000.0))

wi=[]
for xi in [-1,0,1]:
    for yi in [-1,0,1]:
        for zi in [-1,0,1]:
            wi.append([xi,yi,zi])
#model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
#model = torch.nn.DataParallel(model).cuda()
l0=[]
l1=[]

#model=torch.load('../model/2db.pth')
train_dataset = IntracranialDataset(
 D, le=1000)
#train_sampler = DistributedSampler(train_dataset)
data_loader_train = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,shuffle=True, num_workers=0)
val_dataset = VDataset(
 vD, le=1000)
data_loader_val = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0) 
val_loss=1000
va=10000
win=3
lg=16
ko=3
pk=(1,1,1)
def kf(x):
    
    kn=torch.zeros([x.shape[0],x.shape[1]+2,x.shape[2]+2,x.shape[3]+2])
    kq=torch.zeros([x.shape[0],27,x.shape[1],x.shape[2],x.shape[3]])
    kn[:,1:-1,1:-1,1:-1]+=x.cpu()
    num=0
    for xc in [-1,0,1]:
        for y in [-1,0,1]:
            for z in [-1,0,1]:
            
                    kq[:,num]=kn[:,xc+1:xc+1+x.shape[1],y+1:y+1+x.shape[2],z+1:z+1+x.shape[3]]
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
def crloss(a,b):
    tl1=0#criterion5(a,(b).long())
    for ti in range(1,a.shape[1]):
        tl1+=criterion(torch.cat([a[:,:1],a[:,ti:ti+1]],1),(b>=1).long())
    for ti in range(2,a.shape[1]):
        tl1+=criterion(torch.cat([a[:,:ti].max(1)[0].unsqueeze(1),a[:,ti:ti+1]],1),(b>=2).long())*((b==0)+(b>=2))
    for ti in range(3,a.shape[1]):
        tl1+=criterion(torch.cat([a[:,:ti].max(1)[0].unsqueeze(1),a[:,ti:ti+1]],1),(b>=3).long())*((b==0)+(b>=3))
    for ti in range(4,a.shape[1]):
        tl1+=criterion(torch.cat([a[:,:4].max(1)[0].unsqueeze(1),a[:,ti:ti+1]],1),(b>=4).long())*((b==0)+(b>=4))
    return tl1

def PA(x):
    #x=torch.log1p(x)
    pa(x)
    return x
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

class FixedConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,dilation=1, stride=1, padding=0):
        super(FixedConv3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation=dilation
        
        # 定义卷积核参数，这里使用torch.nn.Parameter将其声明为可学习的参数
        #torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        w=torch.ones(out_channels, in_channels, kernel_size, kernel_size,kernel_size)
        cen=kernel_size//2
        w[:,:,cen,cen,cen]=-cen*cen*cen+1
        self.weight = nn.Parameter(w)
        # 固定参数，不进行梯度更新
        self.weight.requires_grad = False
        
    def forward(self, x):
        return torch.nn.functional.conv3d(x, self.weight, stride=self.stride, padding=self.padding+self.dilation-1,dilation=self.dilation)
T=[]
# 示例用法
fixed_conv_d1 = FixedConv3d(in_channels=64, out_channels=1, kernel_size=3,dilation=1, stride=1, padding=1).cuda()
fixed_conv_d5 = FixedConv3d(in_channels=64, out_channels=1, kernel_size=3,dilation=5, stride=1, padding=1).cuda()
for epoch in range(0,n_epochs):
    
    print('Epoch {}/{}'.format(epoch, n_epochs - 1))
    print('-' * 10)

    U.train()
    if epoch<6 or    (epoch>15 and epoch<=20) or    (epoch>35 and epoch<=40) :
        EX.train()
        EN.train()
    
    else:
     
        EX.eval()
        EN.eval()
    tr_loss = 0
    kk=0

    tk0 = tqdm(data_loader_train, desc="Iteration")
    n=0
    for step, batch in enumerate(tk0):
        #break
        if step%10==0:
            print(tk0)
        inputs = batch["image"].unsqueeze(1) 
        
        labels = batch["labels"]
        in2=batch["im2"].unsqueeze(1)
        la2=batch["la2"]
     
        # inputs = inputs.to(device, dtype=torch.float)
        labels = labels.to(device, dtype=torch.float)
        #in2=in2.to(device, dtype=torch.float)
        la2=la2.to(device, dtype=torch.float)
      
        p1=batch["p1"].to(device, dtype=torch.float).unsqueeze(1)
        p2=batch["p2"].to(device, dtype=torch.float).unsqueeze(1)
        pl1=batch["p1"].to(device, dtype=torch.float).unsqueeze(1)
        pl2=batch["p2"].to(device, dtype=torch.float).unsqueeze(1)
        ee1=list((pl1[0,0][pl1[0,0]!=0]).flatten().cpu().numpy())
        ee2=list((pl2[0,0][pl2[0,0]!=0]).flatten().cpu().numpy())
        lx2=torch.unique(pl1[0,0]).shape[0]
        w3=(lx2//50)*1
        pl1=[]
        pl2=[]
         
        lsize1=batch["size1"]#
        lsize2=batch["size2"]#.to(device, dtype=torch.float) 
        s1=batch["s1"]#.to(device, dtype=torch.float)   
        s2=batch["s2"]#.to(device, dtype=torch.float)   
            
        in3= batch["img3"].unsqueeze(1)#.to(device, dtype=torch.float)
 
        #in4= batch["img4"].unsqueeze(1).to(device, dtype=torch.float)
        Uout,uo,fo,zs1,size1 =  U(PA(torch.cat([inputs,in2],1).to(device, dtype=torch.float)))
        Uout2,uo2,fo2,zs2,size2 =  U(PA(torch.cat([in2,in3],1).to(device, dtype=torch.float)))

        p1[:,0]=F.conv3d((p1[:,0]>0).float().unsqueeze(1), DK1, padding=(tm+1,tm+1,0))
        #p1[:,1]=F.conv3d((p1[:,1]>0).float().unsqueeze(1), DK1, padding=(tm+1,tm+1,0))  
        p2[:,0]=F.conv3d((p2[:,0]>0).float().unsqueeze(1), DK1, padding=(tm+1,tm+1,0))
        #p2[:,1]=F.conv3d((p2[:,1]>0).float().unsqueeze(1), DK1, padding=(tm+1,tm+1,0)) 
        for i in range(3):
            p1=ud(p1)
            p2=ud(p2)        

        #p1=(p1[:,1]>0).float()*2+(p1[:,0]>0).float()
        #p2=(p2[:,1]>0).float()*2+(p2[:,0]>0).float()
        zzs=s1.to(device, dtype=torch.float).clone()
        s1=torch.cat([(s1==1).float(),(s2==2)],0)
     
        #s2=torch.cat([(s2==1).float(),(s4==2)],0)
      
        la3=(labels>0).float()#+2*(la3>0).float()
        la3[la3>2]=2
        la4=(la2>0).float()#+2*(la4>0).float()
        la4[la4>2]=2
        wp1=la3>0
        zp1=wp1*(p1[:,0])
        wp2=la4>0
        zp2=wp2*(p2[:,0])
        # w4=inputs[:,0]/30
        # w4[w4>1]=1
        # w4=1/w4
        # w5=in2[:,0]/30
        # w5[w5>1]=1
        # w5=1/w5
        # w4[w4==np.inf]=1
        # w5[w5==np.inf]=1
        nclc=criterion3(Uout,(la3).long())*(1+w3)#*w4.cuda()
      
        nclc2=criterion3(Uout2,(la4).long())*(1+w3)#*w5.cuda()
        w4=1
        w5=1
        #nloss3=torch.mean(criterion(uo,(zp1).long())[labels>0])
        #nloss4=torch.mean(criterion(uo2,(zp2).long())[la2>0])           
    
        lclc=mse(size1,lsize1.to(device, dtype=torch.float))#*(1+31*ws)#.clip(0,100)#*0.1
        lclc2=mse(size2,lsize2.to(device, dtype=torch.float))#*(1+31*ws2)#.clip(0,100)#*0.1
        #lclc[lclc<1]=0
        #lclc2[lclc2<1]=0
        #bclc=criterion(zs1,(s1[:,0]).long())#+criterion3(zs1*Uout,(s1>0).unsqueeze(0).long())
        #bclc2=criterion(zs3,(s2[:,0]).long())#+criterion3(zs2*Uout2,(s2>0).unsqueeze(0).long())
        '''w1=p1[:,0].clone()
        w2=p2[:,0].clone()
        w1[w1>0]=1
        w2[w2>0]=1'''
     
        #ww=torch.cat([ws.unsqueeze(1),ws2.unsqueeze(1)],1)
        sloss=torch.mean(lclc)+torch.mean(lclc2)#+torch.mean(bclc2)+torch.mean(bclc)#+torch.mean(nlcx)+torch.mean(nlcx2)#+(torch.mean(nclcp)+torch.mean(nclcp2)+torch.mean(nlcxp)+torch.mean(nlcxp2))#+torch.mean(nlc+nlc2)+l1loss(nlc,lc)+l1loss(nlc2,lc2)
        if s1.sum()>0:
            #print(s1.max(),s1[0].sum(),s1[1].sum())
            zs1[zs1!=zs1]=0
            sloss1=torch.mean(bce(F.sigmoid(zs1),s1.unsqueeze(0).to(device, dtype=torch.float)))#+torch.mean(bce(F.sigmoid(zs2),s2.unsqueeze(0))[ws2])
            #sloss1=sloss1*(s1.sum()>0)
        else:
            sloss1=0
        rloss3=0
        kloss=[]
        Uf=fo.transpose(0,1)
        Uf2=fo2.transpose(0,1)
        #uc=(uo.argmax(1)==1)*(Uout.argmax(1)==1)*(size1[:,0]>1)
        uo1=uo.argmax(1)
        #ku=kfs(uo1).cuda()
        
        #uu=(kfs2(uo1).cuda().max(1)[0]==3)*(ku==3).sum(1)*(kfs2(uo[:,3]).cuda().max(1)[0]==uo[:,3])
        #uu2=(kfs2(uo1).cuda().max(1)[0]==2)*(ku==2).sum(1)*(kfs2(uo[:,2]).cuda().max(1)[0]==uo[:,2])
        uc=((kflb(uo[:,:].max(1)[0]).cuda().max(1)[0]==uo[:,4]))*(Uout.argmax(1)==1)#*(size1[:,0]>1)
        #uc=((uu2>6)+(uu>4)+uc1)#*(Uout.argmax(1)==1)#*(size1[:,0]>1)
        u,x,y,z=torch.where(uc.to(device, dtype=torch.float))
        lx=torch.unique(labels).shape[0]
        lx1=torch.unique(labels*uc).shape[0]
        lx3=list(torch.unique(labels*uc).cpu().numpy())
        lx3=[i for i in lx3 if i in np.unique(ee1)]
        
        f1=Uf[:,u,x,y,z].transpose(0,1)
        s1=size1[u,0,x,y,z].to(device, dtype=torch.float)
        zs1=F.sigmoid(zs1)[u,:,x,y,z]
        

        lb1=labels[u,x,y,z]
        zb=zzs[u,x,y,z]==1
    
  
        #uu=(kfs2(uo1).cuda().max(1)[0]==3)*(ku==3).sum(1)*(kfs2(uo2[:,3]).cuda().max(1)[0]==uo2[:,3])
        #uu2=(kfs2(uo1).cuda().max(1)[0]==2)*(ku==2).sum(1)*(kfs2(uo2[:,2]).cuda().max(1)[0]==uo2[:,2])
        uc=((kflb(uo2[:,:].max(1)[0]).cuda().max(1)[0]==uo2[:,4]))*(Uout2.argmax(1)==1)#*(size1[:,0]>1)
        #uc=((uu2>6)+(uu>4)+uc1)#*(Uout2.argmax(1)==1)#*(size1[:,0]>1)
    
        '''kl=((la2*uc1))
        kl=F.conv3d((kl>0).float().unsqueeze(1), DK1, padding=(tm+1,tm+1,0))
        kl[kl>0]=0.1'''
        #txl=[]
        #for i in l1:
         #   if i in l2 or i not in kp2:
          ###      w2[w2==i]=0.1  
        #w2[w2!=0.1]=1
       
        #uo=uo*(Uout.argmax(1)==1).float()
        #uo2=uo2*(Uout2.argmax(1)==1).float()
        nloss1=crloss(uo,(zp1).long())#*(Uout.argmax(1)==1).float()#*w4.cuda()
        nloss2=crloss((uo2),(zp2).long())#*(Uout2.argmax(1)==1).float()#*w5.cuda()
      
        nloss3=torch.mean(nloss1*(labels>0))
        nloss4=torch.mean(nloss2*(la2>0))
        nloss1=torch.mean(nloss1*(Uout.argmax(1)==1).float())
        nloss2=torch.mean(nloss2*(Uout2.argmax(1)==1).float())
        
        
        Uloss=0

        Uloss+=torch.mean(nclc)+torch.mean(nclc2)+nloss1+nloss2+nloss3+nloss4
        nclc=0
        nclc2=0
      
        
        if u.shape[0]>0 and epoch>1:

            #uc=F.avg_pool3d(uc.unsqueeze(0).float(),[win,win,win])>0.9
            #uc=uc[0]
            #Uf2=F.avg_pool3d(Uf2.float(),[win,win,win])
            #uc=(kf(uc.float()).sum(1)>ko).float().cuda()*uc
            #u,x,y,z=torch.where(uc)            
            u2,x2,y2,z2=torch.where(uc.to(device, dtype=torch.float))  
            f2=Uf2[:,u2,x2,y2,z2].transpose(0,1)
            s2=size2[u2,0,x2,y2,z2]
            zs2=F.sigmoid(zs2)[u2,:,x2,y2,z2]
            lb2=la2[u2,x2,y2,z2]
            #s21=size1[u2,0,x2,y2,z2+1]
            #s22=size1[u2,0,x2,y2,z2-1]
            #s2=torch.cat([s2.unsqueeze(1),s21.unsqueeze(1),s22.unsqueeze(1)],1).max(1)[0]            
            p1=torch.cat([u.unsqueeze(1),x.unsqueeze(1),y.unsqueeze(1),z.unsqueeze(1)],1)
            p2=torch.cat([u2.unsqueeze(1),x2.unsqueeze(1),y2.unsqueeze(1),z2.unsqueeze(1)],1)
         
                   
            p1=p1[:,1:]
            p2=p2[:,1:]
            p1[:,2]=p1[:,2]*ZS
            p2[:,2]=p2[:,2]*ZS          
            px=p1.float()
            py=p2.float()
        
           
            qf,ql,px,gf,gl,py=f1,lb1,p1,f2,lb2,p2
            if  qf.shape[0]>5 and gf.shape[0]>5:
                px=px.to(device1)
                py=py.to(device1)
                ###########################################################
       

        
#########################################################################
         
                m, n = px.shape[0], px.shape[0]
                distmat = torch.pow(px.float(), 2).sum(dim=1, keepdim=True).expand(m, n) + \
                      torch.pow(px.float(), 2).sum(dim=1, keepdim=True).expand(n, m).t()
                distmat.addmm_(1, -2, px.float(),px.float().t())
                qx,q=distmat.topk(6,largest=False)
                qx=qx[:,1:].to(device)
                q=q[:,1:].to(device)
                px=px.to(device)
                py=py.to(device)
                   
                   
                sx=[0,1,2,3,4]
                if random.randint(0,1)==1:
                    nq=q.clone()
                    nqx=qx.clone()
                    for ki in range(q.shape[0]):
                        random.shuffle(sx)
                        for jk in range(5):
                            nq[ki,jk]=q[ki,sx[jk]]
                            
                            nqx[ki,jk]=qx[ki,sx[jk]]
                   
                    q=nq
                    qx=nqx
                ey=[]
                ep=[]
                es=[]
                ezs=[]
                #epx=px*px
       
                for jk in range(5):
                    ey.append(qf[q[:,jk].unsqueeze(1)])
                    t=px[q[:,jk]]-px
                    ep.append(torch.abs(t).unsqueeze(1))
                epy=torch.cat(ep,1)
                ey=torch.cat(ey,1)
                epyy=torch.sqrt((epy*epy).sum(-1))
            
                for jk in range(5):
                    ts1=s1[q[:,jk]]
                    es.append(torch.cat([(ts1-s1).unsqueeze(1),(ts1/(s1+1)).unsqueeze(1),
                                         epyy[:,jk].unsqueeze(1),(ts1-epyy[:,jk]).unsqueeze(1),((ts1-epyy[:,jk])>0).float().unsqueeze(1),(ts1/(epyy[:,jk]+0.0001)).unsqueeze(1)],1).unsqueeze(1))
                    tzs1=zs1[q[:,jk]]
                    ezs.append(torch.cat([zs1.unsqueeze(1),tzs1.unsqueeze(1),torch.cat([((zs1[:,0]>0.5)*(tzs1[:,1]>0.5)).unsqueeze(1),((zs1[:,0])*(tzs1[:,1])).unsqueeze(1)],1).unsqueeze(1)],2))
           
                esp=torch.cat(es,1)
                ezsp=torch.cat(ezs,1)
                score=EN(qf.unsqueeze(1),ey,epy.float(),esp,ezsp)
                yl=[]
                ya=0
                for jk in range(5):
                    yl.append((ql[q[:,jk]]==ql).unsqueeze(1))
                    ya+=(ql[q[:,jk]]==ql).float()

                yl.append((ya==0).unsqueeze(1))
                yl=torch.cat(yl,1)
                yls=yl[:,:5].float().sum(1)
                ylmax=yl.float().argmax(1)
                ylmin=yl[:,:5].float().argmin(1)
             
                score=score[ql!=0]
                yls=yls[ql!=0]
                yl=yl[ql!=0]
             
             
                tkloss=torch.mean(bce(F.sigmoid(score),yl.float()))#+torch.mean(n2loss(score,yl))
                #tkloss=torch.mean(criterion2(score,(yl).float().argmax(1).long())[yls<=1])
                if step%50==0:
                    print('selfex:acc0:',((score.cpu().argmax(1)<5)==(yls>0).cpu()).sum().item(),score.shape[0],labels.unique().shape,la2.unique().shape)
                    print('selfex:acc:',((F.sigmoid(score).cpu().float()>0.5)==(yl.cpu().float())).sum().item(),((score.cpu().float()>0.5)==(yl.cpu().float())).sum().item()/(1+score.shape[1]*score.shape[0]))
                    #print('selfex:accx:',((al)==(yl[:,:-1].cpu().float())).sum().item(),((al)==(yl[:,:-1].cpu().float())).sum().item()/((score.shape[1]-1)*score.shape[0]))
                   
    ########################################################################
                px=px.to(device1)
                py=py.to(device1)
        
                m, n = px.shape[0], py.shape[0]
                distmat = torch.pow(px.float(), 2).sum(dim=1, keepdim=True).expand(m, n) + \
                      torch.pow(py.float(), 2).sum(dim=1, keepdim=True).expand(n, m).t()
                distmat.addmm_(1, -2, px.float(),py.float().t())
                qx,q=distmat.to(device).topk(5,largest=False)
             
                px=px.to(device)
                py=py.to(device)       
       
                sx=[0,1,2,3,4]
                if random.randint(0,1)==1:
                    nq=q.clone()
                    nqx=qx.clone()
                    for ki in range(q.shape[0]):
                        random.shuffle(sx)
                        for jk in range(5):
                            nq[ki,jk]=q[ki,sx[jk]]
                            
                            nqx[ki,jk]=qx[ki,sx[jk]]
               
                    q=nq
                    qx=nqx
                ey=[]
                ep=[]
                es=[]
                ezs=[]
                #epx=px*px
       
                for jk in range(5):
                    ey.append(gf[q[:,jk].unsqueeze(1)])
                    t=py[q[:,jk]]-px
                    ep.append(torch.abs(t).unsqueeze(1))
                epy=torch.cat(ep,1)
                ey=torch.cat(ey,1)
                epyy=torch.sqrt((epy*epy).sum(-1))
            
                for jk in range(5):
                    ts1=s2[q[:,jk]]
                    es.append(torch.cat([(ts1-s1).unsqueeze(1),(ts1/(s1+1)).unsqueeze(1),
                                         epyy[:,jk].unsqueeze(1),(ts1-epyy[:,jk]).unsqueeze(1),((ts1-epyy[:,jk])>0).float().unsqueeze(1),(ts1/(epyy[:,jk]+0.0001)).unsqueeze(1)],1).unsqueeze(1))
                    tzs1=zs2[q[:,jk]]
                    ezs.append(torch.cat([zs1.unsqueeze(1),tzs1.unsqueeze(1),torch.cat([((zs1[:,0]>0.5)*(tzs1[:,1]>0.5)).unsqueeze(1),((zs1[:,0])*(tzs1[:,1])).unsqueeze(1)],1).unsqueeze(1)],2))
           
                esp=torch.cat(es,1)
                ezsp=torch.cat(ezs,1)
                score=EX(qf.unsqueeze(1),ey,epy.float(),esp,ezsp)
                sp=score[:,-2:]
                score=score[:,:-2]
                yl=[]
                ya=0
                for jk in range(5):
                    yl.append((gl[q[:,jk]]==ql).unsqueeze(1))
                    ya+=(gl[q[:,jk]]==ql).float()
                zcs=criterion6(sp,zb.long())
                yl.append((ya==0).unsqueeze(1))
                yl=torch.cat(yl,1)
                yls=yl[:,:5].float().sum(1)
                ylmax=yl.float().argmax(1)
                ylmin=yl[:,:5].float().argmin(1)
                score=score[ql!=0]
                yls=yls[ql!=0]
                yl=yl[ql!=0]
              
                cl=bce(F.sigmoid(score),yl.float())
                  
                

                tkloss+=torch.mean(cl)*5+5*torch.mean(zcs)#+torch.mean(bl)#+torch.mean(n2loss(score,yl))
                zz=sp.argmax(1)
                
                #tkloss=torch.mean(criterion2(score,(yl).float().argmax(1).long())[yls<=1])
                if step%50==0 or lx2>200:
                    print('ex:acc0:',((score.cpu().argmax(1)<5)==(yls>0).cpu()).sum().item(),score.shape[0],labels.unique().shape,la2.unique().shape)
                    print('ex:acc:',((F.sigmoid(score).cpu().float()>0.5)==(yl.cpu().float())).sum().item(),((score.cpu().float()>0.5)==(yl.cpu().float())).sum().item()/(1+score.shape[1]*score.shape[0]))
                    #print('ex:accx:',(al==yl[:,:-1].cpu().float()).sum().item(),(al==(yl[:,:-1].cpu().float())).sum().item()/((score.shape[1]-1)*score.shape[0]))

                    print('seg:',lx,lx1,lx2,len(lx3))
                    print('zacc:',((zb==zz).sum()/zb.shape[0]).item(),zb.sum().item(),zz.sum().item(),(zb*zz).sum().item())
                  
                tcloss=0
       
                if tkloss.item()==tkloss.item():
                   
                    kloss.append((tcloss+tkloss).unsqueeze(-1)) 
            lb1=ql
            lb2=gl
            til=list((lb1).cpu().numpy())
            z1=pd.value_counts(til).to_dict()
            til=list((lb2).cpu().numpy())
            
            z2=pd.value_counts(til).to_dict()
            cb1=lb1.cpu().numpy().astype(int)
            cb2=lb2.cpu().numpy().astype(int)
        

##########same########################################################
            # kd1={}
            # kd2={}
            # for i in range(len(p1)):
            #     kd1[(lb1[i])]=kd1.get((lb1[i]),[])+[i]
            # for i in range(len(p2)):
            #       kd2[(lb2[i])]=kd2.get((lb2[i]),[])+[i] 
            # t2=time.time()
            # #sb1=[ff1[0,0,i[0],i[1]] for i in p1]
            # #sb2=[ff2[0,0,i[0],i[1]] for i in p2]
            # #wb1=[lb1.count(i) for i in lb1]
            # #wb2=[lb2.count(i) for i in lb2]
            # tlist1=[]
            # ylist1=[]
            # plist1=[]
            # kfl=[]
            # nid=0
            # for i in kd1:
            #     ll=kd1[i]
            #     kfl.append(i)
            #     kfl.append(i)
            #     if len(ll)>1:
            #         x,y,z=p1[ll[0]]
            #         z=z//10
            #         tlist1.append(fo[:,:,x,y,z])
            #         x,y,z=p1[ll[1]]
            #         z=z//10
            #         tlist1.append(fo[:,:,x,y,z])
            #         plist1.append(p1[ll[0]])
            #         plist1.append(p1[ll[1]])
            #         ylist1.append(nid)
            #         ylist1.append(nid)
            #         nid+=1
            #     else:
            #         x,y,z=p1[ll[0]]
            #         z=z//10
            #         tlist1.append(fo[:,:,x,y,z])
            #         x,y,z=p1[ll[0]]
            #         z=z//10
            #         xx=random.choice([-1,1])
            #         yy=random.choice([-1,1])
            #         tlist1.append(fo[:,:,(x+xx).clip(0,255),(y+yy).clip(0,255),z])
            #         ylist1.append(nid)
            #         ylist1.append(nid)
            #         plist1.append(p1[ll[0]])
            #         plist1.append(p1[ll[0]])
            #         nid+=1              
            # # #for i in lb1:
            # # tlist2=[]
            # # ylist2=[]
            # # plist2=[]
            # # nid=0
            # # for i in kd2:
            # #     ll=kd2[i]
            # #     if len(ll)>1:
            # #         x,y,z=p2[ll[0]]
            # #         z=z//10
            # #         tlist2.append(fo2[:,:,x,y,z])
            # #         x,y,z=p2[ll[1]]
            # #         z=z//10
            # #         tlist2.append(fo2[:,:,x,y,z])
            # #         ylist2.append(nid)
            # #         ylist2.append(nid)
            # #         plist2.append(p2[ll[0]])
            # #         plist2.append(p2[ll[1]])
            # #         nid+=1
            # #     else:
            # #         x,y,z=p2[ll[0]]
            # #         z=z//10
            # #         tlist2.append(fo2[:,:,x,y,z])
            # #         x,y,z=p2[ll[0]]
            # #         z=z//10
            # #         xx=random.choice([-1,1])
            # #         yy=random.choice([-1,1])
            # #         tlist2.append(fo2[:,:,(x+xx).clip(0,255),(y+yy).clip(0,255),z])
            # #         ylist2.append(nid)
            # #         ylist2.append(nid)
            # #         plist2.append(p2[ll[0]])
            # #         plist2.append(p2[ll[0]])
            # #         nid+=1
         
            # t3=time.time()
            # if len(tlist1)>2 :
            #     feature1=torch.cat(tlist1)
            #     #feature2=torch.cat(tlist2)
            #     rloss3+=Tloss(feature1,torch.from_numpy(np.array(ylist1)))[0] 
                #rloss3+=Tloss(feature2,torch.from_numpy(np.array(ylist2)))[0] 
               
#############################################################

            tt=[]
            lt=[]
            for ix in range(len(lb1)):
                if  cb1[ix]!=0 and z1.get(cb1[ix],0)>=1 and z2.get(cb1[ix],0)>=2 and cb1[ix] not in lt:
                    tt.append(ix)
                    lt.append(cb1[ix])
            tt=np.array(tt)
            f1m=f1[tt]
            p1m=p1[tt]
            lb1m=lb1[tt]   
########################################################'


            
            tt=[]
            lt=[]
            for ix in range(len(lb1)):
                if  cb1[ix]!=0 and z1.get(cb1[ix],0)>=1 and z2.get(cb1[ix],0)>=1 and cb1[ix] not in lt:
                    tt.append(ix)
                    lt.append(cb1[ix])
        
            tt=np.array(tt)
            f1=f1[tt]
            p1=p1[tt]
            lb1=lb1[tt]
            cb1=lb2.cpu().numpy()
#############################################################

            tt=[]
            lt=[]
            tt1=[]
            for ix in range(len(lb2)):
                if  cb1[ix]!=0 and z1.get(cb1[ix],0)>=1 and z2.get(cb1[ix],0)>=2:
                    if cb1[ix] not in lt:
                        tt.append(ix)
                        lt.append(cb1[ix])
                    elif lt.count(cb1[ix])<2:
                        tt1.append(ix)
                        lt.append(cb1[ix])
            tt=np.array(tt)
            tt1=np.array(tt1)
            f2m=f2[tt]
            p2m=p2[tt]
            lb2m=lb2[tt1] 
            f2m2=f2[tt1]
            p2m2=p2[tt1]
            lb2m2=lb2[tt1] 
########################################################'
            tt=[]
            lt=[]
            for ix in range(len(lb2)):
                if  cb1[ix]!=0 and z1.get(cb1[ix],0)>=1 and z2.get(cb1[ix],0)>=1 and cb1[ix] not in lt:
                    tt.append(ix)
                    lt.append(cb1[ix])
            tt=np.array(tt)
            f2=f2[tt]
            p2=p2[tt]
            lb2=lb2[tt]    
      
  
            
                #ZL=[[[f1,f2],[lb1,lb2]],[[f1m,f2m],[lb1m,lb2m]],[[f1m,f2m2],[lb1m,lb2m2]],[[f1m,torch.cat([f2m,f2m2],0)],[lb1m,torch.cat([lb2m,lb2m2],0)]]]#,[[bf,bf2],[blb,blb2]]]
            ZL=[[[f1,f2],[lb1,lb2]],[[f2m,f2m2],[lb2m,lb2m2]],[[f1m,torch.cat([f2m,f2m2],0)],[lb1m,torch.cat([lb2m,lb2m2],0)]]]#,[[bf,bf2],[blb,blb2]]]

            for inx in ZL:
              
                Ff,L=inx
                Ff=torch.cat(Ff,0)
                L1=torch.cat(L,0)
                #print(1,Ff.shape,L1.shape)
                if Ff.shape[0]>4: 
                    lt=Tloss(Ff,L1)[0] 
                    if lt.item()==lt.item():
                        rloss3+=lt


######################################################
                              
        loss=Uloss+sloss+sloss1
        if rloss3==rloss3 and rloss3>0:
            loss+=rloss3*2   
            #print(epoch,rloss3.item())
        if len(kloss)==0:
            kloss=0
        else:
            kloss=torch.mean(torch.cat(kloss)) 
            kk+=kloss.item()
        if kloss==kloss:
            loss+=kloss*2 
        tr_loss += loss.item()
        
 
        loss.backward()

        oU.step()
        oU.zero_grad()
        oEX.step()
        oEX.zero_grad() 
        oEN.step()
        oEN.zero_grad()     
        #time.sleep(0.001)
        #sU.step(loss)
        #sEX.step(loss)
    epoch_loss = tr_loss / len(data_loader_train)
    print('Training Loss: {:.4f},KK:{:.4f}'.format(epoch_loss,kk/ len(data_loader_train)))


    
    if epoch>=1 :
        win2s=[[128,256,0],[256,0,0]]
        pab=args.patch_size_xy
        pcc=args.patch_size_z
        tk0 = tqdm(data_loader_val, desc="Iteration")
       
        U.eval()
        EN.eval()
        EX.eval()
   
        val_loss=0
        kk=0
        se=0
        iou=0
        n=0
        for step, batch in enumerate(tk0):
            #break
            for iix in range(2):
                win2=win2s[iix]
                inputs = batch["image"].unsqueeze(1)
               
                labels = batch["labels"]
                in2=batch["im2"].unsqueeze(1)
                la2=batch["la2"]
                inputs = inputs[:,:,win2[0]:win2[0]+pab,win2[1]:win2[1]+pab,win2[2]:win2[2]+pcc]#.to(device, dtype=torch.float)
                labels = labels[:,win2[0]:win2[0]+pab,win2[1]:win2[1]+pab,win2[2]:win2[2]+pcc].to(device, dtype=torch.float)
                in2=in2[:,:,win2[0]:win2[0]+pab,win2[1]:win2[1]+pab,win2[2]:win2[2]+pcc]#.to(device, dtype=torch.float)
                la2=la2[:,win2[0]:win2[0]+pab,win2[1]:win2[1]+pab,win2[2]:win2[2]+pcc].to(device, dtype=torch.float)
        
                p1=batch["p1"].unsqueeze(1)[:,:,win2[0]:win2[0]+pab,win2[1]:win2[1]+pab,win2[2]:win2[2]+pcc].to(device, dtype=torch.float)
                p2=batch["p2"].unsqueeze(1)[:,:,win2[0]:win2[0]+pab,win2[1]:win2[1]+pab,win2[2]:win2[2]+pcc].to(device, dtype=torch.float)       
                s1=batch["s1"][:,win2[0]:win2[0]+pab,win2[1]:win2[1]+pab,win2[2]:win2[2]+pcc].to(device, dtype=torch.float)   
                s2=batch["s2"][:,win2[0]:win2[0]+pab,win2[1]:win2[1]+pab,win2[2]:win2[2]+pcc].to(device, dtype=torch.float)   
                lsize1=batch["size1"][:,win2[0]:win2[0]+pab,win2[1]:win2[1]+pab,win2[2]:win2[2]+pcc]#.to(device, dtype=torch.float)   
                lx2=torch.unique(p1[0,0]).shape[0]   
                         
                #s3=batch["s3"][:,win2[0]:win2[0]+pab,win2[1]:win2[1]+pab,win2[2]:win2[2]+32].to(device, dtype=torch.float)   
                #s4=batch["s4"][:,win2[0]:win2[0]+pab,win2[1]:win2[1]+pab,win2[2]:win2[2]+32].to(device, dtype=torch.float)    
                in3= batch["img3"].unsqueeze(1)[:,:,win2[0]:win2[0]+pab,win2[1]:win2[1]+pab,win2[2]:win2[2]+pcc]#.to(device, dtype=torch.float)
                #in4= batch["img4"].unsqueeze(1)[:,:,win2[0]:win2[0]+pab,win2[1]:win2[1]+pab,win2[2]:win2[2]+32].to(device, dtype=torch.float)
                # pl1=batch["p1"].unsqueeze(1)[:,:,win2[0]:win2[0]+pab,win2[1]:win2[1]+pab,win2[2]:win2[2]+pcc].to(device, dtype=torch.float)
                # pl2=batch["p2"].unsqueeze(1)[:,:,win2[0]:win2[0]+pab,win2[1]:win2[1]+pab,win2[2]:win2[2]+pcc].to(device, dtype=torch.float)       
     
                with torch.no_grad():
    
               
                    Uout,uo,fo,zs1,size1 =  U(PA(torch.cat([inputs,in2],1)).to(device, dtype=torch.float))
                    Uout2,uo2,fo2,zs2,size2 =  U(PA(torch.cat([in2,in3],1)).to(device, dtype=torch.float))
                    # uu=Uout.argmax(1).squeeze().cpu().max(-1)[0].numpy().astype(int)*128
                    # iu=inputs[0,0].max(-1)[0].cpu().numpy()
                    # cv2.imwrite('./ctctrain/'+str(step)+'-'+str(iix)+'.png',np.concatenate([uu,iu]))
                    # po=(F.sigmoid(zs1[:,0])>0.5)*100+(F.sigmoid(zs1[:,1])>0.5)*200
                    # po=po.unsqueeze(1)
                    p1[:,0]=F.conv3d((p1[:,0]>0).float().unsqueeze(1), DK1, padding=(tm+1,tm+1,0))
                    #p1[:,1]=F.conv3d((p1[:,1]>0).float().unsqueeze(1), DK1, padding=(tm+1,tm+1,0))  
                    p2[:,0]=F.conv3d((p2[:,0]>0).float().unsqueeze(1), DK1, padding=(tm+1,tm+1,0))
                    #p2[:,1]=F.conv3d((p2[:,1]>0).float().unsqueeze(1), DK1, padding=(tm+1,tm+1,0)) 
                    for i in range(3):
                        p1=ud(p1)
                        p2=ud(p2)  
                    wp1=labels>0
                    zp1=wp1*(p1[:,0])
                    wp2=la2>0
                    zp2=wp2*(p2[:,0])
    
     
                    #bclc=criterion(zs1,(s1[:,0]).long())#+criterion3(zs1*Uout,(s1>0).unsqueeze(0).long())
                    #bclc2=criterion(zs3,(s2[:,0]).long())#+criterion3(zs2*Uout2,(s2>0).unsqueeze(0).long())
        
            
                    #nloss3=torch.mean(criterion(uo,(zp1).long())[labels>0])
                    #nloss4=torch.mean(criterion(uo2,(zp2).long())[la2>0])           
            
                    #sloss=torch.mean(lclc)+torch.mean(lclc2)#+torch.mean(bclc2)+torch.mean(bclc)#+torch.mean(nlcx)+torch.mean(nlcx2)#+(torch.mean(nclcp)+torch.mean(nclcp2)+torch.mean(nlcxp)+torch.mean(nlcxp2))#+torch.mean(nlc+nlc2)+l1loss(nlc,lc)+l1loss(nlc2,lc2)
                      
                    
                    rloss3=0
                    kloss=[]
                    Uf=fo.transpose(0,1)
                    Uf2=fo2.transpose(0,1)
                    fo=[]
                    fo2=[]
                    #uc=(uo.argmax(1)==1)*(Uout.argmax(1)==1)*(size1[:,0]>1)
        
                    
                    #uu=(kfs2(uo1).cuda().max(1)[0]==3)*(ku==3).sum(1)*(kfs2(uo[:,3]).cuda().max(1)[0]==uo[:,3])
                    #uu2=(kfs2(uo1).cuda().max(1)[0]==2)*(ku==2).sum(1)*(kfs2(uo[:,2]).cuda().max(1)[0]==uo[:,2])
                    uc=((kflb(uo[:,:].max(1)[0]).cuda().max(1)[0]==uo[:,4]))*(Uout.argmax(1)==1)#*(size1[:,0]>1)
                    #uc=((uu2>6)+(uu>4)+uc1)*(Uout.argmax(1)==1)#*(size1[:,0]>1)
                    u,x,y,z=torch.where(uc.to(device, dtype=torch.float))
                    lx=torch.unique(labels).shape[0]
                    lx1=torch.unique(labels*uc).shape[0]
                    lx3=list(torch.unique(labels*uc).cpu().numpy())
                    lx3=[i for i in lx3 if i in np.unique(ee1)]
                    f1=Uf[:,u,x,y,z].transpose(0,1)
                    zb=s1[u,x,y,z]==1
                    s1=size1[u,0,x,y,z]
                    zs1=F.sigmoid(zs1[u,:,x,y,z])
                   
                    #s11=size1[u,0,x,y,z+1]
                    #s12=size1[u,0,x,y,z-1]
                    #s1=torch.cat([s1.unsqueeze(1),s11.unsqueeze(1),s12.unsqueeze(1)],1).max(1)[0]
                    
                    lb1=labels[u,x,y,z]
                    uo1=uo2.argmax(1)
                   
              
                    #uu=(kfs2(uo1).cuda().max(1)[0]==3)*(ku==3).sum(1)*(kfs2(uo2[:,3]).cuda().max(1)[0]==uo2[:,3])
                    #uu2=(kfs2(uo1).cuda().max(1)[0]==2)*(ku==2).sum(1)*(kfs2(uo2[:,2]).cuda().max(1)[0]==uo2[:,2])
                    uc=((kflb(uo2[:,:].max(1)[0]).cuda().max(1)[0]==uo2[:,4]))*(Uout2.argmax(1)==1)#*(size1[:,0]>1)
                    #uc=((uu2>6)+(uu>4)+uc1)*(Uout2.argmax(1)==1)#*(size1[:,0]>1)
                    
                    nloss1=crloss(uo,(zp1).long())#*w1
                    nloss2=crloss((uo2),(zp2).long())#*w2
                    # nloss3=torch.mean(nloss1[labels>0])
                    # nloss4=torch.mean(nloss2[la2>0])
                    Uloss=torch.mean(nloss1)+torch.mean(nloss2)#+nloss3+nloss4#+torch.mean(nclc*w1)+torch.mean(nclc2*w2)
                    nloss1=[]
                    nloss2=[]
                    nloss3=[]
                    nloss4=[]
                #po=Uout.clone()
                #po2=Uout2.clone()
                
               
               
                # labelsx=F.conv3d((p1>1)[:,0].float().unsqueeze(1), DK, padding=(tm,tm,4)).squeeze()
                # #po=F.conv3d(po.unsqueeze(0).float(), DK, padding=(tm,tm,4)).squeeze()
                # po1=F.conv3d(uo.argmax(1)[0].float().unsqueeze(0).unsqueeze(0), DK, padding=(tm,tm,4)).squeeze()
                
       
               
           
                #uc=F.avg_pool3d(uc.unsqueeze(0).float(),[win,win,win])>0.9
                #uc=uc[0]
                #Uf=F.avg_pool3d(Uf.float(),[win,win,win])
                #ua=kf(uc.float()).sum(1).cuda()*uc
                #ub=((kf(ua.float()).cuda()-ua.unsqueeze(1))>0).sum(1).cuda()*uc
                #uc=uc*(ub<ko)
                #uc=(kf(uc.float()).sum(1)>ko).float().cuda()*uc
                #u,x,y,z=torch.where(uc.to(device, dtype=torch.float))
                if u.shape[0]>0 :
                   
     
                    #uc=F.avg_pool3d(uc.unsqueeze(0).float(),[win,win,win])>0.9
                    #uc=uc[0]
                    #Uf2=F.avg_pool3d(Uf2.float(),[win,win,win])
                    #uc=(kf(uc.float()).sum(1)>ko).float().cuda()*uc
                    #u,x,y,z=torch.where(uc)            
                    u2,x2,y2,z2=torch.where(uc.to(device, dtype=torch.float))  
                    f2=Uf2[:,u2,x2,y2,z2].transpose(0,1)
                    s2=size2[u2,0,x2,y2,z2]
                    zs2=F.sigmoid(zs2[u2,:,x2,y2,z2])
                    lb2=la2[u2,x2,y2,z2]
                    #s21=size1[u2,0,x2,y2,z2+1]
                    #s22=size1[u2,0,x2,y2,z2-1]
                    #s2=torch.cat([s2.unsqueeze(1),s21.unsqueeze(1),s22.unsqueeze(1)],1).max(1)[0]  
                    p1=torch.cat([u.unsqueeze(1),x.unsqueeze(1),y.unsqueeze(1),z.unsqueeze(1)],1)
                    p2=torch.cat([u2.unsqueeze(1),x2.unsqueeze(1),y2.unsqueeze(1),z2.unsqueeze(1)],1)
                 
                           
                    px=p1.float()
                    py=p2.float()
                    px[:,0]=px[:,0]*100
                    py[:,0]=py[:,0]*100
                    m, n = px.shape[0], py.shape[0]
                    
                
                       
                    p1=p1[:,1:]
                    p2=p2[:,1:]
                    p1[:,2]=p1[:,2]*ZS
                    p2[:,2]=p2[:,2]*ZS          
                    px=p1.float()
                    py=p2.float()
                
                   
                    qf,ql,px,gf,gl,py=f1,lb1,p1,f2,lb2,p2
                    if  qf.shape[0]>5 and gf.shape[0]>5:
                
                        ###########################################################
                       
        ##########################################################
                        m, n = px.shape[0], px.shape[0]
                        distmat = torch.pow(px.float(), 2).sum(dim=1, keepdim=True).expand(m, n) + \
                              torch.pow(px.float(), 2).sum(dim=1, keepdim=True).expand(n, m).t()
                        distmat.addmm_(1, -2, px.float(),px.float().t())
                        qx,q=distmat.topk(6,largest=False)
                        qx=qx[:,1:]
                        q=q[:,1:]
                           
                           
                           
                    
                        ey=[]
                        ep=[]
                        es=[]
                        ezs=[]
                        #epx=px*px
               
                        for jk in range(5):
                            ey.append(qf[q[:,jk].unsqueeze(1)])
                            t=px[q[:,jk]]-px
                            ep.append(torch.abs(t).unsqueeze(1))
                        epy=torch.cat(ep,1)
                        ey=torch.cat(ey,1)
                        epyy=torch.sqrt((epy*epy).sum(-1))
                    
                        for jk in range(5):
                            ts1=s1[q[:,jk]]
                            es.append(torch.cat([(ts1-s1).unsqueeze(1),(ts1/(s1+1)).unsqueeze(1),
                                                 epyy[:,jk].unsqueeze(1),(ts1-epyy[:,jk]).unsqueeze(1),((ts1-epyy[:,jk])>0).float().unsqueeze(1),(ts1/(epyy[:,jk]+0.0001)).unsqueeze(1)],1).unsqueeze(1))
                            tzs1=zs1[q[:,jk]]
                            ezs.append(torch.cat([zs1.unsqueeze(1),tzs1.unsqueeze(1),torch.cat([((zs1[:,0]>0.5)*(tzs1[:,1]>0.5)).unsqueeze(1),((zs1[:,0])*(tzs1[:,1])).unsqueeze(1)],1).unsqueeze(1)],2))
                   
                        esp=torch.cat(es,1)
                        ezsp=torch.cat(ezs,1)
                        with torch.no_grad():
                            score=EN(qf.unsqueeze(1),ey,epy.float(),esp,ezsp)
                        yl=[]
                        ya=0
                        for jk in range(5):
                            yl.append((ql[q[:,jk]]==ql).unsqueeze(1))
                            ya+=(ql[q[:,jk]]==ql).float()
                           
                        yl.append((ya==0).unsqueeze(1))
                        yl=torch.cat(yl,1)
                        yls=yl[:,:5].float().sum(1)
                        ylmax=yl.float().argmax(1)
                        ylmin=yl[:,:5].float().argmin(1)
                        score=score[ql!=0]
                        yls=yls[ql!=0]
                        yl=yl[ql!=0]
                   
                
                        tkloss=torch.mean(bce(F.sigmoid(score),yl.float()))
                        #tkloss=torch.mean(criterion2(score,(yl).float().argmax(1).long())[yls<=1])
                        if step%10==0 and score.shape[0]>0:
                            print('selfex:acc0:',((score.cpu().argmax(1)<5)==(yls>0).cpu()).sum().item(),score.shape[0],labels.unique().shape,la2.unique().shape)
                            print('selfex:acc:',((F.sigmoid(score).cpu().float()>0.5)==(yl.cpu().float())).sum().item(),((score.cpu().float()>0.5)==(yl.cpu().float())).sum().item()/(1+score.shape[1]*score.shape[0]))
                           # print('selfex:accx:',((al)==(yl[:,:-1].cpu().float())).sum().item(),((al)==(yl[:,:-1].cpu().float())).sum().item()/((score.shape[1]-1)*score.shape[0]))
                           
        #################################################
            
                                   
                              
               
                                
                        m, n = qf.shape[0], gf.shape[0]
                        distmat = torch.pow(px.float(), 2).sum(dim=1, keepdim=True).expand(m, n) + \
                              torch.pow(py.float(), 2).sum(dim=1, keepdim=True).expand(n, m).t()
                        distmat.addmm_(1, -2, px.float(),py.float().t())
                        qx,q=distmat.topk(5,largest=False)
               
               
                      
                        ey=[]
                        ep=[]
                        es=[]
                        ezs=[]
                        #epx=px*px
               
                        for jk in range(5):
                            ey.append(gf[q[:,jk].unsqueeze(1)])
                            t=py[q[:,jk]]-px
                            ep.append(torch.abs(t).unsqueeze(1))
                        epy=torch.cat(ep,1)
                        ey=torch.cat(ey,1)
                        epyy=torch.sqrt((epy*epy).sum(-1))
                    
                        for jk in range(5):
                            ts1=s2[q[:,jk]]
                            es.append(torch.cat([(ts1-s1).unsqueeze(1),(ts1/(s1+1)).unsqueeze(1),
                                                 epyy[:,jk].unsqueeze(1),(ts1-epyy[:,jk]).unsqueeze(1),((ts1-epyy[:,jk])>0).float().unsqueeze(1),(ts1/(epyy[:,jk]+0.0001)).unsqueeze(1)],1).unsqueeze(1))
                            tzs1=zs2[q[:,jk]]
                            ezs.append(torch.cat([zs1.unsqueeze(1),tzs1.unsqueeze(1),torch.cat([((zs1[:,0]>0.5)*(tzs1[:,1]>0.5)).unsqueeze(1),((zs1[:,0])*(tzs1[:,1])).unsqueeze(1)],1).unsqueeze(1)],2))
                   
                        esp=torch.cat(es,1)
                        ezsp=torch.cat(ezs,1)
                        with torch.no_grad():
                            score=EX(qf.unsqueeze(1),ey,epy.float(),esp,ezsp)
                        sp=score[:,-2:]
                        score=score[:,:-2]
                        yl=[]
                        ya=0
                        for jk in range(5):
                            yl.append((gl[q[:,jk]]==ql).unsqueeze(1))
                            ya+=(gl[q[:,jk]]==ql).float()
                        zcs=criterion(sp,zb.long())
                        yl.append((ya==0).unsqueeze(1))
                        yl=torch.cat(yl,1)
                        yls=yl[:,:5].float().sum(1)
                        ylmax=yl.float().argmax(1)
                        ylmin=yl[:,:5].float().argmin(1)
                        score=score[ql!=0]
                        yls=yls[ql!=0]
                        yl=yl[ql!=0]
                        #ws=ws[ql.cpu()!=0]
                        cl=bce(F.sigmoid(score),yl.float())
                        #cl[ws>1]=cl[ws>1]*2
                     
                  
            
                        tkloss+=torch.mean(cl)+torch.mean(zcs)#+torch.mean(n2loss(score,yl))
                        zz=sp.argmax(1)
                        
                        #tkloss=torch.mean(criterion2(score,(yl).float().argmax(1).long())[yls<=1])
                        if step%10==0 and score.shape[0]>0 or lx2>200:
                            print('ex:acc0:',((score.cpu().argmax(1)<5)==(yls>0).cpu()).sum().item(),score.shape[0],labels.unique().shape,la2.unique().shape)
                            print('ex:acc:',((F.sigmoid(score).cpu().float()>0.5)==(yl.cpu().float())).sum().item(),((score.cpu().float()>0.5)==(yl.cpu().float())).sum().item()/(1+score.shape[1]*score.shape[0]))
                            #print('ex:accx:',(al==yl[:,:-1].cpu().float()).sum().item(),(al==(yl[:,:-1].cpu().float())).sum().item()/((score.shape[1]-1)*score.shape[0]))
        
                            print('seg:',lx,lx1,lx2,len(lx3))
                            print('zacc:',((zb==zz).sum()/zb.shape[0]).item(),zb.sum().item(),zz.sum().item(),(zb*zz).sum().item())
                          
                        
                        tcloss=0
                        
               
                        kloss.append((tcloss+tkloss).unsqueeze(-1))       
                    lb1=ql
                    lb2=gl
                    til=list((lb1).cpu().numpy())
                    z1=pd.value_counts(til).to_dict()
                    til=list((lb2).cpu().numpy())
                    
                    z2=pd.value_counts(til).to_dict()
                    cb1=lb1.cpu().numpy().astype(int)
                    cb2=lb2.cpu().numpy().astype(int)
        
        
        #############################################################
            
                    tt=[]
                    tt1=[]
                    lt=[]
                    for ix in range(len(lb1)):
                        if  cb1[ix]!=0 and cb1[ix] not in lt:
                            ta=z2.get(cb1[ix],0)
                     
                            
                            if ta>=1:
                                tt.append(ix)
                            if ta>=2:
                                tt1.append(ix)
                            lt.append(cb1[ix])
                    tt1=np.array(tt1)
                    f1m=f1[tt1]
                    p1m=p1[tt1]
                    lb1m=lb1[tt1]   
                    tt=np.array(tt)
                    f1=f1[tt]
                    p1=p1[tt]
                    lb1=lb1[tt]
                    
        ########################################################'
        
                    
        
                    tt=[]
                    lt=[]
                    tt1=[]
                    for ix in range(len(lb2)):
                        if  cb2[ix]!=0 and z1.get(cb2[ix],0)>0:
                            ta=z2.get(cb2[ix],0)
                            kt=lt.count(cb2[ix])
                            if kt==0 and ta>=1:
                                tt.append(ix)
                                lt.append(cb2[ix])
                            if kt==1 and ta>=2:
                                tt1.append(ix)
                                lt.append(cb2[ix])
                    tt1=np.array(tt1)
                    f2m=f2[tt1]
                    p2m=p2[tt1]
                    lb2m=lb2[tt1]   
                    tt=np.array(tt)
                    f2=f2[tt]
                    p2=p2[tt]
                    lb2=lb2[tt]
        ########################################################'
                    tt=[]
                    #print(f1.shape,f2.shape,lb1.shape,lb2.shape)
                    #print(f1m.shape,f2m.shape,lb1m.shape,lb2m.shape)
                    #print(lb1,lb1m)
                    #print(lb2,lb2m)
                        #ZL=[[[f1,f2],[lb1,lb2]],[[f1m,f2m],[lb1m,lb2m]],[[f1m,f2m2],[lb1m,lb2m2]],[[f1m,torch.cat([f2m,f2m2],0)],[lb1m,torch.cat([lb2m,lb2m2],0)]]]#,[[bf,bf2],[blb,blb2]]]
                    ZL=[[[f1,f2],[lb1,lb2]],[[f1m,f2m],[lb1m,lb2m]]]#,[[bf,bf2],[blb,blb2]]]
        
                    for inx in ZL:
                      
                        Ff,L=inx
                        Ff=torch.cat(Ff,0)
                        L1=torch.cat(L,0)
                        #print(1,Ff.shape,L1.shape)
                        if Ff.shape[0]>4: 
                            lt=Tloss(Ff,L1)[0] 
                            if lt.item()==lt.item():
                                rloss3+=lt
                   
    
                loss=Uloss
                if rloss3==rloss3:
                    loss+=rloss3*3      
                if len(kloss)==0:
                    kloss=0
                else:
                    kloss=torch.mean(torch.cat(kloss)) 
                    kk+=kloss.item()
                if kloss==kloss:
                    loss+=kloss*3       
    
                #outputs=torch.sigmoid(outputs).long()
        
                val_loss += loss.item()
             
                se += Uloss.item()
                
                n=n+1
        sU.step(se)
        sEX.step(kk) 
        sEN.step(kk)  
       
        epoch_loss = val_loss / (3*len(data_loader_val))
        #epoch_iou = np.mean(np.array(iou.tolist()) / len(x_train))
        print('val Loss: {:.4f},uloss:{:.4f},kloss:{:.4f}'.format(np.mean(epoch_loss),np.mean(se/ len(data_loader_val)),np.mean(kk/ len(data_loader_val))))
        if  epoch >2:
            va=epoch_loss
            torch.save(EX.state_dict(),args.out_dir+'/EX+sc2n-{:.1f}.pth'.format(epoch))
            torch.save(EN.state_dict(),args.out_dir+'/EN+sc2n-{:.1f}.pth'.format(epoch))
            torch.save(U.state_dict(),args.out_dir+'/U-ext+sc2n-{:.1f}.pth'.format(epoch))
            #torch.save(U2,'../v/model/U+-{:.1f}-{:.4f}.pth'.format(epoch,va))

  
