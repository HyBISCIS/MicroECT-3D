# Copyright (c) 2023, HyBISCIS Team (Brown University, Boston University)
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. 

import torch
from torch import nn
from torch.nn import functional as F
from torchsummary import summary


class Generator3d_2(nn.Module):

    def __init__(self, input_dim, output_image_size=64, num_channels=1, hidden_activation='ReLU', head_activation='Tanh'):
        super(Generator3d_2, self).__init__()
        self.input_dim = input_dim 

        head_activ = getattr(nn, head_activation)()
        hidden_activ = getattr(nn, hidden_activation)(True)

        # Decoder 
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose3d(input_dim, output_image_size * 8, kernel_size=(3,3,6), stride=(1, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(output_image_size * 8),
            nn.ReLU(True),

            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose3d(in_channels=output_image_size * 8, out_channels=output_image_size * 4, kernel_size=(3, 4, 4), stride=(1,2,2), padding=(1,1,1)),
            nn.BatchNorm3d(output_image_size * 4),
            nn.ReLU(True),

            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose3d(output_image_size * 4, output_image_size * 2, kernel_size=(3,4,5), stride=(1,2,2), padding=(1, 1, 1)),
            nn.BatchNorm3d(output_image_size * 2),
            nn.ReLU(True),

            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose3d(output_image_size * 2, output_image_size, kernel_size=(3, 5, 4), stride=(1,2,2), padding=(1,1,1), bias=False),
            nn.BatchNorm3d(output_image_size),
            nn.ReLU(True),

            nn.ConvTranspose3d(output_image_size, int(output_image_size/2), kernel_size=(3, 4, 4), stride=(1,2,2), padding=(1,1,1), bias=False),
            nn.BatchNorm3d(int(output_image_size/2)),
            nn.ReLU(True),

            # state size. (ngf) x 32 x 32
            nn.ConvTranspose3d(int(output_image_size/2), num_channels, kernel_size=(3,4,4), stride=(1,2,2), padding=(1,1,1), bias=False),
            head_activ
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input) 

class Generator3d(nn.Module):

    def __init__(self, input_dim, output_image_size=64, num_channels=1, hidden_activation='ReLU', head_activation='Tanh'):
        super(Generator3d, self).__init__()
        self.input_dim = input_dim 

        head_activ = getattr(nn, head_activation)()
        hidden_activ = getattr(nn, hidden_activation)(True)

        # Decoder 
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose3d(input_dim, output_image_size * 8, kernel_size=(1,3,6), stride=1, padding=0),
            nn.BatchNorm3d(output_image_size * 8),
            nn.ReLU(True),

            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose3d(in_channels=output_image_size * 8, out_channels=output_image_size * 4, kernel_size=(1, 4, 4), stride=(1,2,2), padding=(0,1,1)),
            nn.BatchNorm3d(output_image_size * 4),
            nn.ReLU(True),

            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose3d(output_image_size * 4, output_image_size * 2, kernel_size=(1,4,5), stride=(1,2,2), padding=(0, 1, 1)),
            nn.BatchNorm3d(output_image_size * 2),
            nn.ReLU(True),

            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose3d(output_image_size * 2, output_image_size, kernel_size=(1, 5, 4), stride=(1,2,2), padding=(0,1,1), bias=False),
            nn.BatchNorm3d(output_image_size),
            nn.ReLU(True),

            nn.ConvTranspose3d(output_image_size, int(output_image_size/2), kernel_size=(1, 4, 4), stride=(1,2,2), padding=(0,1,1), bias=False),
            nn.BatchNorm3d(int(output_image_size/2)),
            nn.ReLU(True),

            # state size. (ngf) x 32 x 32
            nn.ConvTranspose3d(int(output_image_size/2), num_channels, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1), bias=False),
            head_activ
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input) 


class ResidualGenerator3d(nn.Module):

    def __init__(self, input_dim, output_image_size=64, num_channels=1, hidden_activation='ReLU', head_activation='Tanh'):
    
        super(ResidualGenerator3d, self).__init__()
        self.input_dim = input_dim 

        head_activ = getattr(nn, head_activation)()
        hidden_activ = getattr(nn, hidden_activation)(True)
    
        self.main = nn.Sequential(
            ResidualBlock3d(in_channels=input_dim, out_channels=output_image_size * 8, kernel_size=(1, 3, 6), stride=1, padding=0, output_padding=0),
          
            ResidualBlock3d(in_channels=output_image_size * 8, out_channels=output_image_size * 4, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1), output_padding=(0, 1, 1)),
            ResidualBlock3d(in_channels=output_image_size * 4, out_channels=output_image_size * 2, kernel_size=(1, 4, 5), stride=(1, 2, 2), padding=(0, 1, 1), output_padding=(0, 1, 1),  pad=True),
        
            ResidualBlock3d(in_channels=output_image_size * 2, out_channels=output_image_size, kernel_size=(1, 5, 4), stride=(1, 2, 2), padding=(0, 1, 1), bias=True, output_padding=(0, 1, 1), pad=True),
            ResidualBlock3d(in_channels=output_image_size , out_channels=output_image_size // 2, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1), bias=True, output_padding=(0, 1, 1)),
            
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose3d(int(output_image_size/2), num_channels, kernel_size=(1,4,4), stride=(1,2,2), padding=(0, 1, 1), bias=True),
            head_activ
        ) 
    

    def forward(self, input):
        return self.main(input) 


class ResidualGenerator3d_2(nn.Module):

    def __init__(self, input_dim, output_image_size=64, num_channels=1, hidden_activation='ReLU', head_activation='Tanh'):
    
        super(ResidualGenerator3d_2, self).__init__()
        self.input_dim = input_dim 

        head_activ = getattr(nn, head_activation)()
        hidden_activ = getattr(nn, hidden_activation)(True)

        self.main = nn.Sequential(
            ResidualBlock3d(in_channels=input_dim, out_channels=output_image_size * 8, kernel_size=(3, 3, 6), stride=1, padding=(1, 0, 0), output_padding=0),
          
            ResidualBlock3d(in_channels=output_image_size * 8, out_channels=output_image_size * 4, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1), output_padding=(0, 1, 1)),
            ResidualBlock3d(in_channels=output_image_size * 4, out_channels=output_image_size * 2, kernel_size=(3, 4, 5), stride=(1, 2, 2), padding=(1, 1, 1), output_padding=(0, 1, 1),  pad=True),
        
            ResidualBlock3d(in_channels=output_image_size * 2, out_channels=output_image_size, kernel_size=(3, 5, 4), stride=(1, 2, 2), padding=(1, 1, 1), bias=True, output_padding=(0, 1, 1), pad=True),
            ResidualBlock3d(in_channels=output_image_size , out_channels=output_image_size // 2, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1), bias=True, output_padding=(0, 1, 1)),
            
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose3d(int(output_image_size/2), num_channels, kernel_size=(3,4,4), stride=(1,2,2), padding=(1, 1, 1), bias=True),
            head_activ
        ) 
    
    def forward(self, input):
        return self.main(input) 


class ResidualBlock3d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding=0, bias=True, pad=False):
        super(ResidualBlock3d, self).__init__()
        self.pad = pad

        self.dconv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(True)

        """ Shortcut Connection (Identity Mapping) """
        self.s = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=(1,1,1), padding=(0,0, 0), output_padding=output_padding, stride=(1, 2, 2))


    def forward(self, input):
        x = self.dconv(input)
        x = self.bn(x)
        x = self.relu(x)

        s = self.s(input)

        if self.pad: 
            pad_dims = (x.shape[4]-s.shape[4], 0, x.shape[3]-s.shape[3], 0)
            s =  F.pad(s, pad_dims, "constant", 0)  # effectively zero padding

        x = x + s
        return x   


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def main():
    
    batch_size = 1
    num_measurements = 100
    input_depth = 10         # number of channles in the input matrix; this should be set to the number of matrices
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Generator Network
    model = Generator3d(input_dim=num_measurements, output_image_size=64, num_channels=1)
    model = model.to(device)
    summary(model, (num_measurements, input_depth, 1, 1)) 
    model.apply(weights_init)
    
    # input shape: [batch_size, input_dim, depth_dim, width, height]
    input = torch.randn(batch_size, num_measurements, input_depth, 1, 1)
    input = input.to(device)

    output = model.forward(input) 
    print("Generator Output: ", output.shape)

    # # Generator Network
    model = ResidualGenerator3d(input_dim=num_measurements, output_image_size=64, num_channels=1)
    model = model.to(device)
    summary(model, (num_measurements, input_depth, 1, 1)) 

    model.apply(weights_init)

    output = model.forward(input) 
    print("Residual Generator Output: ", output.shape)

    model = Generator3d_2(input_dim=num_measurements, output_image_size=64, num_channels=1)
    model = model.to(device)
    summary(model, (num_measurements, input_depth, 1, 1)) 

    model.apply(weights_init)

    output = model.forward(input) 
    print("Generator3d_2 Output: ", output.shape)

    model = ResidualGenerator3d_2(input_dim=num_measurements, output_image_size=64, num_channels=1)
    model = model.to(device)
    summary(model, (num_measurements, input_depth, 1, 1)) 

    model.apply(weights_init)

    output = model.forward(input) 
    print("ResidualGenerator3d_2 Output: ", output.shape)


if __name__ == "__main__":
    main()