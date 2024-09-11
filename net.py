import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWT, IDWT
import random


class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=256, gc=128, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        self.conv1 = nn.Conv2d(nf, gc, 7, 1, 3,padding_mode="reflect")
        self.IN1 = nn.InstanceNorm2d(gc)
        self.conv2 = nn.Conv2d(nf + gc, gc, 5, 1, 2,padding_mode="reflect")
        self.IN2 = nn.InstanceNorm2d(gc)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1,padding_mode="reflect")
        self.IN3 = nn.InstanceNorm2d(gc)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1,padding_mode="reflect")
        self.IN4 = nn.InstanceNorm2d(gc)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1,padding_mode="reflect")
        self.IN5 = nn.InstanceNorm2d(nf)
        self.lrelu1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)



    def forward(self, x):

        x1 = self.lrelu1(self.IN1(self.conv1(x)))

        x2 = self.lrelu1(self.IN2(self.conv2(torch.cat((x, x1), 1))))

        x3 = self.lrelu1(self.IN3(self.conv3(torch.cat((x, x1, x2), 1))))

        x4 = self.lrelu1(self.IN4(self.conv4(torch.cat((x, x1, x2, x3), 1))))

        x5 = self.lrelu1(self.IN5(self.conv5(torch.cat((x, x1, x2, x3, x4), 1))))

        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, nf=256, gc=128):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x

class Conv(nn.Module):
    def __init__(self, input_nc=3, output_nc=4, kernel_size=3, stride=2, padding=1,
                 active_function="LeakyReLU"):
        super(Conv, self).__init__()
        self.input_nc = input_nc
        self.output_nc  =output_nc
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.function = active_function
        self.conv = nn.Conv2d(self.input_nc, self.output_nc, self.kernel_size, self.stride, self.padding)
        self.IN = nn.InstanceNorm2d(output_nc)
        self.fun = nn.LeakyReLU(negative_slope=0.2,
                                inplace=True) if active_function == "LeakyReLU" else nn.Sigmoid()

    def forward(self, x):
        return self.fun(self.IN(self.conv(x)))

class ConvT(nn.Module):
    def __init__(self, input_nc=3, output_nc=4):
        super(ConvT, self).__init__()
        self.convT = nn.ConvTranspose2d(input_nc, output_nc, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.IN = nn.InstanceNorm2d(output_nc)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        return self.lrelu(self.IN(self.convT(x)))
class DepthNet(nn.Module):
    def __init__(self):
        super(DepthNet, self).__init__()
        self.conv1 = Conv(3, 16)
        self.conv2 = Conv(16, 32)
        self.conv3 = Conv(32, 64)
        self.conv4 = Conv(64, 128)
        self.conv5 = Conv(128, 256)

        self.sameconv4 = Conv(256, 4, 1, 1, 0,  active_function="Sigmoid")
        self.sameconv3 = Conv(192, 3, 1, 1, 0,  active_function="Sigmoid")
        self.sameconv2 = Conv(128, 3, 1, 1, 0,  active_function="Sigmoid")
        self.sameconv1 = Conv(80, 3, 1, 1, 0,  active_function="Sigmoid")

        self.convt1 = ConvT(128, 64)
        self.convt2 = ConvT(192, 96)
        self.convt3 = ConvT(256, 128)
        self.convt4 = ConvT(256, 128)

        self.ifm = IDWT(mode='zero', wave='haar').cuda()
        self.SIGMOID = nn.Sigmoid()
        #self.result = result
        #self.MaxUnpool = nn.MaxUnpool2d(2, stride=2)
        #self.Maxpool = nn.MaxPool2d(2, stride=2, return_indices=True)
        self.Avgpool = nn.AvgPool2d(8, stride=8)

        #self.rrdb = RRDB()
    def firstnewcat(self, x):
        y1 = x[:, 0:int(x.shape[1] / 4), :, :]
        y2 = x[:, int(x.shape[1] / 4):int(x.shape[1] / 2), :, :]
        y3 = x[:, int(x.shape[1] / 2):int(x.shape[1] * 3 / 4), :, :]
        y4 = x[:, int(x.shape[1] * 3 / 4):, :, :]
        result1 = torch.cat((y1, y2), dim=2)
        result2 = torch.cat((y3, y4), dim=2)
        result = torch.cat((result1, result2), dim=3)
        return result

    def continue_newcat(self, x, y):
        y1 = x[:, 0:int(x.shape[1] / 3), :, :]
        y2 = x[:, int(x.shape[1] / 3):int(x.shape[1] * 2 / 3), :, :]
        y3 = x[:, int(x.shape[1] * 2 / 3):, :, :]
        result1 = torch.cat((y1, y2), dim=2)
        result2 = torch.cat((y3, y), dim=2)
        result = torch.cat((result1, result2), dim=3)
        return result

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        out4 = self.conv4(out3)
        out5 = self.conv5(out4)
        #out5 = self.rrdb(out5)

        rout4 = torch.cat((self.convt4(out5), out4), dim=1)
        # print(rout4.shape)
        rout3 = torch.cat((self.convt3(rout4), out3), dim=1)
        # print(rout3.shape)
        rout2 = torch.cat((self.convt2(rout3), out2), dim=1)
        # print(rout2.shape)
        rout1 = torch.cat((self.convt1(rout2), out1), dim=1)
        # print(rout1.shape)

        finalout4 = self.sameconv4(rout4)
        finalout3 = self.sameconv3(rout3)
        finalout2 = self.sameconv2(rout2)
        finalout1 = self.sameconv1(rout1)
        # print(finalout4[:,0:int(finalout4.shape[1]/4),:,:].shape)
        # print(torch.cat((finalout4[:,int(finalout4.shape[1]/4):int(finalout4.shape[1]/2),:,:],finalout4[:,int(finalout4.shape[1]/4):int(finalout4.shape[1]/2),:,:],finalout4[:,int(finalout4.shape[1]/4):int(finalout4.shape[1]/2),:,:]),dim=1).shape)
        # print(torch.cat((finalout4[:,int(finalout4.shape[1]/4):int(finalout4.shape[1]/2),:,:],finalout4[:,int(finalout4.shape[1]/2):int(finalout4.shape[1]*3/4),:,:],finalout4[:,int(finalout4.shape[1]*3/4):,:,:]),dim=1).unsqueeze(dim=1).shape)



        result4 = self.ifm((finalout4[:, 0:int(finalout4.shape[1] / 4), :, :], [torch.cat((finalout4[:, int(
            finalout4.shape[1] / 4):int(finalout4.shape[1] / 2), :, :], finalout4[:, int(finalout4.shape[1] / 2):int(
            finalout4.shape[1] * 3 / 4), :, :], finalout4[:, int(finalout4.shape[1] * 3 / 4):, :, :]), dim=1).unsqueeze(
            dim=1)]))
        result3 = self.ifm((finalout3[:, 0:int(finalout3.shape[1] / 3), :, :], [torch.cat((finalout3[:, int(
            finalout3.shape[1] / 3):int(finalout3.shape[1] * 2 / 3), :, :], finalout3[:,
                                                                            int(finalout3.shape[1] * 2 / 3):int(
                                                                                finalout3.shape[1]), :, :], result4),
                                                                                          dim=1).unsqueeze(dim=1)]))
        result2 = self.ifm((finalout2[:, 0:int(finalout2.shape[1] / 3), :, :], [torch.cat((finalout2[:, int(
            finalout2.shape[1] / 3):int(finalout2.shape[1] * 2 / 3), :, :], finalout2[:,
                                                                            int(finalout2.shape[1] * 2 / 3):int(
                                                                                finalout2.shape[1]), :, :], result3),
                                                                                          dim=1).unsqueeze(dim=1)]))
        result1 = self.ifm((finalout1[:, 0:int(finalout1.shape[1] / 3), :, :], [torch.cat((finalout1[:, int(
            finalout1.shape[1] / 3):int(finalout1.shape[1] * 2 / 3), :, :], finalout1[:,
                                                                            int(finalout1.shape[1] * 2 / 3):int(
                                                                                finalout1.shape[1]), :, :], result2),
                                                                                          dim=1).unsqueeze(dim=1)]))
        result1 = self.Avgpool(result1)
        result1 = torch.nn.functional.interpolate(result1, scale_factor=8, mode='nearest')
        result1 = torch.nn.Sigmoid()(result1)
        return result1  # output

# class Discriminator_earlystage(nn.Module):
#     def __init__(self,input_nc=3):
#         super(Discriminator_earlystage, self).__init__()
#
#         model = [nn.Conv2d(input_nc, 64, 4, stride=2,padding=1),
#                  #nn.InstanceNorm2d(64),
#                  nn.LeakyReLU(0.2, inplace=True)]
#
#         model += [nn.Conv2d(64, 128, 4, stride=2,padding=1),
#                   nn.InstanceNorm2d(128),
#                   nn.LeakyReLU(0.2, inplace=True)]
#
#         model += [nn.Conv2d(128, 256, 4, stride=2,padding=1),
#                   nn.InstanceNorm2d(256),
#                   nn.LeakyReLU(0.2, inplace=True)]
#
#         # FCN classification layer
#         model += [nn.Conv2d(256, 512, 4,stride=1,padding=1),
#                   nn.InstanceNorm2d(64),
#                   nn.LeakyReLU(0.2, inplace=True)]
#
#         model += [nn.Conv2d(512, 1, 4, stride=1,padding=1),
#                   nn.InstanceNorm2d(1),
#                   nn.Sigmoid()]
#
#         self.model = nn.Sequential(*model)
#
#     def forward(self, x):
#         x = self.model(x)
#
#         return x#F.avg_pool2d(x,x.size()[2:]).view(x.size()[0],-1)


class Discriminator_earlystage(nn.Module):
    def __init__(self, input_nc=3):
        super(Discriminator_earlystage, self).__init__()

        model = [nn.Conv2d(input_nc, 32, 4, stride=2, padding=1),
                 # nn.InstanceNorm2d(64),
                 nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(32, 64, 4, stride=2, padding=1),
                  nn.InstanceNorm2d(128),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(64, 128, 4, stride=2, padding=1),
                  nn.InstanceNorm2d(256),
                  nn.LeakyReLU(0.2, inplace=True)]

        # FCN classification layer
        model += [nn.Conv2d(128, 256, 4, stride=1, padding=1),
                  nn.InstanceNorm2d(64),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(256, 1, 4, stride=1, padding=1),
                  nn.InstanceNorm2d(1),
                  nn.Sigmoid()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)

        return x  # F.avg_pool2d(x,x.size()[2:]).view(x.size()[0],-1)
class ResBlock(nn.Module):
    def __init__(self, n_chans):
        super(ResBlock, self).__init__()
        self.conv = nn.Conv2d(n_chans, n_chans,
                              kernel_size=3,padding=1,
                              bias=False)
        self.ins_norm = nn.InstanceNorm2d(num_features=n_chans)  # <5>


    def forward(self, x):
        out = self.conv(x)
        out = self.ins_norm(out)
        out = torch.relu(out)
        return out + x
class Discriminator_latestage(nn.Module):
    def __init__(self,input_nc=3):
        super(Discriminator_latestage, self).__init__()

        # A bunch of convolutions one after another
        model = [nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                 # nn.InstanceNorm2d(64),
                 nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(64, 128, 4, stride=2, padding=1),
                  nn.InstanceNorm2d(128),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(128, 256, 4, stride=2, padding=1),
                  nn.InstanceNorm2d(256),
                  nn.LeakyReLU(0.2, inplace=True)]

        # FCN classification layer
        model += [nn.Conv2d(256, 512, 4, stride=1, padding=1),
                  nn.InstanceNorm2d(64),
                  nn.LeakyReLU(0.2, inplace=True)]

        for _ in range(1):
            model += [ResBlock(512)]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, stride=1,padding=1),
                  nn.InstanceNorm2d(1),
                  nn.Sigmoid()]
        self.model = nn.Sequential(*model)

        #self.conv16 = nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1)
        #self.conv17 = nn.Conv2d(32, 1, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.model(x)

        return x#F.avg_pool2d(x,x.size()[2:]).view(x.size()[0],-1)  # output



class Res2Block(nn.Module):
    def __init__(self):
        super(Res2Block, self).__init__()
        self.covn1_1 = nn.Conv2d(32*4, 32*4, kernel_size=1)
        self.covn1_2 = nn.Conv2d(48*4, 48*4, kernel_size=1)
        self.covn1_3 = nn.Conv2d(56*4, 56*4, kernel_size=1)
        self.IN1 = nn.InstanceNorm2d(32*4)
        self.IN4 = nn.InstanceNorm2d(48*4)
        self.IN5 = nn.InstanceNorm2d(56*4)
        self.relu = nn.ReLU()
    def forward(self,x):
        out1 = x[:, :32*4, :, :]
        out2_1 = self.relu(self.IN1(self.covn1_1(x[:, 32*4:64*4, :, :])))
        out2 = out2_1[:, :16*4, :, :]
        out3_1 = self.relu(self.IN4(self.covn1_2(torch.cat([x[:, 64*4:96*4, :], out2_1[:, 16*4:, :, :]], dim=1))))
        out3 = out3_1[:, :24*4, :, :]
        out4 = self.relu(self.IN5(self.covn1_3(torch.cat([x[:, 96*4:, :, :], out3_1[:, 24*4:, :, :]], dim=1))))
        x = torch.cat([out1, out2, out3, out4], dim=1)
        return x

class Discriminator_finalstage(nn.Module):
    def __init__(self,input_nc=3):
        super(Discriminator_finalstage, self).__init__()

        # A bunch of convolutions one after another
        model = [nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                 # nn.InstanceNorm2d(64),
                 nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(64, 128, 4, stride=2, padding=1),
                  nn.InstanceNorm2d(128),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(128, 256, 4, stride=2, padding=1),
                  nn.InstanceNorm2d(256),
                  nn.LeakyReLU(0.2, inplace=True)]

        # FCN classification layer
        model += [nn.Conv2d(256, 512, 4, stride=1, padding=1),
                  nn.InstanceNorm2d(64),
                  nn.LeakyReLU(0.2, inplace=True)]


        # model += [nn.Conv2d(128, 256, 4, stride=2),
        #           nn.InstanceNorm2d(512),
        #           nn.LeakyReLU(0.2, inplace=True)]

        for _ in range(3):
            model += [Res2Block()]

        # FCN classification layer
        model += [nn.Conv2d(512,1, 4, stride=1,padding=1),
                  nn.InstanceNorm2d(1),
                  nn.Sigmoid()]


        self.model = nn.Sequential(*model)

        #self.conv16 = nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1)
        #self.conv17 = nn.Conv2d(32, 1, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.model(x)

        return x#F.avg_pool2d(x,x.size()[2:]).view(x.size()[0],-1)  # output

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator, self).__init__()

        self.pad1 = nn.ReflectionPad2d(3)
        self.con1 = nn.Conv2d(input_nc, 64, kernel_size=7)
        self.inorm1 = nn.InstanceNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)

        # Downsampling
        in_features = 64
        out_features = in_features * 2
        self.con2_1 = nn.Conv2d(in_features, out_features, kernel_size=3, stride=2, padding=1)
        self.inorm2_1 = nn.InstanceNorm2d(out_features)
        self.relu2_1 = nn.ReLU(inplace=True)

        in_features = out_features
        out_features = in_features * 2
        self.con2_2 = nn.Conv2d(in_features, out_features, kernel_size=3, stride=2, padding=1)
        self.inorm2_2 = nn.InstanceNorm2d(out_features)
        self.relu2_2 = nn.ReLU(inplace=True)


        self.res_block1 = RRDB()
        # self.res_block2 = RRDB()
        # self.res_block3 = RRDB()



        # Upsampling
        in_features = out_features
        out_features = in_features // 2
        self.con2t_1 = nn.ConvTranspose2d(in_features, out_features, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.inorm2t_1 = nn.InstanceNorm2d(out_features)
        self.relu2t_1 = nn.ReLU(inplace=True)

        in_features = out_features
        out_features = in_features // 2
        self.con2t_2 = nn.ConvTranspose2d(in_features, out_features, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.inorm2t_2 = nn.InstanceNorm2d(out_features)
        self.relu2t_2 = nn.ReLU(inplace=True)

        # Output layer
        self.con3 = nn.Conv2d(out_features, output_nc, kernel_size=7, padding=3)
        self.tanh = nn.Tanh()#Sigmoid()




    def forward(self, x):

        samex = x

        x = self.pad1(x)
        x = self.con1(x)
        x = self.inorm1(x)
        x = self.relu1(x)



        # Downsampling
        x = self.con2_1(x)
        x = self.inorm2_1(x)
        x = self.relu2_1(x)

        x = self.con2_2(x)
        x = self.inorm2_2(x)
        x = self.relu2_2(x)





        x = self.res_block1(x)


        # x = self.res_block2(x)
        #
        #
        # x = self.res_block3(x)


        #Upsampling
        x = self.con2t_1(x)
        x = self.inorm2t_1(x)
        x = self.relu2t_1(x)

        x = self.con2t_2(x)
        x = self.inorm2t_2(x)
        x = self.relu2t_2(x)

        # Output layer
        x = self.con3(x)

        x = self.tanh(x)*0.5+0.5


        output = x# 1.001- samex + x
        #print(output)

        return output


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)
class Generator2(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, n_residual_blocks=9):
        super(Generator2, self).__init__()

        # Initial convolution block
        model = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 64, 7),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True) ]

        # Downsampling
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features//2
        for _ in range(2):
            model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2

        # Output layer
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(64, output_nc, 7),
                    nn.Sigmoid() ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.tensor11 = torch.tensor([[[[1,0],[0,0]],[[0,0],[0,0]],[[0,0],[0,0]]],[[[0,0],[0,0]],[[1,0],[0,0]],[[0,0],[0,0]]],[[[0,0],[0,0]],[[0,0],[0,0]],[[1,0],[0,0]]]],dtype=torch.float32)

        self.weight1 = nn.Parameter(self.tensor11, requires_grad=False)  # 自定义的权值
        self.bias1 = nn.Parameter(torch.zeros(3), requires_grad=False)  # 自定义的偏置



    def forward(self, x):

        out = F.conv2d(x, self.weight1, self.bias1, stride=2, padding=0)
        return out

class CNN_channel1(nn.Module):
    def __init__(self):
        super(CNN_channel1, self).__init__()
        self.tensor11 = torch.tensor([[[[1, 0], [0, 0]]]], dtype=torch.float32)

        self.weight1 = nn.Parameter(self.tensor11, requires_grad=False)  # 自定义的权值
        self.bias1 = nn.Parameter(torch.zeros(1), requires_grad=False)  # 自定义的偏置



    def forward(self, x):

        out = F.conv2d(x, self.weight1, self.bias1, stride=2, padding=0)
        return out

if __name__ =="__main__":
   model = Discriminator_finalstage()
   inputTensor = torch.randn([1,3,256,256])
   print(model(inputTensor).shape)
