
import argparse
from torch.utils.data import DataLoader
from net import *
from utils import ReplayBuffer
from tqdm import tqdm
from datasets import ImageDataset
import os
import numpy as np
import cv2
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

init_lr = 0.0001
batchSize = 4
n_cpu = 4

continue_train = 0

netDepth1 = torch.load(r"F:\pythonDoc\wavelet-monodepth-main\wavelet-monodepth-main\KITTI\encoder.pt").eval().cuda()
netDepth2 = torch.load(r"F:\pythonDoc\wavelet-monodepth-main\wavelet-monodepth-main\KITTI\decoder.pt").eval().cuda()

netDehaze = Generator(3,3)



netD_A1 = Discriminator_earlystage()
netD_A2 = Discriminator_latestage()
netD_A3 = Discriminator_finalstage()

netD_B3 = Discriminator_finalstage()
netD_B2 = Discriminator_latestage()
netD_B1 = Discriminator_earlystage()


netDepth1.to(device, non_blocking=True).train()
netDepth2.to(device, non_blocking=True).train()
netDehaze.to(device, non_blocking=True).train()

netD_A1.to(device, non_blocking=True).train()
netD_A2.to(device, non_blocking=True).train()
netD_A3.to(device, non_blocking=True).train()

netD_B3.to(device, non_blocking=True).train()
netD_B2.to(device, non_blocking=True).train()
netD_B1.to(device, non_blocking=True).train()



if continue_train:
    step = 0#opt.epoch

    netDepth1.load_state_dict(torch.load(r""))
    netDepth2.load_state_dict(torch.load(r""))
    netDehaze.load_state_dict(torch.load(r""))

    netD_A3.load_state_dict(torch.load(r""))
    netD_A2.load_state_dict(torch.load(r""))
    netD_A1.load_state_dict(torch.load(r""))


else:
    step = 0




criterion_GAN = nn.BCEWithLogitsLoss()
criterion_cycle = torch.nn.MSELoss()
criterion_identity = torch.nn.MSELoss()






optimizer_G1 = torch.optim.AdamW(netDehaze.parameters(), lr=init_lr, betas=(0.5, 0.999))
optimizer_G2 = torch.optim.AdamW(netDepth1.parameters(), lr=init_lr*0.1, betas=(0.5, 0.999))
optimizer_G3 = torch.optim.AdamW(netDepth2.parameters(), lr=init_lr*0.1, betas=(0.5, 0.999))

optimizer_D_A3 = torch.optim.AdamW(netD_A3.parameters(), lr=init_lr, betas=(0.5, 0.999))
optimizer_D_A2 = torch.optim.AdamW(netD_A2.parameters(), lr=init_lr, betas=(0.5, 0.999))
optimizer_D_A1 = torch.optim.AdamW(netD_A1.parameters(), lr=init_lr, betas=(0.5, 0.999))

optimizer_D_B3 = torch.optim.AdamW(netD_B3.parameters(), lr=init_lr, betas=(0.5, 0.999))
optimizer_D_B2 = torch.optim.AdamW(netD_B2.parameters(), lr=init_lr, betas=(0.5, 0.999))
optimizer_D_B1 = torch.optim.AdamW(netD_B1.parameters(), lr=init_lr, betas=(0.5, 0.999))





target_real = torch.ones([batchSize, 1, 30, 30],requires_grad=False).cuda()
target_fake = torch.zeros([batchSize, 1, 30, 30],requires_grad=False).cuda()

fakeClear_buffer = ReplayBuffer()
fakeHaze_buffer = ReplayBuffer()


dataloader = DataLoader(ImageDataset(),batch_size=batchSize, shuffle=True, num_workers=n_cpu,pin_memory=True,drop_last=True)

def lr_schedule_cosdecay(t, T, init_lr=init_lr, end_lr=0.0000001):
    lr = init_lr+end_lr-t*init_lr/T
    return lr

def calculateA(image):
    A = torch.max(torch.max_pool2d(image, kernel_size=256), dim=1)[0].unsqueeze(dim=1).detach()
    return A.detach()

def getHazeImage(net1,net2,image):

    max_values_1, _ = torch.max(image[:, :3, :, :], dim=1)

    max_values_2, _ = torch.max(max_values_1, dim=1)
    max_values, _ = torch.max(max_values_2, dim=1)
    A = max_values.unsqueeze(1).unsqueeze(2).unsqueeze(3)

    features = net1(image)
    outputs = net2(features)
    depth = outputs[("disp", 0)]

    min_val = torch.min(depth)
    max_val = torch.max(depth)


    depth = torch.div(torch.sub(depth, min_val), torch.sub(max_val, min_val))
    depth = 1-depth



    beta = random.uniform(0.1,1)
    return (torch.exp(-1 * depth * beta) * (image - A) + A)

def getClearImage(net,image):

    max_values_1, _ = torch.max(image[:, :3, :, :], dim=1)

    max_values_2, _ = torch.max(max_values_1, dim=1)
    max_values, _ = torch.max(max_values_2, dim=1)
    A = max_values.unsqueeze(1).unsqueeze(2).unsqueeze(3)

    transmap = net(image)
    transmap = (A - image + transmap) / A + 0.001
    clearImage = (image - A*(1-transmap))/transmap
    return clearImage

if __name__ == "__main__":

        for param_group in optimizer_G1.param_groups:
            print(f"Learning rate: {param_group['lr']}")
            break
        epoch = 0
        lossa = 0
        losscycle = 0
        lossg = 0
        lossd = 0
        loop = tqdm(enumerate(dataloader), total=len(dataloader))
        threshold = 10

        for i, batch in loop:

            realClear = batch['clearimage'].to(device)
            realHaze = batch['hazeimage'].to(device)



            if True:  #train dehaze network 训练去雾网络
                optimizer_G1.zero_grad()


                sameClear = getClearImage(netDehaze,realClear)
                loss_identity = criterion_identity(sameClear,realClear) *0.1
                

                fakeHaze =  getHazeImage(netDepth1,netDepth2,realClear)
                reClear = getClearImage(netDehaze,fakeHaze)
                loss_cycle = criterion_cycle(reClear, realClear)

                fakeClear = getClearImage(netDehaze,realHaze)

                pred_fake3 = netD_A3(fakeClear)
                pred_fake2 = netD_A2(fakeClear)
                pred_fake1 = netD_A1(fakeClear)

                # print(pred_fake1.shape)
                # print(pred_fake2.shape)
                # print(pred_fake3.shape)

                y1 = (i/ len(dataloader) - 1)*(i/ len(dataloader) - 1);
                y3 = (i/len(dataloader))*(i/ len(dataloader))
                y2  = 1-y1-y3
                loss_GAN_Dehaze3 = criterion_GAN(pred_fake3, target_real) * 0.5*y3
                loss_GAN_Dehaze2 = criterion_GAN(pred_fake2, target_real) * 0.5*y2
                loss_GAN_Dehaze1 = criterion_GAN(pred_fake1, target_real) * 0.5*y1


                (loss_GAN_Dehaze3 + loss_GAN_Dehaze2 + loss_GAN_Dehaze1 + loss_cycle + loss_identity).backward()#(loss_GAN_Dehaze3 + loss_GAN_Dehaze2 + loss_GAN_Dehaze1 + loss_cycle_ABA).backward()#

                optimizer_G1.step()

                lossa = lossa + loss_identity.detach()  # +loss_identity_B.detach()
                losscycle = losscycle + loss_cycle.detach()  # +loss_cycle_BAB.detach()


            if True:  # train depth net 训练深度网络
                optimizer_G2.zero_grad()
                optimizer_G3.zero_grad()


                fakeHaze = getHazeImage(netDepth1, netDepth2, realClear)

                pred_fake3 = netD_A3(fakeHaze)
                pred_fake2 = netD_A2(fakeHaze)
                pred_fake1 = netD_A1(fakeHaze)
                y1 = (epoch / 300 - 1) * (epoch / 300 - 1);
                y3 = (epoch / 300) * (epoch / 300)
                y2 = 1 - y1 - y3
                loss_GAN_Haze3 = criterion_GAN(pred_fake3, target_real) * 0.5 * y3
                loss_GAN_Haze2 = criterion_GAN(pred_fake2, target_real) * 0.5 * y2
                loss_GAN_Haze1 = criterion_GAN(pred_fake1, target_real) * 0.5 * y1

                (loss_GAN_Haze3 + loss_GAN_Haze2 + loss_GAN_Haze1).backward()  # (loss_GAN_Dehaze3 + loss_GAN_Dehaze2 + loss_GAN_Dehaze1 + loss_cycle_ABA).backward()#

                optimizer_G2.step()
                optimizer_G3.step()


            if True: #train discriminator 训练判别器


                optimizer_D_A3.zero_grad()
                optimizer_D_A2.zero_grad()
                optimizer_D_A1.zero_grad()

                pred_real3 = netD_A3(realClear)
                pred_real2 = netD_A2(realClear)
                pred_real1 = netD_A1(realClear)
                loss_D_real3 = criterion_GAN(pred_real3, target_real)
                loss_D_real2 = criterion_GAN(pred_real2, target_real)
                loss_D_real1 = criterion_GAN(pred_real1, target_real)

                fakeClear = fakeClear_buffer.push_and_pop(fakeClear)
                pred_fake3 = netD_A3(fakeClear.detach())
                pred_fake2 = netD_A2(fakeClear.detach())
                pred_fake1 = netD_A1(fakeClear.detach())
                loss_D_fake3 = criterion_GAN(pred_fake3, target_fake)
                loss_D_fake2 = criterion_GAN(pred_fake2, target_fake)
                loss_D_fake1 = criterion_GAN(pred_fake1, target_fake)

                loss_D_A = (loss_D_real3 + loss_D_fake3 + loss_D_real2 + loss_D_fake2 + loss_D_real1 + loss_D_fake1) * 0.5


                loss_D_A.backward()

                optimizer_D_A3.step()
                optimizer_D_A2.step()
                optimizer_D_A1.step()



                optimizer_D_B3.zero_grad()
                optimizer_D_B2.zero_grad()
                optimizer_D_B1.zero_grad()


                pred_real3 = netD_B3(realHaze)
                pred_real2 = netD_B2(realHaze)
                pred_real1 = netD_B1(realHaze)
                loss_D_real3 = criterion_GAN(pred_real3, target_real)
                loss_D_real2 = criterion_GAN(pred_real2, target_real)
                loss_D_real1 = criterion_GAN(pred_real1, target_real)

                # Fake loss
                fakeHaze = fakeHaze_buffer.push_and_pop(fakeHaze)
                pred_fake3 = netD_B3(fakeHaze.detach())
                pred_fake2 = netD_B2(fakeHaze.detach())
                pred_fake1 = netD_B1(fakeHaze.detach())
                loss_D_fake3 = criterion_GAN(pred_fake3, target_fake)
                loss_D_fake2 = criterion_GAN(pred_fake2, target_fake)
                loss_D_fake1 = criterion_GAN(pred_fake1, target_fake)

                loss_D_B = (loss_D_real3 + loss_D_fake3 + loss_D_real2 + loss_D_fake2 + loss_D_real1 + loss_D_fake1) * 0.5

                loss_D_B.backward()

                optimizer_D_B3.step()
                optimizer_D_B2.step()
                optimizer_D_B1.step()






            lossg = lossg  + loss_GAN_Dehaze3.detach()  + loss_GAN_Dehaze2.detach()  + loss_GAN_Dehaze1.detach() # +loss_cycle_BAB.detach()

            lossd = lossd +loss_D_A.detach()

            loop.set_description(f'finalEpoch [{epoch}/{100}]')
            loop.set_postfix(lossa=lossa, loss_gan=lossg, losscycle=losscycle,lossd= lossd)

        if (i % 1000 == 0):

            lr = lr_schedule_cosdecay(i, len(dataloader))
            for param_group in optimizer_G1.param_groups:
                param_group["lr"] = lr
            for param_group in optimizer_G2.param_groups:
                param_group["lr"] = lr*0.1
            for param_group in optimizer_G3.param_groups:
                param_group["lr"] = lr*0.1
            for param_group in optimizer_D_B3.param_groups:
                param_group["lr"] = lr
            for param_group in optimizer_D_B2.param_groups:
                param_group["lr"] = lr
            for param_group in optimizer_D_B1.param_groups:
                param_group["lr"] = lr
            for param_group in optimizer_D_A3.param_groups:
                param_group["lr"] = lr
            for param_group in optimizer_D_A2.param_groups:
                param_group["lr"] = lr
            for param_group in optimizer_D_A1.param_groups:
                param_group["lr"] = lr


            epoch = epoch + 1

            for name, param in netDehaze.named_parameters():
                print(f"{name}: {optimizer_G1.param_groups[0]['lr']}")
                break
            for name, param in netDepth1.named_parameters():
                print(f"{name}: {optimizer_G2.param_groups[0]['lr']}")
                break

            #torch.save(netG_B2A.state_dict(),r'C:\Users\Admin\Desktop\paper2\paper2Code\weight\\' + str(epoch) + "_" + str(losscycle)[7:15] + "_" + str(lossa)[7:15] + '.pth')

            torch.save(netDepth1.state_dict(), r'C:\Users\Admin\Desktop\paper2\paper2Code\weightall\netDepth1'+ str(epoch) + "_" + str(losscycle)[7:15] + "_" + str(lossa)[7:15] + '.pth')
            torch.save(netDepth2.state_dict(), r'C:\Users\Admin\Desktop\paper2\paper2Code\weightall\netDepth2' + str(epoch) + "_" + str(losscycle)[7:15] + "_" + str(lossa)[7:15] + '.pth')
            torch.save(netDehaze.state_dict(),r'C:\Users\Admin\Desktop\paper2\paper2Code\weightall\netDehaze' + str(epoch) + "_" + str(losscycle)[7:15] + "_" + str(lossa)[7:15] + '.pth')

            torch.save(netD_A3.state_dict(),r'C:\Users\Admin\Desktop\paper2\paper2Code\weightall\netD_A3_' + str(epoch) + "_" + str(losscycle)[7:15] + "_" + str(lossa)[7:15] + '.pth')
            torch.save(netD_B3.state_dict(), r'C:\Users\Admin\Desktop\paper2\paper2Code\weightall\net_D_B3_' + str(epoch) + "_" + str(losscycle)[7:15] + "_" + str(lossa)[7:15] + '.pth')
            torch.save(netD_A2.state_dict(),r'C:\Users\Admin\Desktop\paper2\paper2Code\weightall\net_D_A2_' + str(epoch) + "_" + str(losscycle)[7:15] + "_" + str(lossa)[7:15] + '.pth')
            torch.save(netD_B2.state_dict(), r'C:\Users\Admin\Desktop\paper2\paper2Code\weightall\net_D_B2_' + str(epoch) + "_" + str(losscycle)[7:15] + "_" + str(lossa)[7:15] + '.pth')
            torch.save(netD_A1.state_dict(),r'C:\Users\Admin\Desktop\paper2\paper2Code\weightall\net_D_A1_' + str(epoch) + "_" + str(losscycle)[7:15] + "_" + str(lossa)[7:15] + '.pth')
            torch.save(netD_B1.state_dict(), r'C:\Users\Admin\Desktop\paper2\paper2Code\weightall\net_D_B1_' + str(epoch) + "_" + str(losscycle)[7:15] + "_" + str(lossa)[7:15] + '.pth')

            lossa = 0
            losscycle = 0
            lossg = 0
            lossd = 0
