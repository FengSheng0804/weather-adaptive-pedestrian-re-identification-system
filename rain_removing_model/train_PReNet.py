# This code is used to train a rain removal model using PReNet.
import os
import argparse
import torch
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from data.DerainDataset import *
from utils import *
from torch.optim.lr_scheduler import ReduceLROnPlateau
from SSIM import SSIM
from models.PReNet import *


parser = argparse.ArgumentParser(description="PReNet_train")
parser.add_argument("--data_path",type=str, default="datasets/DerainDataset/train",help='path to training data')
parser.add_argument("--log_save_path", type=str, default="rain_removing_model/results/logs", help='path to save log files')
parser.add_argument("--model_save_path", type=str, default="rain_removing_model/weights", help='path to save model files')
parser.add_argument("--save_freq",type=int,default=1,help='save intermediate model')
parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs")
parser.add_argument("--lr", type=float, default=5e-4, help="initial learning rate")
parser.add_argument("--use_gpu", type=bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
parser.add_argument("--recurrent_iter", type=int, default=6, help='number of recursive stages')
opt = parser.parse_args()

if opt.use_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id


def train():

    print('Loading dataset ...\n')
    loader_train = train_dataloader(
        data_path=opt.data_path,
        batch_size=opt.batch_size,
        num_workers=4,
        use_patch=True
    )
    print("# of training samples: %d\n" % len(loader_train.dataset))

    # Build model
    model = PReNet(recurrent_iter=opt.recurrent_iter, use_GPU=opt.use_gpu)
    print_network(model)

    # loss function
    # criterion = nn.MSELoss(size_average=False)
    criterion = SSIM()

    # Move to GPU
    if opt.use_gpu:
        model = model.cuda()
        criterion.cuda()

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=opt.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    # record training
    writer = SummaryWriter(opt.log_save_path)

    # load the lastest model
    initial_epoch = findLastCheckpoint(save_dir=opt.model_save_path)  # load the lastest model in matconvnet style
    if initial_epoch > 0:
        print('resuming by loading epoch %d' % initial_epoch)
        model.load_state_dict(torch.load(os.path.join(opt.model_save_path, 'net_epoch%d.pth' % initial_epoch)))

    # start training
    step = 0
    best_psnr = 0.0
    for epoch in range(initial_epoch, opt.epochs):
        for param_group in optimizer.param_groups:
            print('learning rate %f' % param_group["lr"])

        ## epoch training start
        for i, (input_train, target_train) in enumerate(loader_train, 0):
            model.train()
            model.zero_grad()
            optimizer.zero_grad()

            input_train, target_train = Variable(input_train), Variable(target_train)

            if opt.use_gpu:
                input_train, target_train = input_train.cuda(), target_train.cuda()

            out_train, _ = model(input_train)
            pixel_metric = criterion(target_train, out_train)
            loss = -pixel_metric

            loss.backward()
            optimizer.step()

            # training curve
            model.eval()
            out_train, _ = model(input_train)
            out_train = torch.clamp(out_train, 0., 1.)
            psnr_train = batch_PSNR(out_train, target_train, 1.)
            print("[epoch %d][%d/%d] loss: %.4f, pixel_metric: %.4f, PSNR: %.4f" %
                  (epoch+1, i+1, len(loader_train), loss.item(), pixel_metric.item(), psnr_train))

            if step % 10 == 0:
                # Log the scalar values
                writer.add_scalar('loss', loss.item(), step)
                writer.add_scalar('PSNR on training data', psnr_train, step)
            step += 1
        ## epoch training end
        # 自适应调整学习率
        scheduler.step(loss.item())

        # log the images
        model.eval()
        out_train, _ = model(input_train)
        out_train = torch.clamp(out_train, 0., 1.)
        im_target = utils.make_grid(target_train.data, nrow=8, normalize=True, scale_each=True)
        im_input = utils.make_grid(input_train.data, nrow=8, normalize=True, scale_each=True)
        im_derain = utils.make_grid(out_train.data, nrow=8, normalize=True, scale_each=True)
        writer.add_image('clean image', im_target, epoch+1)
        writer.add_image('rainy image', im_input, epoch+1)
        writer.add_image('deraining image', im_derain, epoch+1)

        # save model
        if psnr_train > best_psnr:
            best_psnr = psnr_train
            torch.save(model.state_dict(), os.path.join(opt.model_save_path, 'best.pth'))
            print(f"[epoch {epoch+1}] best.pth updated, PSNR: {psnr_train:.4f}")
        if epoch % opt.save_freq == 0:
            torch.save(model.state_dict(), os.path.join(opt.model_save_path, 'net_epoch%d.pth' % (epoch+1)))


if __name__ == "__main__":
    # train时的pic_size应该和推理测试时的一致
    train()