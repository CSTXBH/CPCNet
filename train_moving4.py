# LiDAR Segmentation
# 与salsanext的差异,主要是数据读取
# salsanext并未对网络权重进行初始化处理
# 雷达语义+运动分割,全部训练,训练数据为Odometry数据集
# filepath_sm对应Semantic-KITTI数据集,雷达数据,语义和运动真值
# 与train_moving的区别在于只进行当前帧的语义与运动分割

import os
import time
import timeit
import torch.optim
from torch.utils import data
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from argparse import ArgumentParser
from model import CPCNet  # network
from utils.modeltools import netParams2
from utils.loss import *  # loss function
from utils.Lovasz_Softmax import *  # loss function
from utils.ioueval import *
from utils.utils import *
from utils.warmupLR import *
from dataset.kitti_moving import *  # dataset

def val(args, val_loader, model, criterion, criterion_ls, criterion_sh, epoch, DATA, evaluator):
    # 运动分割
    acc_m1 = AverageMeter()
    iou_m1 = AverageMeter()

    epoch_loss = []
    epoch_loss_m1 = []

    evaluator_m1 = evaluator

    # evaluation mode
    model.eval()
    evaluator_m1.reset()

    # empty the cache to infer in high res
    torch.cuda.empty_cache()

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            imagery_velo, imagery_velo2, imagery_velo_std, imagery_velo2_std, _, _, imagery_movinggt, imagery_movinggt2, pose, pose2 = batch
            imagery_velo = Variable(imagery_velo).cuda()
            imagery_velo2 = Variable(imagery_velo2).cuda()
            imagery_velo_std = Variable(imagery_velo_std).cuda()
            imagery_velo2_std = Variable(imagery_velo2_std).cuda()
            imagery_movinggt = Variable(imagery_movinggt.long()).cuda()
            # imagery_movinggt2 = Variable(imagery_movinggt2.long()).cuda()
            # pose = Variable(pose).cuda()
            pose2 = Variable(pose2).cuda()

            # network1_2()
            outputs_s1, outputs_m1 = model(imagery_velo2, imagery_velo_std, imagery_velo2_std, pose2)
            # network1_2(normal=True)
            # outputs_s1, outputs_m1 = model(imagery_velo, imagery_velo2, imagery_velo_std, imagery_velo2_std, pose2)

            # 运动分割
            loss_wce_m1 = criterion(torch.log(outputs_m1.clamp(min=1e-8)), imagery_movinggt)
            loss_jacc_m1 = criterion_ls(outputs_m1, imagery_movinggt)
            loss_m1 = loss_wce_m1 + loss_jacc_m1

            loss = loss_m1

            epoch_loss.append(loss.item())
            epoch_loss_m1.append(loss_m1.item())

            # 运动分割
            argmax_m1 = outputs_m1.argmax(dim=1)
            evaluator_m1.addBatch(argmax_m1, imagery_movinggt)

            # 结果可视化输出，仅20帧
            # if ((epoch + 1) % 5 == 0 and i % 50 == 0):
            if (i % 50 == 0):
                output2 = outputs_m1.cpu().data[0].numpy()
                z_thr2 = np.asarray(np.argmax(output2, axis=0), dtype=np.uint8)
                temp2 = imagery_movinggt.cpu().data[0].numpy()
                z_thr2 = np.vstack((np.uint8(temp2), z_thr2))
                z_fil2 = np.zeros([z_thr2.shape[0], z_thr2.shape[1], 3])
                for l in range(0, args.classes):
                    z_fil2[z_thr2 == l, :] = class_color_moving[l]
                z_img2 = Image.fromarray(np.uint8(z_fil2))
                if not os.path.exists(args.save_seg_dir):
                    os.makedirs(args.save_seg_dir)
                z_img2.save("%s/%s" % (args.save_seg_dir, str(i) + '.png'))

    # 运动分割
    accuracy_m1 = evaluator_m1.getacc()
    jaccard_m1, class_jaccard_m1 = evaluator_m1.getIoU()
    acc_m1.update(accuracy_m1.item(), imagery_velo_std.size(0))
    iou_m1.update(jaccard_m1.item(), imagery_velo_std.size(0))
    # 运动分割
    print('Validation set (moving segmentation, current frame):\n'
          'Acc avg {acc.avg:.3f}\n'
          'IoU avg {iou.avg:.3f}'.format(acc=acc_m1, iou=iou_m1))
    for i, jacc in enumerate(class_jaccard_m1):
        print('IoU class {i:} [{class_str:}] = {jacc:.3f}'.format(
            i=i, class_str=get_xentropy_class_string(DATA, i), jacc=jacc))

    loss = sum(epoch_loss) / len(epoch_loss)
    loss_m1 = sum(epoch_loss_m1) / len(epoch_loss)
    print("Loss = %.4f\t moving loss(current) = %.4f" % (loss, loss_m1))

    return loss

def adjust_learning_rate(args, cur_epoch, max_epoch, baselr):
    lr = baselr * pow((1 - 1.0 * cur_epoch / max_epoch), 0.9)
    return lr

def get_xentropy_class_string(DATA, idx):
    labels = DATA["labels"]
    learning_map_inv = DATA["learning_map_inv"]
    return labels[learning_map_inv[idx]]

def map(label, mapdict):
    # put label from original values to xentropy
    # or vice-versa, depending on dictionary values
    # make learning map a lookup table
    maxkey = 0
    for key, data in mapdict.items():
      if isinstance(data, list):
        nel = len(data)
      else:
        nel = 1
      if key > maxkey:
        maxkey = key
    # +100 hack making lut bigger just in case there are unknown labels
    if nel > 1:
      lut = np.zeros((maxkey + 100, nel), dtype=np.int32)
    else:
      lut = np.zeros((maxkey + 100), dtype=np.int32)
    for key, data in mapdict.items():
      try:
        lut[key] = data
      except IndexError:
        print("Wrong key ", key)
    # do the mapping
    return lut[label]

def to_xentropy(DATA, label):
    learning_map = DATA["learning_map"]
    # put label in xentropy values
    return map(label, learning_map)

def train(args, train_loader, model, criterion, criterion_ls, criterion_sh, optimizer, epoch, DATA, evaluator, scheduler):
    # 运动分割
    acc_m1 = AverageMeter()
    iou_m1 = AverageMeter()

    epoch_loss = []
    epoch_loss_m1 = []

    evaluator_m1 = evaluator

    # empty the cache to infer in high res
    torch.cuda.empty_cache()

    model.train()
    evaluator_m1.reset()

    total_batches = len(train_loader)
    print("=====> the number of iterations per epoch: ", total_batches)

    # lr = adjust_learning_rate(args, cur_epoch=epoch, max_epoch=args.max_epochs, baselr=args.lr)
    for param_group in optimizer.param_groups:
        # param_group['lr'] = lr
        lr = param_group['lr']

    for iteration, batch in enumerate(train_loader, 0):
        imagery_velo, imagery_velo2, imagery_velo_std, imagery_velo2_std, _, _, imagery_movinggt, imagery_movinggt2, pose, pose2 = batch
        imagery_velo = Variable(imagery_velo).cuda()
        imagery_velo2 = Variable(imagery_velo2).cuda()
        imagery_velo_std = Variable(imagery_velo_std).cuda()
        imagery_velo2_std = Variable(imagery_velo2_std).cuda()
        imagery_movinggt = Variable(imagery_movinggt.long()).cuda()
        # imagery_movinggt2 = Variable(imagery_movinggt2.long()).cuda()
        # pose = Variable(pose).cuda()
        pose2 = Variable(pose2).cuda()

        # network1_2()
        # outputs_s1, outputs_m1 = model(imagery_velo2, imagery_velo_std, imagery_velo2_std, pose2)
        # network1_2(normal=True)
        outputs_s1, outputs_m1 = model(imagery_velo, imagery_velo2, imagery_velo_std, imagery_velo2_std, pose2)

        # 运动分割
        loss_wce_m1 = criterion(torch.log(outputs_m1.clamp(min=1e-8)), imagery_movinggt)
        loss_jacc_m1 = criterion_ls(outputs_m1, imagery_movinggt)
        loss_m1 = loss_wce_m1 + loss_jacc_m1

        loss = loss_m1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss.append(loss.item())
        epoch_loss_m1.append(loss_m1.item())

        if args.iou == True:
            # 运动分割
            argmax_m1 = outputs_m1.argmax(dim=1)
            evaluator_m1.addBatch(argmax_m1, imagery_movinggt)

        # step scheduler, 调整学习率,逐batch而不是逐epoch更新学习率
        scheduler.step()

    if args.iou == True:
        # 运动分割
        accuracy_m1 = evaluator_m1.getacc()
        jaccard_m1, class_jaccard_m1 = evaluator_m1.getIoU()
        acc_m1.update(accuracy_m1.item(), imagery_velo_std.size(0))
        iou_m1.update(jaccard_m1.item(), imagery_velo_std.size(0))
        # 运动分割
        print('Training set (moving segmentation, current frame):\n'
              'Acc avg {acc.avg:.3f}\n'
              'IoU avg {iou.avg:.3f}'.format(acc=acc_m1, iou=iou_m1))

    loss = sum(epoch_loss) / len(epoch_loss)
    loss_m1 = sum(epoch_loss_m1) / len(epoch_loss)
    print("Loss = %.4f\t moving loss(current) = %.4f\t lr = %.6f" % (loss, loss_m1, lr))

    return loss, lr

def train_model(args):
    global network_type
    cudnn.enabled = True
    model = CPCNet.network1_2(nclasses=20, nclasses2=args.classes)
    network_type = "Moving-Net"

    args.savedir = (args.savedir + args.dataset + '/' + network_type + '_bs' + str(args.batch_size) +
                    'gpu' + args.gpus + '/')

    if not os.path.exists(args.savedir):
        os.makedirs(args .savedir)

    sys.stdout = Logger(os.path.join(args.savedir, args.logFile))

    print(args)
    print("=====> current architeture: Seg-Moving-Net")

    if args.cuda:
        print("=====> use gpu id: '{}'".format(args.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

    # 语义分割部分不参与训练
    if args.official == True:
        frozen_params = model.seg.parameters()
        for p in frozen_params:
            p.requires_grad = False

    print("=====> computing network parameters")
    total_paramters = netParams2(model)
    print("the number of parameters: " + str(total_paramters))

    # open data config file
    print("Opening data config file %s" % (args.data_cfg))
    DATA = yaml.safe_load(open(args.data_cfg, 'r'))

    # weights for loss (and bias)
    # 运动分割
    epsilon_w = 1e-3
    content = torch.zeros(args.classes, dtype=torch.float)
    for cl, freq in DATA["content"].items():
        x_cl = to_xentropy(DATA, cl)  # map actual class to xentropy class
        content[x_cl] += freq
    loss_w = 1 / (content + epsilon_w)  # get weights
    for x_cl, w in enumerate(loss_w):  # ignore the ones necessary to ignore
        if DATA["learning_ignore"][x_cl]:
            # don't weigh
            loss_w[x_cl] = 0
    print("Loss weights from moving content: ", loss_w.data)

    evaluator = iouEval(n_classes=args.classes, device='cuda', ignore=0)

    # define optimization criteria
    criterion = nn.NLLLoss(weight=loss_w)
    criterion_ls = Lovasz_softmax(ignore=0)
    criterion_sh = SoftmaxHeteroscedasticLoss()

    if args.cuda:
        if torch.cuda.device_count() > 1:
            print("torch.cuda.device_count()=", torch.cuda.device_count())
            args.gpu_nums = torch.cuda.device_count()
            model = torch.nn.DataParallel(model).cuda()  # multi-card data parallel
        else:
            print("single GPU for training")
            model = model.cuda()  # 1-card data parallel
        criterion = criterion.cuda()
        criterion_ls = criterion_ls.cuda()
        criterion_sh = criterion_sh.cuda()

    # 预先生成,semantic-kitti方式生成的imagery数据
    trainDatasets = KittiSemanticDataSetOff(mode='train', path=args.data_dir, dataset=args.datatype, nums=args.num_train)
    trainLoader = data.DataLoader(trainDatasets, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    valDatasets = KittiSemanticDataSetOff(mode='val', path=args.data_dir, dataset=args.datatype, nums=args.num_val)
    valLoader = data.DataLoader(valDatasets, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)
    print("{'Semantic-KITTI way, train': {%d}, 'val': {%d}}" % (len(trainDatasets), len(valDatasets)))

    # 加载官方提供的salsanext语义分割预训练模型
    if args.official == True:
        target_state = model.module.seg.state_dict()
        w_dict = torch.load(args.pretrained)
        check = w_dict['state_dict']
        for name, v in check.items():
            # Exclude multi GPU prefix
            # 多GPU训练相比单GPU多了前缀(module.)需去掉
            mono_name = name[7:]
            if mono_name not in target_state:
                # print(mono_name)
                continue
            try:
                target_state[mono_name].copy_(v)
            except RuntimeError:
                continue
        print('Successfully loaded SalsaNext semantic segmentation pretrained model')

    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            # ---- loading part pre-trained model
            checkpoint = torch.load(args.resume)
            model_dict = model.state_dict()
            checkpoint = {k: v for k, v in checkpoint['model'].items() if k in model_dict}
            model_dict.update(checkpoint)
            model.load_state_dict(model_dict)
            print("=====> loaded checkpoint '{}')".format(args.resume))
            # ----
        else:
            print("=====> no checkpoint found at '{}'".format(args.resume))

    model.train()
    cudnn.benchmark = True

    # ---- single learning rate
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=0.9, weight_decay=1e-4)
    steps_per_epoch = 1.0 * len(trainDatasets) / args.batch_size
    up_steps = int(steps_per_epoch)
    final_decay = 0.99 ** (1 / steps_per_epoch)
    scheduler = warmupLR(optimizer, args.lr, up_steps, 0.9, final_decay)
    # ----

    print('=====> beginning training')
    best_loss_val = 100
    print("Start Time : " + time.strftime('%m.%d.%H:%M:%S', time.localtime(time.time())))
    training_time = 0
    for epoch in range(start_epoch, args.max_epochs):
        # training
        print("\nEpoch : " + str(epoch) + ' Details')
        torch.cuda.synchronize()
        start_epoch_time = time.time()
        train(args, trainLoader, model, criterion, criterion_ls, criterion_sh, optimizer, epoch, DATA, evaluator, scheduler)

        if True:
            # validation
            loss_val = val(args, valLoader, model, criterion, criterion_ls, criterion_sh, epoch, DATA, evaluator)

            if (loss_val < best_loss_val):
                best_loss_val = loss_val
                model_file_name = args.savedir + '/model_best.pth'
                state = {"epoch": epoch + 1, "model": model.state_dict()}
                torch.save(state, model_file_name)

        model_file_name = args.savedir + '/model_' + str(epoch) + '.pth'
        state = {"epoch": epoch + 1, "model": model.state_dict()}
        torch.save(state, model_file_name)

        torch.cuda.synchronize()
        end_epoch_time = time.time()
        epoch_duration = end_epoch_time - start_epoch_time
        training_time += epoch_duration
        print('Have trained %.2f Hours, and %.2f Hours this epoch' % (training_time / 3600, epoch_duration / 3600))

if __name__ == '__main__':
    start = timeit.default_timer()
    parser = ArgumentParser()
    parser.add_argument('--model', default="Moving-Net", help="model name: Seg-Net")
    parser.add_argument('--dataset', default="kitti_moving", help="dataset: cityscapes or camvid")
    parser.add_argument('--data_dir', default="/test/Odometry/dataset/", help='data directory')
    parser.add_argument('--datatype', default="filepath_sm", help='filepath_sd or filepath_sd2')
    parser.add_argument('--max_epochs', type=int, default=30, help="the number of epochs")
    parser.add_argument('--num_workers', type=int, default=8, help=" the number of parallel threads")
    parser.add_argument('--batch_size', type=int, default=2, help="the batch size is set to 16 for 2 GPUs")
    parser.add_argument('--num_train', type=int, default=4000, help="the number of train frames")
    parser.add_argument('--num_val', type=int, default=0, help="the number of val frames")
    parser.add_argument('--lr', type=float, default=1e-2, help="initial learning rate")
    parser.add_argument('--data_cfg', default="./semantic-kitti-mos.yaml", help="Moving yaml cfg file.")
    parser.add_argument('--savedir', default="./checkpoint/", help="directory to save the model snapshot")
    parser.add_argument('--save_seg_dir', type=str, default="./result/kitti_moving/",
                        help="saving path of prediction result")
    parser.add_argument('--resume', type=str,
                        default="./checkpoint/kitti_moving/1/Moving-Net_bs20gpu0,1/2_1/model_best2.pth",
                        help="use this file to load last checkpoint for continuing training")
    parser.add_argument('--pretrained', type=str,
                        default="./pretrained/SalsaNext/SalsaNext",
                        help="use this file to load seg checkpoint")
    parser.add_argument('--official', default=True, help="use SalsaNext pretrained model")
    parser.add_argument('--iou', default=False, help="calculate train iou")
    parser.add_argument('--classes', type=int, default=3, help="the number of semantic classes")
    parser.add_argument('--logFile', default="log.txt", help="storing the training and validation logs")
    parser.add_argument('--cuda', type=bool, default=True, help="running on CPU or GPU")
    parser.add_argument('--multi', default=True, help="use multiple GPUs")
    parser.add_argument('--gpus', type=str, default="0,1", help="default GPU devices (0,1)")
    args = parser.parse_args()
    train_model(args)
    end = timeit.default_timer()
    print("training time:", 1.0 * (end - start) / 3600)