# LiDAR Segmentation
# 与salsanext的差异,主要是数据读取,optimizer和学习率
# salsanext并未对网络权重进行初始化处理
# 雷达语义分割,训练数据为Odometry数据集
# filepath_s对应KITTI Semantic Depth数据集,雷达和语义真值

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
from dataset.kitti_semantic_bev import *  # dataset
from dataset.nuscenes_bev import NuscenesBevDataSet, collate_fn_BEV_nuscenes

# os.environ['CUDA_VISIBLE_DEVICES']='0,1'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# print(torch.__version__)
# print(torch.backends.cudnn.version())

color_map={
  "0" : [255, 255, 255],
  "1": [255, 0, 0],
  "11": [0, 60, 135],
  "3": [150, 60, 30],
  "4": [90, 30, 150],
  "5": [255, 0, 255],
  "8": [0, 200, 255],
  "10": [0, 175, 0],
  "6": [75, 0, 75],
  "7": [255, 150,255],
  "2": [30, 30, 255],
  "12": [80, 240, 150],
  "9": [50, 120, 255]
}

color_map2 ={
"0": [255, 255, 255],
"1": [144, 128, 112],
"2": [60, 20, 220],
"3": [80, 127, 255],
"4": [0, 158, 255],
"5": [70, 150, 233],
"6": [99, 61, 255],
"7": [230, 0, 0],
"8": [79, 79, 47],
"9": [0, 140, 255],
"10": [71, 99, 255],
"11": [191, 207, 0],
"12": [75, 0, 175],
"13": [75, 0, 75],
"14": [60, 180, 112],
"15": [135, 184, 222],
"16": [0, 175, 0]
}
def val(args, val_loader, model, criterion, criterion_ls, criterion_sh, epoch, DATA, evaluator):
    acc = AverageMeter()
    iou = AverageMeter()

    epoch_loss = []

    # evaluation mode
    model.eval()
    evaluator.reset()

    # empty the cache to infer in high res
    torch.cuda.empty_cache()

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            # imagery_velo, imagery_velo_std, imagery_seggt, bev_velo_std, bev_seggt, Tr, voxels, coordinates, num_points, obser,\
            #     demo_voxel_position, demo_grid, demo_pt_fea= batch
            if args.cyl == True:
                # bev_seggt,demo_voxel_position, demo_grid, demo_pt_fea, pc_xyz,velo_name,cy_fn= batch
                # # imagery_velo = Variable(imagery_velo).cuda()
                # # imagery_velo_std = Variable(imagery_velo_std).cuda()
                # # imagery_seggt = Variable(imagery_seggt.long()).cuda()
                # # bev_velo_std = Variable(bev_velo_std).cuda()
                
                # bev_seggt = Variable(bev_seggt.long()).cuda()
                # # bev_seggt = [torch.from_numpy(i).type(torch.LongTensor).cuda() for i in
                #                 # bev_seggt]
                # # Tr = Variable(Tr).cuda()
                # # voxels = Variable(voxels).cuda()
                # # coordinates = Variable(coordinates).cuda()
                # # num_points = Variable(num_points).cuda()
                # # obser = Variable(obser).cuda()
                # demo_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).cuda() for i in
                #                 demo_pt_fea]
                # demo_grid_ten = [torch.from_numpy(i).cuda() for i in demo_grid]
                # pc_xyz_ten = [torch.from_numpy(i).type(torch.FloatTensor).cuda() for i in pc_xyz]
                # demo_voxel_position = Variable(demo_voxel_position).cuda()
                # cy_fn = [torch.from_numpy(i).type(torch.FloatTensor).cuda() for i in cy_fn]
                # outputs_s2 = model(demo_voxel_position, demo_pt_fea_ten, demo_grid_ten, pc_xyz_ten, velo_name, cy_fn)
                
                # bev_seggt, pc_xyz,velo_name,cy_fn= batch
                # bev_seggt = Variable(bev_seggt.long()).cuda()
                # pc_xyz_ten = [torch.from_numpy(i).type(torch.FloatTensor).cuda() for i in pc_xyz]
                # cy_fn = [torch.from_numpy(i).type(torch.FloatTensor).cuda() for i in cy_fn]
                # outputs_s2 = model( pc_xyz_ten, velo_name, cy_fn)
                
                if args.nuscenes == True:
                    bev_seggt, demo_voxel_position, demo_grid, demo_pt_fea, pc_xyz, velo_name = batch
                    bev_seggt = Variable(bev_seggt.long()).cuda()
                    pc_xyz_ten = [torch.from_numpy(i).type(torch.FloatTensor).cuda() for i in pc_xyz]
                    cy_voxel_position = Variable(demo_voxel_position).cuda()
                    cy_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).cuda() for i in
                                    demo_pt_fea]
                    cy_grid_ten = [torch.from_numpy(i).cuda() for i in demo_grid]
                    outputs_s2 = model(pc_xyz_ten, velo_name, cy_fn=None, nuscenes=args.nuscenes, cy_voxel_position=cy_voxel_position,\
                        cy_pt_fea_ten=cy_pt_fea_ten, cy_grid_ten=cy_grid_ten)
                else:
                    bev_seggt, pc_xyz,velo_name,cy_fn, velo= batch
                    # bev_seggt, pc_xyz,velo_name,cy_fn, demo_voxel_position, demo_grid, demo_pt_fea, pc_xyz= batch
                    bev_seggt = Variable(bev_seggt.long()).cuda()
                    pc_xyz_ten = [torch.from_numpy(i).type(torch.FloatTensor).cuda() for i in pc_xyz]
                    cy_fn = [torch.from_numpy(i).type(torch.FloatTensor).cuda() for i in cy_fn]
                    velo = [torch.from_numpy(i).type(torch.FloatTensor).cuda() for i in velo]
                    pc_xyz_ten = [torch.from_numpy(i).type(torch.FloatTensor).cuda() for i in pc_xyz]
                    # cy_voxel_position = Variable(demo_voxel_position).cuda()
                    # cy_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).cuda() for i in
                    #                 demo_pt_fea]
                    # cy_grid_ten = [torch.from_numpy(i).cuda() for i in demo_grid]
                    # cy_voxel_position = Variable(demo_voxel_position).cuda()
                    # cy_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).cuda() for i in
                    #                 demo_pt_fea]
                    # cy_grid_ten = [torch.from_numpy(i).cuda() for i in demo_grid]
                    outputs_s2 = model( pc_xyz_ten, velo_name, cy_fn,velo=velo)
            else :
                imagery_velo, imagery_velo_std, imagery_seggt, bev_velo_std, bev_seggt, Tr, voxels, coordinates, num_points, obser,\
                demo_voxel_position, demo_grid, demo_pt_fea= batch #
                imagery_velo = Variable(imagery_velo).cuda()
                imagery_velo_std = Variable(imagery_velo_std).cuda()
                imagery_seggt = Variable(imagery_seggt.long()).cuda()
                bev_velo_std = Variable(bev_velo_std).cuda()
                bev_seggt = Variable(bev_seggt.long()).cuda()
                # imagery_velo_std= Variable(imagery_velo_std).cuda()
                Tr = Variable(Tr).cuda()
                voxels = Variable(voxels).cuda()
                coordinates = Variable(coordinates).cuda()
                num_points = Variable(num_points).cuda()
                obser = Variable(obser).cuda()
                
                outputs_s2 = model(imagery_velo, imagery_velo_std, bev_velo_std, Tr, voxels, coordinates, num_points, obser) #demo_pt_fea_ten, args.batch_size
            # 放大,裁切
            if args.nuscenes == False:
                if args.precision == 0.2:
                    outputs_s2 = F.interpolate(outputs_s2, scale_factor=2, mode='bilinear', align_corners=True)
                outputs_s2 = outputs_s2[:, :,  6:506, 12:1012]
            # else:
            #     outputs_s2 = F.interpolate(outputs_s2, scale_factor=2, mode='bilinear', align_corners=True)

            # bev_seggt = torch.cat(bev_seggt)
            loss_wce = criterion(torch.log(outputs_s2.clamp(min=1e-8)), bev_seggt)
            loss_jacc = criterion_ls(outputs_s2, bev_seggt)
            loss = loss_wce + loss_jacc

            epoch_loss.append(loss.item())

            # measure accuracy and record loss
            argmax_s2 = outputs_s2.argmax(dim=1)
            evaluator.addBatch(argmax_s2, bev_seggt)

            # ys, xs = torch.meshgrid(torch.arange(0, 500),torch.arange(0, 1000))
            # coord = torch.cat([xs.unsqueeze(-1), ys.unsqueeze(-1)], dim=-1).view((-1,2)).long()
            # coord = coord.data.cpu().numpy()
            # figure_path = '/opt/data/private/projects/copy/SemanticDepth/figure/'
            # # #/opt/data/common/SemanticKITTI/dataset/sequences/00/velodyne/000000.bin
            
            # for i in range(argmax_s2.shape[0]):
            #     label = argmax_s2[i].data.cpu().numpy()
            #     seg_gt = bev_seggt[i].data.cpu().numpy()
            #     img = np.zeros((500,1000,3))
            #     img1 = np.zeros((500,1000,3))
            #     for j in range(500):
            #         for k in range(1000):
            #             img[j,k,:] = color_map[str(label[j, k])]
            #             img1[j,k,:] = color_map[str(seg_gt[j, k])]
            #     velo_name_split = velo_name[i].split('/')
            #     seq_number = velo_name_split[-3]
            #     file_number = velo_name_split[-1].split('.')[0]
                
            #     cv2.imwrite(figure_path+seq_number+'/'+file_number+'.jpg', img)
            #     cv2.imwrite(figure_path+seq_number+'/'+file_number+'_gt.jpg', img1)
            # 结果可视化输出，仅20帧
            # if (i % 50 == 0):
            # # if True:
            #     output = outputs_s2.cpu().data[0].numpy()
            #     z_thr = np.asarray(np.argmax(output, axis=0), dtype=np.uint8)
            #     temp = bev_seggt.cpu().data[0].numpy()
            #     z_thr = np.hstack((np.uint8(temp), z_thr))
            #     z_fil = np.zeros([z_thr.shape[0], z_thr.shape[1], 3])
            #     for l in range(0, args.classes):
            #         z_fil[z_thr == l, :] = class_color_bev[l]
            #     z_img2 = Image.fromarray(np.uint8(z_fil))
            #     if not os.path.exists(args.save_seg_dir):
            #         os.makedirs(args.save_seg_dir)
            #     z_img2.save("%s/%s" % (args.save_seg_dir, str(i) + '.png'))

    accuracy = evaluator.getacc()
    jaccard, class_jaccard = evaluator.getIoU()
    acc.update(accuracy.item(), bev_seggt.size(0))
    iou.update(jaccard.item(), bev_seggt.size(0))

    print('Validation set:\n'
          'Acc avg {acc.avg:.3f}\n'
          'IoU avg {iou.avg:.3f}'.format(acc=acc, iou=iou))
    # print also classwise
    for i, jacc in enumerate(class_jaccard):
        print('IoU class {i:} [{class_str:}] = {jacc:.3f}'.format(
            i=i, class_str=DATA["labels"][i], jacc=jacc))

    loss = sum(epoch_loss) / len(epoch_loss)
    print("Loss = %.4f" % (loss))

    return loss

def adjust_learning_rate(args, cur_epoch, max_epoch, baselr):
    lr = baselr * pow((1 - 1.0 * cur_epoch / max_epoch), 0.9)
    return lr

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

def load_checkpoint(model_load_path, model):
    # print(torch.cuda.is_available())
    my_model_dict = model.state_dict()
    pre_weight = torch.load(model_load_path)

    part_load = {}
    match_size = 0
    nomatch_size = 0
    for k in pre_weight.keys():
        value = pre_weight[k]
        if k in my_model_dict and my_model_dict[k].shape == value.shape:
            # print("loading ", k)
            match_size += 1
            part_load[k] = value
        else:
            nomatch_size += 1
    print("matched parameter sets: {}, and no matched: {}".format(match_size, nomatch_size))

    my_model_dict.update(part_load)
    
    model.load_state_dict(my_model_dict)
    # my_model_dict = model.state_dict()
    # for key in my_model_dict:
    #     if "<INSERT KEYWORD>" in str(key):
    #         change_tensor = my_model_dict[key].clone()
    #         my_model_dict[key] = change_tensor.permute(0, 1, 3, 2, 4)
    # model.load_state_dict(my_model_dict)
    return model

def train(args, train_loader, model, criterion, criterion_ls, criterion_sh, optimizer, epoch, DATA, evaluator, scheduler):
    acc = AverageMeter()
    iou = AverageMeter()

    epoch_loss = []

    # empty the cache to infer in high res
    torch.cuda.empty_cache()

    model.train()
    evaluator.reset()

    total_batches = len(train_loader)
    print("=====> the number of iterations per epoch: ", total_batches)

    # lr = adjust_learning_rate(args, cur_epoch=epoch, max_epoch=args.max_epochs, baselr=args.lr)
    for param_group in optimizer.param_groups:
        # param_group['lr'] = lr
        lr = param_group['lr']

    for iteration, batch in enumerate(train_loader, 0):
        # imagery_velo, imagery_velo_std, imagery_seggt, bev_velo_std, bev_seggt, Tr, voxels, coordinates, num_points, obser,\
        #     demo_voxel_position, demo_grid, demo_pt_fea= batch #
        if args.cyl == True:
            # bev_seggt, demo_voxel_position, demo_grid, demo_pt_fea, pc_xyz, velo_name, cy_fn= batch
            # # imagery_velo = Variable(imagery_velo).cuda()
            # # imagery_velo_std = Variable(imagery_velo_std).cuda()
            # # imagery_seggt = Variable(imagery_seggt.long()).cuda()
            # # bev_velo_std = Variable(bev_velo_std).cuda()
            
            # bev_seggt = Variable(bev_seggt.long()).cuda()
            # # bev_seggt = [torch.from_numpy(i).type(torch.LongTensor).cuda() for i in
            # #                     bev_seggt]
            # # imagery_velo_std= Variable(imagery_velo_std).cuda()
            # # Tr = Variable(Tr).cuda()
            # # voxels = Variable(voxels).cuda()
            # # coordinates = Variable(coordinates).cuda()
            # # num_points = Variable(num_points).cuda()
            # # obser = Variable(obser).cuda()
            
            # demo_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).cuda() for i in
            #                     demo_pt_fea]
            # demo_grid_ten = [torch.from_numpy(i).cuda() for i in demo_grid]
            # pc_xyz_ten = [torch.from_numpy(i).type(torch.FloatTensor).cuda() for i in pc_xyz]
            # # demo_voxel_position = Variable(demo_voxel_position).cuda()
            # # demo_pt_fea_ten = Variable(demo_pt_fea).cuda()
            # # demo_grid_ten = Variable(demo_grid).cuda()
            # cy_fn = [torch.from_numpy(i).type(torch.FloatTensor).cuda() for i in cy_fn]
            # outputs_s2 = model(demo_voxel_position, demo_pt_fea_ten, demo_grid_ten, pc_xyz_ten, velo_name, cy_fn) #demo_pt_fea_ten, args.batch_size
            
            
            if args.nuscenes == True:
                bev_seggt, demo_voxel_position, demo_grid, demo_pt_fea, pc_xyz, velo_name = batch
                bev_seggt = Variable(bev_seggt.long()).cuda()
                pc_xyz_ten = [torch.from_numpy(i).type(torch.FloatTensor).cuda() for i in pc_xyz]
                cy_voxel_position = Variable(demo_voxel_position).cuda()
                cy_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).cuda() for i in
                                demo_pt_fea]
                cy_grid_ten = [torch.from_numpy(i).cuda() for i in demo_grid]
                outputs_s2 = model(pc_xyz_ten, velo_name, cy_fn=None, nuscenes=args.nuscenes, cy_voxel_position=cy_voxel_position,\
                    cy_pt_fea_ten=cy_pt_fea_ten, cy_grid_ten=cy_grid_ten)
            else:
                bev_seggt, pc_xyz,velo_name,cy_fn,velo= batch
                # bev_seggt, pc_xyz,velo_name,cy_fn, demo_voxel_position, demo_grid, demo_pt_fea, pc_xyz= batch
                bev_seggt = Variable(bev_seggt.long()).cuda()
                pc_xyz_ten = [torch.from_numpy(i).type(torch.FloatTensor).cuda() for i in pc_xyz]
                cy_fn = [torch.from_numpy(i).type(torch.FloatTensor).cuda() for i in cy_fn]
                velo = [torch.from_numpy(i).type(torch.FloatTensor).cuda() for i in velo]
                # pc_xyz_ten = [torch.from_numpy(i).type(torch.FloatTensor).cuda() for i in pc_xyz]
                
                # cy_voxel_position = Variable(demo_voxel_position).cuda()
                # cy_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).cuda() for i in
                #                 demo_pt_fea]
                # cy_grid_ten = [torch.from_numpy(i).cuda() for i in demo_grid]
                # cy_voxel_position = Variable(demo_voxel_position).cuda()
                # cy_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).cuda() for i in
                #                 demo_pt_fea]
                # cy_grid_ten = [torch.from_numpy(i).cuda() for i in demo_grid]
                outputs_s2 = model( pc_xyz_ten, velo_name, cy_fn,velo=velo)
        else :
            imagery_velo, imagery_velo_std, imagery_seggt, bev_velo_std, bev_seggt, Tr, voxels, coordinates, num_points, obser,\
                demo_voxel_position, demo_grid, demo_pt_fea= batch #
            imagery_velo = Variable(imagery_velo).cuda()
            imagery_velo_std = Variable(imagery_velo_std).cuda()
            imagery_seggt = Variable(imagery_seggt.long()).cuda()
            bev_velo_std = Variable(bev_velo_std).cuda()
            bev_seggt = Variable(bev_seggt.long()).cuda()
            # imagery_velo_std= Variable(imagery_velo_std).cuda()
            Tr = Variable(Tr).cuda()
            voxels = Variable(voxels).cuda()
            coordinates = Variable(coordinates).cuda()
            num_points = Variable(num_points).cuda()
            obser = Variable(obser).cuda()
            
            outputs_s2 = model(imagery_velo, imagery_velo_std, bev_velo_std, Tr, voxels, coordinates, num_points, obser) #demo_pt_fea_ten, args.batch_size
        # 放大,裁切
        if args.nuscenes == False:
            if args.precision == 0.2:
                outputs_s2 = F.interpolate(outputs_s2, scale_factor=2, mode='bilinear', align_corners=True)
            outputs_s2 = outputs_s2[:, :,  6:506, 12:1012]
        # else:
        #     outputs_s2 = F.interpolate(outputs_s2, scale_factor=2, mode='bilinear', align_corners=True)
        
        # bev_seggt = torch.cat(bev_seggt)
        loss_wce = criterion(torch.log(outputs_s2.clamp(min=1e-8)), bev_seggt)
        loss_jacc = criterion_ls(outputs_s2, bev_seggt)
        loss = loss_wce + loss_jacc
        # loss.requires_grad_(True)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss.append(loss.item())

        if args.iou == True:
            argmax_s2 = outputs_s2.argmax(dim=1)
            evaluator.addBatch(argmax_s2, bev_seggt)

        # argmax_s2 = outputs_s2.argmax(dim=1)
        # # ys, xs = torch.meshgrid(torch.arange(0, 500),torch.arange(0, 1000))
        # # coord = torch.cat([xs.unsqueeze(-1), ys.unsqueeze(-1)], dim=-1).view((-1,2)).long()
        # # coord = coord.data.cpu().numpy()
        # figure_path = '/opt/data/private/projects/copy/SemanticDepth/figure/'
        # # #/opt/data/common/SemanticKITTI/dataset/sequences/00/velodyne/000000.bin
        
        # for i in range(argmax_s2.shape[0]):
        #     label = argmax_s2[i].data.cpu().numpy()
        #     seg_gt = bev_seggt[i].data.cpu().numpy()
        #     img = np.zeros((500,1000,3))
        #     for j in range(500):
        #         for k in range(1000):
        #             img[j,k,:] = color_map[str(seg_gt[j, k])]
        #     velo_name_split = velo_name[i].split('/')
        #     seq_number = velo_name_split[-3]
        #     file_number = velo_name_split[-1].split('.')[0]
            
        #     cv2.imwrite(figure_path+seq_number+'/'+file_number+'.jpg', img)
        # step scheduler, 调整学习率,逐batch而不是逐epoch更新学习率
        scheduler.step()

    if args.iou == True:
        accuracy = evaluator.getacc()
        jaccard, class_jaccard = evaluator.getIoU()
        acc.update(accuracy.item(), bev_seggt.size(0))
        iou.update(jaccard.item(), bev_seggt.size(0))

        print('Training set:\n'
              'Acc avg {acc.avg:.3f}\n'
              'IoU avg {iou.avg:.3f}'.format(acc=acc, iou=iou))

    loss = sum(epoch_loss) / len(epoch_loss)
    print("Loss = %.4f\t lr = %.6f" % (loss, lr))

    return loss, lr

def train_model(args):
    global network_type
    cudnn.enabled = True
    # model = SalsaNext.network4(20, nclasses2=args.classes, height=256, width=512)
    if args.nuscenes == True:
        network_type = "Seg-Net-Bev"
        model = CPCNet.network4(20, nclasses2=args.classes, height=512, width=512, nuscenes=args.nuscenes)
    else:
        network_type = "Seg-Net-Bev"
        model = CPCNet.network4(20, nclasses2=args.classes, height=256, width=512, nuscenes=args.nuscenes)

    args.savedir = (args.savedir + args.dataset + '/' + network_type + '_bs' + str(args.batch_size) +
                    'gpu' + args.gpus + '/')

    if not os.path.exists(args.savedir):
        os.makedirs(args .savedir)

    sys.stdout = Logger(os.path.join(args.savedir, args.logFile))

    print(args)
    print("=====> current architeture: Seg-Net-Bev")
    
    # print(torch.cuda.is_available())
    # print(torch.cuda.device_count())
    if args.cuda:
        print("=====> use gpu id: '{}'".format(args.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        print(torch.__version__)
        print(torch.version.cuda)
        print(torch.cuda.is_available())
        if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

    # 语义分割部分不参与训练
    if args.official == True:
        frozen_params = model.seg.parameters()
        for p in frozen_params:
            p.requires_grad = False

    if args.cyl_pre == True:
        # model_load_path = './pretrained/Cylinder/model_save_backup.pt'
        # model = load_checkpoint(model_load_path, model)
        # frozen_params = model.seg.parameters()
        # for p in frozen_params:
        #     p.requires_grad = False
        frozen_params = model.cylinder_3d_spconv_seg.parameters()
        for p in frozen_params:
            p.requires_grad = False
        frozen_params = model.cylinder_3d_generator.parameters()
        for p in frozen_params:
            p.requires_grad = False

    print("=====> computing network parameters")
    total_paramters = netParams2(model)
    print("the number of parameters: " + str(total_paramters))

    # open data config file
    print("Opening data config file %s" % args.data_cfg)
    DATA = yaml.safe_load(open(args.data_cfg, 'r'))

    # weights for loss (and bias)
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
    if args.nuscenes == True:
        loss_w = torch.tensor([0,1,8,2,2,2,8,8,2,2,2,1,1,1,1,1,1], dtype=torch.float)
    else:
        loss_w = torch.tensor([0,2,8,8,8,1,1,1,1,1,1,1,1], dtype=torch.float)
    
    # loss_w = torch.tensor([0,8,2,2,2,8,8,1,2,2,1,1,1,1,1,1,1], dtype=torch.float)
    # loss_w = torch.tensor([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], dtype=torch.float)
    print("Loss weights from content: ", loss_w.data)

    # evaluator = iouEval(n_classes=args.classes, device='cuda')
    evaluator = iouEval(n_classes=args.classes, device='cuda', ignore=0)
    # define optimization criteria
    criterion = nn.NLLLoss(weight=loss_w)
    criterion_ls = Lovasz_softmax(ignore=0)
    criterion_sh = SoftmaxHeteroscedasticLoss()

    if args.cuda:
        # if torch.cuda.device_count() > 1:
        if args.gpus == "0,1":
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
    if args.nuscenes == True:
        trainDatasets = NuscenesBevDataSet(mode='train', path=args.data_dir, dataset=args.datatype, nums=args.num_train, height=args.height, width=args.width, precision=args.precision)
        trainLoader = data.DataLoader(trainDatasets, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True,collate_fn=collate_fn_BEV_nuscenes) #
        valDatasets = NuscenesBevDataSet(mode='val', path=args.data_dir, dataset=args.datatype, nums=args.num_val, height=args.height, width=args.width, precision=args.precision)
        valLoader = data.DataLoader(valDatasets, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False,collate_fn=collate_fn_BEV_nuscenes)
        print("{'Nuscenes way, train': {%d}, 'val': {%d}}" % (len(trainDatasets), len(valDatasets)))
    else:
        trainDatasets = KittiSemanticBevDataSetOff2(mode='train', path=args.data_dir, dataset=args.datatype, nums=args.num_train, height=args.height, width=args.width, precision=args.precision)
        trainLoader = data.DataLoader(trainDatasets, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True,collate_fn=collate_fn_BEV) #
        valDatasets = KittiSemanticBevDataSetOff2(mode='val', path=args.data_dir, dataset=args.datatype, nums=args.num_val, height=args.height, width=args.width, precision=args.precision)
        valLoader = data.DataLoader(valDatasets, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False,collate_fn=collate_fn_BEV)
        print("{'Semantic-KITTI way, train': {%d}, 'val': {%d}}" % (len(trainDatasets), len(valDatasets)))

    # 加载官方提供的salsanext语义分割预训练模型
    if args.official == True:
        if args.gpus == "0,1":
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
        else:
            target_state = model.seg.state_dict()
            w_dict = torch.load(args.pretrained)
            check = w_dict['state_dict']
            for name, v in check.items():
                # Exclude multi GPU prefix
                # 多GPU训练相比单GPU多了前缀(module.)需去掉
                mono_name = name
                if mono_name not in target_state:
                    # print(mono_name)
                    continue
                try:
                    target_state[mono_name].copy_(v)
                except RuntimeError:
                    continue
            print('Successfully loaded SalsaNext semantic segmentation pretrained model')
    
    if args.cyl_pre == True:
        if args.nuscenes == True:
            # model_load_path = './checkpoint/nuscenes_seg_bev/nuscenes_point_seg/Seg-Net-Bev_bs2gpu0/model_18.pth'
            model_load_path = './pretrained/Cylinder/model_save_backup.pt'
        else:
            model_load_path = './pretrained/Cylinder/model_save_backup.pt'
        if args.gpus == "0,1":
            my_model_dict = model.module.cylinder_3d_spconv_seg.state_dict()
        else:
            my_model_dict = model.cylinder_3d_spconv_seg.state_dict()
        
        pre_weight = torch.load(model_load_path)
        # if args.nuscenes == True:
        #     pre_weight = pre_weight['model']
        part_load = {}
        match_size = 0
        nomatch_size = 0
        for k in pre_weight.keys():
            # print(k)
            value = pre_weight[k]
            mono_name = k[23:]
            if mono_name in my_model_dict and my_model_dict[mono_name].shape == value.shape:
                # print("loading ", k)
                match_size += 1
                part_load[mono_name] = value
                try:
                    my_model_dict[mono_name].copy_(value)
                except RuntimeError:
                    continue
            else:
                nomatch_size += 1
                # print(mono_name)
        print("matched parameter sets1: {}, and no matched: {}".format(match_size, nomatch_size))
        
        if args.gpus == "0,1":
            my_model_dict = model.module.cylinder_3d_generator.state_dict()
        else:
            my_model_dict = model.cylinder_3d_generator.state_dict()
        
        part_load = {}
        match_size = 0
        nomatch_size = 0
        for k in pre_weight.keys():
            value = pre_weight[k]
            mono_name = k[22:]
            if mono_name in my_model_dict and my_model_dict[mono_name].shape == value.shape: #
                # print("loading ", k)
                match_size += 1
                part_load[mono_name] = value
                try:
                    my_model_dict[mono_name].copy_(value)
                except RuntimeError:
                    continue
            else:
                nomatch_size += 1
                # print(mono_name)
        print("matched parameter sets2: {}, and no matched: {}".format(match_size, nomatch_size))

        # my_model_dict.update(part_load)
        
        # model.module.cylinder_3d_spconv_seg.load_state_dict(my_model_dict)
        

    start_epoch = 0
    # debug
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
    # optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=0.9, weight_decay=1e-4)
    optimizer = torch.optim.AdamW(model.parameters(),args.lr, weight_decay=1e-4)
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

        # val(args, valLoader, model, criterion, criterion_ls, criterion_sh, epoch, DATA, evaluator)

        # training
        print("\nEpoch : " + str(epoch) + ' Details')
        torch.cuda.synchronize()
        start_epoch_time = time.time()
        train(args, trainLoader, model, criterion, criterion_ls, criterion_sh, optimizer, epoch, DATA, evaluator, scheduler)

        # validation
        loss_val = val(args, valLoader, model, criterion, criterion_ls, criterion_sh, epoch, DATA, evaluator)

        if (loss_val<best_loss_val):
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
    
    dataset='kitti'   #kitti, nuscenes
    if dataset == 'nuscenes':
        parser.add_argument('--dataset', default="nuscenes_seg_bev", help="dataset: cityscapes or camvid")
        parser.add_argument('--data_cfg', default="./nuscenes-bev.yaml", help="Classification yaml cfg file.")
        parser.add_argument('--height', default=512, help="height of the bev map, -50m-50m")
        parser.add_argument('--width', default=512, help="width of the bev map, -25m-25m")
        parser.add_argument('--savedir', default="./checkpoint2/nuscenes_seg_bev/", help="directory to save the model snapshot")
        parser.add_argument('--save_seg_dir', type=str, default="./checkpoint2/nuscenes_seg_bev/", help="saving path of prediction result")
        parser.add_argument('--nuscenes', default=True, help="use nuscenes dataset ")
        parser.add_argument('--cyl_pre', default=True, help="use cylinder pretrained model")
        parser.add_argument('--classes', type=int, default=17, help="the number of classes in the dataset. 19 and 11 for cityscapes and camvid, respectively")
        parser.add_argument('--batch_size', type=int, default=1, help="the batch size is set to 16 for 2 GPUs")
        parser.add_argument('--num_train', type=int, default=4000, help="the number of train frames")
        parser.add_argument('--num_val', type=int, default=0, help="the number of val frames")
        parser.add_argument('--resume', type=str,
                        default=None,
                        help="use this file to load last checkpoint for continuing training")
    else:
        parser.add_argument('--dataset', default="kitti_seg_bev", help="dataset: cityscapes or camvid")
        parser.add_argument('--data_cfg', default="./semantic-kitti-bev.yaml", help="Classification yaml cfg file.")
        parser.add_argument('--height', default=250, help="height of the bev map, -50m-50m")
        parser.add_argument('--width', default=500, help="width of the bev map, -25m-25m")
        parser.add_argument('--savedir', default="./checkpoint3/", help="directory to save the model snapshot")
        parser.add_argument('--save_seg_dir', type=str, default="./result/kitti_seg_bev/", help="saving path of prediction result")
        parser.add_argument('--nuscenes', default=False, help="use nuscenes dataset ")
        parser.add_argument('--cyl_pre', default=False, help="use cylinder pretrained model")
        parser.add_argument('--classes', type=int, default=13, help="the number of classes in the dataset. 19 and 11 for cityscapes and camvid, respectively")
        parser.add_argument('--batch_size', type=int, default=2, help="the batch size is set to 16 for 2 GPUs")
        parser.add_argument('--num_train', type=int, default=4000, help="the number of train frames")
        parser.add_argument('--num_val', type=int, default=0, help="the number of val frames")
        parser.add_argument('--resume', type=str,
                        # default='/opt/data/private/projects/copy/SemanticDepth/checkpoint3/kitti_seg_bev/cylinder/Seg-Net-Bev_bs2gpu0 0.491/model_6.pth',
                        default=None,
                        help="use this file to load last checkpoint for continuing training")
    parser.add_argument('--model', default="Seg-Net-Bev", help="model name: Seg-Net-Bev")
    # parser.add_argument('--dataset', default="kitti_seg_bev", help="dataset: cityscapes or camvid")
    parser.add_argument('--data_dir', default="/opt/data/common/SemanticKITTI/dataset/", help='data directory')
    parser.add_argument('--datatype', default="filepath_s", help='filepath_sd or filepath_sd2')
    parser.add_argument('--max_epochs', type=int, default=30, help="the number of epochs")
    parser.add_argument('--num_workers', type=int, default=8, help=" the number of parallel threads")
    # parser.add_argument('--batch_size', type=int, default=1, help="the batch size is set to 16 for 2 GPUs")
    # parser.add_argument('--num_train', type=int, default=6000, help="the number of train frames")
    # parser.add_argument('--num_val', type=int, default=0, help="the number of val frames")
    parser.add_argument('--lr', type=float, default=1e-2, help="initial learning rate")
    # parser.add_argument('--data_cfg', default="./semantic-kitti-bev.yaml", help="Classification yaml cfg file.")
    # parser.add_argument('--savedir', default="./checkpoint3/", help="directory to save the model snapshot")
    # parser.add_argument('--save_seg_dir', type=str, default="./result/kitti_seg_bev/", help="saving path of prediction result")
    # parser.add_argument('--resume', type=str,
    #                     default='/opt/data/private/projects/copy/SemanticDepth/checkpoint2/nuscenes_seg_bev/nuscenes_seg_bev/Seg-Net-Bev_bs1gpu0/model_best.pth',
    #                     help="use this file to load last checkpoint for continuing training")
    parser.add_argument('--pretrained', type=str,
                        default="./pretrained/SalsaNext/SalsaNext",
                        help="use this file to load seg checkpoint")
    parser.add_argument('--official', default=False, help="use SalsaNext pretrained model")
    parser.add_argument('--cyl', default=True, help="use cylinder model")
    # parser.add_argument('--cyl_pre', default=True, help="use cylinder pretrained model")
    parser.add_argument('--sparse', default=True, help="use sparse label train val")
    # parser.add_argument('--nuscenes', default=True, help="use nuscenes dataset ")
    parser.add_argument('--iou', default=False, help="calculate train iou")
    # parser.add_argument('--classes', type=int, default=13, help="the number of classes in the dataset. 19 and 11 for cityscapes and camvid, respectively")
    # parser.add_argument('--height', default=250, help="height of the bev map, -50m-50m")
    # parser.add_argument('--width', default=500, help="width of the bev map, -25m-25m")
    parser.add_argument('--precision', default=0.2, help="precision of the bev map")
    parser.add_argument('--logFile', default="log.txt", help="storing the training and validation logs")
    parser.add_argument('--cuda', type=bool, default=True, help="running on CPU or GPU")
    parser.add_argument('--multi', default=True, help="use multiple GPUs")
    parser.add_argument('--gpus', type=str, default="0", help="default GPU devices (0,1)")
    args = parser.parse_args()
    train_model(args)
    
    end = timeit.default_timer()
    print("training time:", 1.0 * (end - start) / 3600)