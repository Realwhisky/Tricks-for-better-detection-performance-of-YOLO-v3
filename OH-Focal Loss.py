import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from lib.utils.box_utils import match, log_sum_exp


### 这里使用MultiBoxLoss类，后面计算分类损失的时候换损失函数 ###

class MultiBoxLoss(nn.Module):   
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, cfg, priors, use_gpu=True):
        super(MultiBoxLoss, self).__init__()
        self.use_gpu = use_gpu
        self.num_classes = cfg.NUM_CLASSES
        self.background_label = cfg.BACKGROUND_LABEL
        self.negpos_ratio = cfg.NEGPOS_RATIO
        self.threshold = cfg.MATCHED_THRESHOLD
        self.unmatched_threshold = cfg.UNMATCHED_THRESHOLD
        self.variance = cfg.VARIANCE
        self.priors = priors

        self.alpha = Variable(torch.ones(self.num_classes, 1) * 1)
        self.gamma = 2

    def forward(self, predictions, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)
            ground_truth (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """
        loc_data, conf_data = predictions
        num = loc_data.size(0)
        priors = self.priors
        # priors = priors[:loc_data.size(1), :]
        num_priors = (priors.size(0))
        num_classes = self.num_classes

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)
        for idx in range(num):
            truths = targets[idx][:,:-1].data
            labels = targets[idx][:,-1].data
            defaults = priors.data
            match(self.threshold,truths,defaults,self.variance,labels,loc_t,conf_t,idx)

        # print('loc_data',loc_data,'conf_data',conf_data,'num = loc_data.size(0)',num,'priors',priors,'loc_t ',loc_t,'conf_t',
        #       'truths','labels','self.variance',)

        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
        # wrap targets
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t,requires_grad=False)

        pos = conf_t > 0
        # num_pos = pos.sum()

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1,4)
        loc_t = loc_t[pos_idx].view(-1,4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1,1))

        # Hard Negative Mining
        loss_c = loss_c.view(num, -1)
        loss_c[pos] = 0 # filter out pos boxes for now
        _,loss_idx = loss_c.sort(1, descending=True)
        _,idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1,keepdim=True) #new sum needs to keep the same dim
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1,self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        pos_samp = conf_data[pos_idx].view(-1, self.num_classes)
        neg_samp = conf_data[neg_idx].view(-1, self.num_classes)
        pos_targ = conf_t[(pos).gt(0)].view(-1, 1)
        neg_targ = conf_t[(neg).gt(0)].view(-1, 1)


        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        loss_c = self.OH_FOCAL_loss(pos_samp, neg_samp, pos_targ,neg_targ)

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N

        N = num_pos.data.sum()#.double()
        loss_l= loss_l.double()
        loss_c= loss_c.double()
        loss_l/=N
        loss_c/=((1)*N)#.double()
        return loss_l,loss_c

    
    ### 定义的改动的Focal loss, 进一步精简了负样本的学习比例，主要是重点学习最难分类的难样本，而不是为了拟合所有样本（包括一些没有必要学习的样本）而牺牲模型的判断能力
    ### 防止正负样本不均衡，比例设为1:3，给难样本加了1.3的权重（实验结果）
    
    def OH_FOCAL_loss(self, inputspos, inputsneg, targetspos, targetsneg):
        '''
        weighted-OHEM-loss ---by whisky

        '''
        predt_pos = F.softmax(inputspos,dim=1)
        predt_neg = F.softmax(inputsneg,dim=1)

        AT = inputspos.size(0)
        AY = inputspos.size(1)
        AU = inputsneg.size(0)
        AN = inputsneg.size(1)

        class_maskpos = inputspos.data.new(AT, AY).fill_(0)
        class_maskpos = Variable(class_maskpos)
        idspos = targetspos.view(-1, 1)
        idspos = idspos.cuda()
     ###   print('class_maskpos=', class_maskpos, 'idspos=', idspos)
        class_maskpos.scatter_(1, idspos, 1.)

        class_maskneg = inputsneg.data.new(AU, AN).fill_(0)
        class_maskneg = Variable(class_maskneg)
        idsneg = targetsneg.view(-1, 1)
        idsneg = idsneg.cuda()
     ###   print(class_maskpos.scatter_(1, idspos, 1.),'class_maskneg=',class_maskneg,'idsneg=',idsneg)

        class_maskneg.scatter_(1, idsneg, 1.)

        # if inputspos. is_cuda and inputsneg. is_cuda and not self.alpha.is_cuda:
        # self.alpha = self.alpha.cuda()

        # alphapos = self.alpha[idspos.data.view(-1)]
        # alphapos = self.alpha[idsneg.data.view(-1)]

        probspos = (predt_pos * class_maskpos).sum(1).view(-1, 1)
        probsneg = (predt_neg * class_maskneg).sum(1).view(-1, 1)

        log_pp = probspos.log()
        log_pn = probsneg.log()

        # imd = torch.Tensor([[1], [2], [3]])
        # imds = imd.mean()
        # print('imds = ', imds)

      #  print('-(torch.pow((1 - probspos), self.gamma)) * log_pp = ',-(torch.pow((1 - probspos), self.gamma)) * log_pp)
      #  print(' 1.1 * (torch.pow((1 - probsneg), self.gamma)) * log_pn = ',  1.1 * (torch.pow((1 - probsneg), self.gamma)) * log_pn)

        batch_loss_pos = -(torch.pow((1 - probspos), self.gamma)) * log_pp
        batch_loss_neg = -(torch.pow((1 - probsneg), self.gamma)) * log_pn

        loss =  (batch_loss_pos.sum() +  1.3 * batch_loss_neg.sum())

        return loss
