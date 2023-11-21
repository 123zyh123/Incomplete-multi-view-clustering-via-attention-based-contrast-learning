import argparse
import time
import torch.nn.functional as F
import logging
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from models import *
from Clustering import Clustering
from sure_inference import both_infer
from data_loader import loader
from loss import *
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


"""
(1) partially aligned, --settings 0 --aligned-prop 0.5 --complete-prop 1.0
(2) fully aligned, --settings 0 --aligned-prop 1.0 --complete-prop 1.0
(3) Incomplete, --settings 1 --aligned-prop 1.0 --complete-prop 0.5
(4) complete, --settings 1 --aligned-prop 1.0 --complete-prop 1.0
(5)PVP+PSP, --settings 2 --aligned-prop 0.5 --complete-prop 0.5
"""

parser = argparse.ArgumentParser(description='MCAC in PyTorch')
parser.add_argument('--data', default='0', type=int,
                    help='choice of dataset, 0-Scen e15, 1-Caltech101, 2-Reuters_dim10, '
                         '3-NoisyMNIST-30000, 4-2view-caltech101-8677sample,5-AWA-7view-10158sample, 6-MNIST-USPS')
parser.add_argument('-li', '--log-interval', default='1', type=int, help='interval for logging info')
parser.add_argument('-bs', '--batch-size', default='1024', type=int, help='number of batch size')
parser.add_argument('-e', '--epochs', default='80', type=int, help='number of epochs to run')
parser.add_argument('-lr', '--learn-rate', default='1e-3', type=float, help='learning rate of adam')
parser.add_argument('--lam', default='0.5', type=float, help='hyper-parameter between losses')
parser.add_argument('--lac', default='0.5', type=float, help='contract losses')
parser.add_argument('-noise', '--noisy-training', type=bool, default=True,
                    help='training with real labels or noisy labels,是否使用嘈杂标签训练，0-no,1-yes')
parser.add_argument('-np', '--neg-prop', default='30', type=int, help='the ratio of negative to positive pairs 30')
parser.add_argument('-m', '--margin', default='5', type=int, help='initial margin')
parser.add_argument('--gpu', default='0', type=str, help='GPU device idx to use.')
parser.add_argument('-r', '--robust', default=True, type=bool, help='use our robust loss or not')
parser.add_argument('-t', '--switching-time', default=1.0, type=float, help='start fine when neg_dist>=t*margin')
parser.add_argument('-s', '--start-fine', default=False, type=bool,
                    help='flag to start use robust loss or not,开始使用稳健损失的标志')
parser.add_argument('--settings', default=1, type=int, help='0-PVP, 1-PSP, 2-Both')
parser.add_argument('-ap', '--aligned-prop', default='1.0', type=float,
                    help='originally aligned proportions in the partially view-unaligned data,'
                         '部分视图未对齐数据中的原始对齐比例->用于训练的已知对齐比例')
parser.add_argument('-cp', '--complete-prop', default='1.0', type=float,
                    help='originally complete proportions in the partially sample-missing data,'
                         '部分样本缺失数据中的原始完整比例->完整的训练比例')

args = parser.parse_args()
print("==========\nArgs:{}\n==========".format(args))
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# mean distance of four kinds of pairs, namely, pos., neg., true neg., and false neg. (noisy labels)
# 四种对的平均距离，即pos.、neg.、true neg.和false neg。 （嘈杂的标签
pos_dist_mean_list, neg_dist_mean_list, true_neg_dist_mean_list, false_neg_dist_mean_list = [], [], [], []


# L_ncl损失
class NoiseRobustLoss(nn.Module):
    def __init__(self):
        super(NoiseRobustLoss, self).__init__()

    def forward(self, pair_dist, P, margin, use_robust_loss, args):
        # L_pos损失
        dist_sq = pair_dist * pair_dist
        P = P.to(torch.float32)
        N = len(P)
        # 使用Noise-robust损失
        if use_robust_loss == 1:
            if args.start_fine:

                loss = P * dist_sq + (1 - P) * (1 / margin) * torch.pow(
                    torch.clamp(torch.pow(pair_dist, 0.5) * (margin - pair_dist), min=0.0), 2)
            else:

                loss = P * dist_sq + (1 - P) * torch.pow(torch.clamp(margin - pair_dist, min=0.0), 2)
        else:
            loss = P * dist_sq + (1 - P) * torch.pow(torch.clamp(margin - pair_dist, min=0.0), 2)
        loss = torch.sum(loss) / (2 * N)
        return loss


def get_kNNgraph2(data, K_num):
    # each row of data is a sample
    data = data.cpu().numpy()
    x_norm = np.reshape(np.sum(np.square(data), 1), [-1, 1])  # column vector
    x_norm2 = np.reshape(np.sum(np.square(data), 1), [1, -1])  # column vector
    dists = x_norm - 2 * np.matmul(data, np.transpose(data)) + x_norm2
    num_sample = data.shape[0]
    graph = np.zeros((num_sample, num_sample), dtype=np.int)
    for i in range(num_sample):
        distance = dists[i, :]
        small_index = np.argsort(distance)
        graph[i, small_index[0:K_num]] = 1
    graph = graph - np.diag(np.diag(graph))
    resultgraph = np.maximum(graph, np.transpose(graph))
    return resultgraph


def train(train_loader, model, criterion, optimizer, epoch, args):
    pos_dist = 0  # mean distance of pos. pairs
    neg_dist = 0
    false_neg_dist = 0  # mean distance of false neg. pairs (pairs in noisy labels)
    true_neg_dist = 0
    pos_count = 0  # count of pos. pairs
    neg_count = 0
    false_neg_count = 0  # count of neg. pairs (pairs in noisy labels)
    true_neg_count = 0

    if epoch % args.log_interval == 0:
        logging.info("=======> Train epoch: {}/{}".format(epoch, args.epochs))
    model.train()
    time0 = time.time()
    ncl_loss_value = 0
    ver_loss_value = 0
    cross_value = 0

    for batch_idx, (x0, x1, labels, real_labels) in enumerate(train_loader):
        # labels refer to noisy labels for the constructed pairs, while real_labels are the clean labels for these pairs
        # labels是指构造对的噪声标签，而 real_labels 是这些对的干净标签
        x0, x1, labels, real_labels = x0.to(device), x1.to(device), labels.to(device), real_labels.to(device)
        x0 = x0.view(x0.size()[0], -1)
        x1 = x1.view(x1.size()[0], -1)

        try:
            h0, h1, z0, z1 = model(x0, x1)
        except:
            print("error raise in batch", batch_idx)

        pair_dist = F.pairwise_distance(h0, h1)  # use Euclidean distance to measure similarity,使用欧几里得距离来衡量相似度
        pos_dist += torch.sum(pair_dist[labels == 1])
        neg_dist += torch.sum(pair_dist[labels == 0])
        true_neg_dist += torch.sum(pair_dist[torch.logical_and(labels == 0, real_labels == 0)])
        false_neg_dist += torch.sum(pair_dist[torch.logical_and(labels == 0, real_labels == 1)])
        pos_count += len(pair_dist[labels == 1])
        neg_count += len(pair_dist[labels == 0])
        true_neg_count += len(pair_dist[torch.logical_and(labels == 0, real_labels == 0)])
        false_neg_count += len(pair_dist[torch.logical_and(labels == 0, real_labels == 1)])

        # 对比学习
        cross = crossview_contrastive_Loss(F.softmax(h0, dim=1), F.softmax(h1, dim=1))
        # 抗噪对比
        ncl_loss = criterion[0](pair_dist, labels, args.margin, args.robust, args)
        # 编码解码的重构损失
        ver_loss = criterion[1](x0, z0) + criterion[1](x1, z1)
        # 分类对比损失
        ccl_loss = category_contrastive_loss(z0, real_labels, len(real_labels)) + category_contrastive_loss(z1,
                                                                                                            real_labels,
                                                                                                            len(
                                                                                                                real_labels))

        # 总损失
        loss = ncl_loss + args.lac * ver_loss + 0.5 * (cross+ccl_loss)

        # 得到两部分损失值
        ncl_loss_value += ncl_loss.item()
        ver_loss_value += ver_loss.item()
        cross_value += cross.item()

        if epoch != 0:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    epoch_time = time.time() - time0

    pos_dist /= pos_count
    neg_dist /= neg_count
    true_neg_dist /= true_neg_count
    false_neg_dist /= false_neg_count
    if epoch != 0 and args.robust and neg_dist >= args.switching_time * args.margin and not args.start_fine:
        # start fine when the mean distance of neg. pairs is greater than switching_time * margin
        args.start_fine = True
        logging.info("******* neg_dist_mean >= {} * margin, start using fine loss at epoch: {} *******"
                     .format(args.switching_time, epoch + 1))

    # margin = the pos. distance + neg. distance before training
    if epoch == 0 and args.margin != 1.0:
        args.margin = max(1, round((pos_dist + neg_dist).item()))
        logging.info("margin = {}".format(args.margin))

    if epoch % args.log_interval == 0:
        logging.info("dist: P = {}, N = {}, TN = {}, FN = {},loss={}; time = {} s"
                     .format(round(pos_dist.item(), 2), round(neg_dist.item(), 2),
                             round(true_neg_dist.item(), 2), round(false_neg_dist.item(), 2),
                             loss, round(epoch_time, 2)))

    return pos_dist, neg_dist, false_neg_dist, true_neg_dist, epoch_time


def plot(acc, nmi, ari, args, data_name):
    x = range(0, args.epochs + 1, 1)
    fig_clustering = plt.figure()
    ax_clustering = fig_clustering.add_subplot(1, 1, 1)
    ax_clustering.set_title(data_name + ", " + "Noise=" + str(args.noisy_training) + ", RobustLoss=" + str(
        int(args.robust) * args.switching_time) + ", neg_prop=" + str(args.neg_prop))
    lns1 = ax_clustering.plot(x, acc, label='acc')
    lns2 = ax_clustering.plot(x, ari, label='ari')
    lns3 = ax_clustering.plot(x, nmi, label='nmi')
    lns = lns1 + lns2 + lns3
    labs = [l.get_label() for l in lns]
    ax_clustering.legend(lns, labs, loc=0)
    ax_clustering.grid()
    ax_clustering.set_xlabel("epoch(s)")
    ax_clustering.plot()

    fig_dist = plt.figure()
    ax_dist_mean = fig_dist.add_subplot(1, 1, 1)
    ax_dist_mean.set_title(data_name + ", " + "Noise=" + str(args.noisy_training) + ", RobustLoss=" + str(
        int(args.robust) * args.switching_time) + ", neg_prop=" + str(args.neg_prop))
    lns1 = ax_dist_mean.plot(x, pos_dist_mean_list, label='pos. dist')
    lns2 = ax_dist_mean.plot(x, neg_dist_mean_list, label='neg. dist')
    lns3 = ax_dist_mean.plot(x, false_neg_dist_mean_list, label='false neg. dist')
    lns4 = ax_dist_mean.plot(x, true_neg_dist_mean_list, label='true neg. dist')
    lns = lns1 + lns2 + lns3 + lns4
    labs = [l.get_label() for l in lns]
    ax_dist_mean.legend(lns, labs, loc=0)
    ax_dist_mean.grid()
    ax_dist_mean.set_xlabel("epoch(s)")
    plt.show()


def main():  # deep features of Caltech101
    data_name = ['Scene15', 'Caltech101', 'Reuters_dim10', 'NoisyMNIST-30000', '2view-caltech101-8677sample',
                 'AWA-7view-10158sample', 'MNIST-USPS']
    NetSeed = 64
    # random.seed(NetSeed)
    np.random.seed(NetSeed)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(NetSeed)
    torch.cuda.manual_seed(NetSeed)
    # 数据加载
    train_pair_loader, all_loader, divide_seed = loader(args.batch_size, args.neg_prop, args.aligned_prop,
                                                        args.complete_prop, args.noisy_training,
                                                        data_name[args.data])

    if args.data == 0:
        model = SUREfcScene().to(device)
    elif args.data == 1:
        model = SUREfcCaltech().to(device)
    elif args.data == 2:
        model = SUREfcReuters().to(device)
    elif args.data == 3:
        model = SUREfcNoisyMNIST().to(device)
    elif args.data == 4:
        model = SUREfcDeepCaltech().to(device)
    elif args.data == 5:
        model = SUREfcDeepAnimal().to(device)
    elif args.data == 6:
        model = SUREfcMNISTUSPS().to(device)

    criterion_ncl = NoiseRobustLoss().to(device)

    # 均方差损失
    criterion_mse = nn.MSELoss().to(device)

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learn_rate)

    # cls_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 30, 60],gamma=0.1)

    if not os.path.exists('./log/'):
        os.mkdir("./log/")
        if not os.path.exists('./log/' + str(data_name[args.data]) + '/'):
            os.mkdir('./log/' + str(data_name[args.data]) + '/')
    # path = os.path.join("./log/" + str(data_name[args.data]) + "/" + 'time=' + time
    #                     .strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    log_format = '%(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler('./log/' + str(data_name[args.data]) + '.txt')
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    file = open('log/' + str(data_name[args.data]) + '.txt', 'w').close()
    logging.info("******** Training begin ********")

    acc_list, nmi_list, ari_list = [], [], []
    best_list = {}
    acc = {}
    train_time = 0
    # train
    for epoch in range(0, args.epochs + 1):

        if epoch == 0:
            with torch.no_grad():
                pos_dist_mean, neg_dist_mean, false_neg_dist_mean, true_neg_dist_mean, epoch_time = \
                    train(train_pair_loader, model, [criterion_ncl, criterion_mse], optimizer, epoch, args)
        else:
            pos_dist_mean, neg_dist_mean, false_neg_dist_mean, true_neg_dist_mean, epoch_time = \
                train(train_pair_loader, model, [criterion_ncl, criterion_mse], optimizer, epoch, args)
        train_time += epoch_time
        pos_dist_mean_list.append(pos_dist_mean.item())
        neg_dist_mean_list.append(neg_dist_mean.item())
        true_neg_dist_mean_list.append(true_neg_dist_mean.item())
        false_neg_dist_mean_list.append(false_neg_dist_mean.item())

        v0, v1, gt_label = both_infer(model, device, all_loader, args.settings)
        data = [v0, v1]
        y_pred, ret = Clustering(data, gt_label)

        if epoch % args.log_interval == 0:
            # logging.info("******** testing ********")
            logging.info(
                "Clustering: 'acc':{}, 'nmi':{}, 'ari':{}".format(ret['kmeans']['accuracy'],
                                                                  ret['kmeans']['NMI'], ret['kmeans']['ARI']))
            # acc_list.append(ret['kmeans']['accuracy'])
            # acc_list.sort()
            # print('-' * 50 + 'best acc', max(acc_list))

        # best_list[epoch]=ret['kmeans']
        # acc[epoch]=ret['kmeans']['accuracy']
        #
        # max_acc = max(zip(acc.values(), acc.keys()))
        acc_list.append(ret['kmeans']['accuracy'])
        nmi_list.append(ret['kmeans']['NMI'])
        ari_list.append(ret['kmeans']['ARI'])

    # logging.info('********* best {}'.format(best_list[max_acc[1]]))
    # print('-' * 50 + '平均 acc', sum(acc_list[-5:]) / 5)
    # plot(acc_list, nmi_list, ari_list, args, data_name[args.data])
    logging.info('******** End, training time = {} s ********'.format(round(train_time, 2)))


if __name__ == '__main__':
    main()
