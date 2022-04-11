from __future__ import print_function
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import os
import math
import resnet as models
from matplotlib import pyplot as plt
from datetime import datetime
from my_dataset import filterdataset
from torchvision import transforms
from torch.utils.data import DataLoader

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Training settings
batch_size = 16
batch_size_test = 400
iteration = 15000
lr = [0.0015, 0.002]
seed = 8
momentum = 0.9
log_interval = 10
l2_decay = 5e-4
param = 0.5
class_num = 3
cuda = True
# root_path = "./dataset/"

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
ROOT_PATH_scr1 = os.path.join(r'E:\轴承故障实验数据集\一维数据集\实验台二（新）')
ROOT_PATH_scr2 = os.path.join(r'E:\轴承故障实验数据集\一维数据集\MFPT(new)')
ROOT_PATH_scr3 = os.path.join(r'E:\轴承故障实验数据集\一维数据集\PU')
ROOT_PATH_tar = os.path.join(r'E:\轴承故障实验数据集\一维数据集\实验台一')

source1_name = '实验台二'
source2_name = 'MFPT'
source3_name = 'PU'
target_name = '实验台一'

torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

trainTransform = transforms.Compose([
])

validTransform = transforms.Compose([
])

source1_data = filterdataset(data_dir=ROOT_PATH_scr1, transform=trainTransform)
source2_data = filterdataset(data_dir=ROOT_PATH_scr2, transform=trainTransform)
source3_data = filterdataset(data_dir=ROOT_PATH_scr3, transform=trainTransform)
target_data = filterdataset(data_dir=ROOT_PATH_tar, transform=validTransform)

source1_loader = DataLoader(dataset=source1_data, batch_size=batch_size, shuffle=True)
source2_loader = DataLoader(dataset=source2_data, batch_size=batch_size, shuffle=True)
source3_loader = DataLoader(dataset=source3_data, batch_size=batch_size, shuffle=True)
target_train_loader = DataLoader(dataset=target_data, batch_size=batch_size, shuffle=True)
target_test_loader = DataLoader(dataset=target_data, batch_size=batch_size_test, shuffle=True)

def train(model):
    source1_iter = iter(source1_loader)
    source2_iter = iter(source2_loader)
    source3_iter = iter(source3_loader)
    target_iter = iter(target_train_loader)
    correct = 0

    lmmd_loss1 = 0
    lmmd_loss2 = 0
    lmmd_loss3 = 0
    LEARNING_RATE = lr[1]
    optimizer = torch.optim.SGD([
            {'params': model.sharedNet.parameters()},
            {'params': model.cls_fc_son1.parameters(), 'lr': LEARNING_RATE},
            {'params': model.cls_fc_son2.parameters(), 'lr': LEARNING_RATE},
            {'params': model.cls_fc_son3.parameters(), 'lr': LEARNING_RATE},
            {'params': model.sonnet1.parameters(), 'lr': LEARNING_RATE},
            {'params': model.sonnet2.parameters(), 'lr': LEARNING_RATE},
            {'params': model.sonnet3.parameters(), 'lr': LEARNING_RATE},
        ], lr=LEARNING_RATE / 10, momentum=momentum, weight_decay=l2_decay)

    for i in range(1, iteration + 1):
        model.train()
        optimizer.param_groups[0]['lr'] = lr[0] / math.pow((1 + 10 * (i - 1) / (iteration)), 0.75)
        optimizer.param_groups[1]['lr'] = lr[1] / math.pow((1 + 10 * (i - 1) / (iteration)), 0.75)
        optimizer.param_groups[2]['lr'] = lr[1] / math.pow((1 + 10 * (i - 1) / (iteration)), 0.75)
        optimizer.param_groups[3]['lr'] = lr[1] / math.pow((1 + 10 * (i - 1) / (iteration)), 0.75)
        optimizer.param_groups[4]['lr'] = lr[1] / math.pow((1 + 10 * (i - 1) / (iteration)), 0.75)
        optimizer.param_groups[5]['lr'] = lr[1] / math.pow((1 + 10 * (i - 1) / (iteration)), 0.75)
        optimizer.param_groups[6]['lr'] = lr[1] / math.pow((1 + 10 * (i - 1) / (iteration)), 0.75)
        
        try:
            source_data, source_label = source1_iter.next()
        except Exception as err:
            source1_iter = iter(source1_loader)
            source_data, source_label = source1_iter.next()
        try:
            target_data, __ = target_iter.next()
        except Exception as err:
            target_iter = iter(target_train_loader)
            target_data, __ = target_iter.next()
        if cuda:
            source_data, source_label = source_data.cuda(), source_label.cuda()
            target_data = target_data.cuda()
        source_data, source_label = Variable(source_data), Variable(source_label)
        target_data = Variable(target_data)
        optimizer.zero_grad()

        cls_loss, lmmd_loss, l1_loss = model(source_data, target_data, source_label, mark=1)
        lmmd_loss1 += lmmd_loss
        lambd = 2 / (1 + math.exp(-10 * (i) / (iteration) )) - 1
        loss = cls_loss + param * lambd * lmmd_loss + lambd * l1_loss
        loss.backward()
        optimizer.step()

        if i % log_interval == 0:
            print('Train source1 iter: {} [({:.0f}%)]\tLoss: {:.6f}\tsoft_Loss: {:.6f}\tlmmd_Loss: {:.6f}\tl1_Loss: {:.6f}'.format(
                i, 100. * i / iteration, loss.item(), cls_loss.item(), lmmd_loss.item(), l1_loss.item()))

        try:
            source_data, source_label = source2_iter.next()
        except Exception as err:
            source2_iter = iter(source2_loader)
            source_data, source_label = source2_iter.next()
        try:
            target_data, __ = target_iter.next()
        except Exception as err:
            target_iter = iter(target_train_loader)
            target_data, __ = target_iter.next()
        if cuda:
            source_data, source_label = source_data.cuda(), source_label.cuda()
            target_data = target_data.cuda()
        source_data, source_label = Variable(source_data), Variable(source_label)
        target_data = Variable(target_data)
        optimizer.zero_grad()

        cls_loss, lmmd_loss, l1_loss = model(source_data, target_data, source_label, mark=2)

        lmmd_loss2 += lmmd_loss

        lambd = 2 / (1 + math.exp(-10 * (i) / (iteration) )) - 1
        loss = cls_loss + param * lambd * lmmd_loss + lambd * l1_loss
        loss.backward()
        optimizer.step()

        if i % log_interval == 0:
            print(
                'Train source2 iter: {} [({:.0f}%)]\tLoss: {:.6f}\tsoft_Loss: {:.6f}\tlmmd_Loss: {:.6f}\tl1_Loss: {:.6f}'.format(
                    i, 100. * i / iteration, loss.item(), cls_loss.item(), lmmd_loss.item(), l1_loss.item()))


        try:
            source_data, source_label = source3_iter.next()
        except Exception as err:
            source3_iter = iter(source3_loader)
            source_data, source_label = source3_iter.next()
        try:
            target_data, __ = target_iter.next()
        except Exception as err:
            target_iter = iter(target_train_loader)
            target_data, __ = target_iter.next()
        if cuda:
            source_data, source_label = source_data.cuda(), source_label.cuda()
            target_data = target_data.cuda()
        source_data, source_label = Variable(source_data), Variable(source_label)
        target_data = Variable(target_data)
        optimizer.zero_grad()

        cls_loss, lmmd_loss, l1_loss = model(source_data, target_data, source_label, mark=3)
        lmmd_loss3 += lmmd_loss
        lambd = 2 / (1 + math.exp(-10 * (i) / (iteration))) - 1
        loss = cls_loss + param * lambd * lmmd_loss + lambd * l1_loss
        loss.backward()
        optimizer.step()


        if i % log_interval == 0:
            print(
                'Train source3 iter: {} [({:.0f}%)]\tLoss: {:.6f}\tsoft_Loss: {:.6f}\tlmmd_Loss: {:.6f}\tl1_Loss: {:.6f}'.format(
                    i, 100. * i / iteration, loss.item(), cls_loss.item(), lmmd_loss.item(), l1_loss.item()))

        if i % (log_interval * 20) == 0:  #20
            #----add
            print(lmmd_loss1.item(), lmmd_loss2.item(), lmmd_loss3.item())
            lmmd_loss1 = 0
            lmmd_loss2 = 0
            lmmd_loss3 = 0

            t_correct = tst(model)
            if t_correct > correct:
                correct = t_correct
            print(source1_name, source2_name, source3_name, "to", target_name, "%s max correct:" % target_name, correct.item(), "\n")

            print('\n{} set:  MaxAccuracy: {}/{} ({:.2f}%)\n'.format(
                target_name, correct, len(target_test_loader.dataset),
                100. * correct / len(target_test_loader.dataset)))

#------------------------t-sne----------------------start----------
result_dir = os.path.join("Result")

now_time = datetime.now()
time_str = datetime.strftime(now_time, '%m-%d_%H-%M-%S')

log_dir = os.path.join(result_dir, time_str)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

cnt = 0       # 记录可视化图片的个数

def plot_with_labels(lowDWeights, labels, cnt):
    plt.cla()
    X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    for x, y, s in zip(X, Y, labels):
        c = cm.rainbow(int(255 * s / 3));
        plt.text(x, y,' ', backgroundcolor=c, fontsize=7)

    min_ = min(X.min(), Y.min())
    max_ = max(X.max(), Y.max())
    plt.xlim(min_-1, max_+1);
    plt.ylim(min_-1, max_+1);

    plt.xticks([])  # 不显示x轴
    plt.yticks([])

    str_cnt = str(cnt)       # 数字转字符串
    plt.savefig(os.path.join(log_dir, 'Visualize' + str_cnt + '.png'))
    #plt.pause(0.01)
    plt.close()


from matplotlib import cm

try:
    from sklearn.manifold import TSNE; HAS_SK = True
except:
    HAS_SK = False; print('Please install sklearn for layer visualization')
#------------------------t-sne----------------------end---

def tst(model):
    model.eval()
    test_loss = 0
    correct = 0
    correct1 = 0
    correct2 = 0
    correct3 = 0
    with torch.no_grad():
        count = 0
        for data, target in target_test_loader:
            count += 1
            if cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            pred1, pred2, pred3, last_layer = model(data)

            pred1 = torch.nn.functional.softmax(pred1, dim=1)
            pred2 = torch.nn.functional.softmax(pred2, dim=1)
            pred3 = torch.nn.functional.softmax(pred3, dim=1)

            pred = (pred1 + pred2 + pred3) / 3
            test_loss += F.nll_loss(F.log_softmax(pred, dim=1), target).item()  # sum up batch loss
            pred = pred.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            pred = pred1.data.max(1)[1]  # get the index of the max log-probability
            correct1 += pred.eq(target.data.view_as(pred)).cpu().sum()
            pred = pred2.data.max(1)[1]  # get the index of the max log-probability
            correct2 += pred.eq(target.data.view_as(pred)).cpu().sum()
            pred = pred3.data.max(1)[1]  # get the index of the max log-probability
            correct3 += pred.eq(target.data.view_as(pred)).cpu().sum()


            # # t-sne
            # tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
            # plot_only = 500
            # low_dim_embs = tsne.fit_transform(last_layer.cpu().data.numpy()[:plot_only, :])
            # labels = target.cpu().numpy()[:plot_only]
            # global cnt  # 用于命名可视化图片
            # plot_with_labels(low_dim_embs, labels, cnt)
            # cnt += 1


        test_loss /= len(target_test_loader.dataset)

        print(target_name, '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(target_test_loader.dataset),
            100. * correct / len(target_test_loader.dataset)))
        print('\nsource1 accnum {}, source2 accnum {}，source3 accnum {}'.format(correct1, correct2, correct3))


    return correct

if __name__ == '__main__':
    model = models.MSDAN(num_classes=class_num)
    if cuda:
        model.cuda()
    train(model)

