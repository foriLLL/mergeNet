import random
from PtrNet import PointerNet
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import RobertaModel
import numpy as np
from tqdm import tqdm



# 参数设置，ObjectDict是为了直接改之前代码
class ObjectDict:
  pass

params = ObjectDict()
params.batch_size = 32
params.nof_epoch = 20
params.lr = 0.01
params.gpu = True
params.embedding_size = 768
params.hiddens = 1024
params.nof_lstms = 4
params.dropout = 0.4
params.bidir = True
params.embed_batch = 1
params.bert_grad = False
params.codeBERT = './bert/CodeBERTa-small-v1'
params.save_path = './output/finalModel4MergeBertData.pt'

if params.gpu and torch.cuda.is_available():
    USE_CUDA = True
    print('Using GPU, %i devices.' % torch.cuda.device_count())
else:
    USE_CUDA = False
    print('Using CPU')

embed_model = RobertaModel.from_pretrained(params.codeBERT)

if not params.bert_grad:
    for param in embed_model.parameters():
        param.requires_grad_(False)

model = PointerNet(params.embedding_size,
                   params.hiddens,
                   params.nof_lstms,
                   params.dropout,
                   params.embed_batch,
                   embed_model,
                   params.bidir, )

dataset_path = './output/tokenized_output'
dataset = load_from_disk(dataset_path).with_format(type='torch')
# dataset.train_test_split(test_size=0.1)
dataset = dataset.shuffle(seed=random.randint(0, 100))['train']     # 注意这里和师兄代码有改动，师兄代码加载的DatasetDict没有splits

dataloader = DataLoader(dataset,
                        batch_size=params.batch_size,
                        shuffle=True,
                        num_workers=0)

if USE_CUDA:
    model.cuda()
    net = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

CCE = torch.nn.CrossEntropyLoss()
model_optim = optim.Adam(filter(lambda p: p.requires_grad,
                                model.parameters()),
                         lr=params.lr)

losses = []
max_len = 30


def max_index(tl):
    '''
    返回 list 最大值下标
    '''
    ret = 0
    max_prob = tl[0]
    for ii in range(1, len(tl)):
        if tl[ii] > max_prob:
            max_prob = tl[ii]
            ret = ii
    return ret



# 开始训练
for epoch in range(params.nof_epoch):
    batch_loss = []
    batch_acc = 0
    iterator = tqdm(dataloader, unit='Batch')

    for i_batch, sample_batched in enumerate(iterator):
        iterator.set_description('Epoch %i/%i' % (epoch + 1, params.nof_epoch))

        input_batch = sample_batched['input_ids']         # 32 * 30 * 512     tokenized 后的每一行
        att_batch = sample_batched['attention_mask']      # 32 * 30 * 512     tokenized 后的每一行 token 中有效的部分（01矩阵）
        target_batch = sample_batched['label']            # 32 * 30          resolution 的行号排序

        
       # 改成不用batch试试
        model = PointerNet(params.embedding_size,
                    params.hiddens,
                    params.nof_lstms,
                    params.dropout,
                    params.embed_batch,
                    embed_model,
                    params.bidir, )
        if USE_CUDA:
            input_batch = input_batch.cuda()
            att_batch = att_batch.cuda()
            target_batch = target_batch.cuda()
            model = model.cuda()

        zs = []
        # valid_len_batch = sample_batched['lines'] # bug
        for i in range(len(sample_batched['label'])):   # 0-32
            max_lines = max(sample_batched['label'][i]) + 1 # 拿到的是 lines 中<eos> 的索引 + 1
            # print(max_lines)
            probabilities, indices = model([input_batch[i:i+1, 0:max_lines], att_batch[i:i+1, 0:max_lines]]) # valid_len_batch
            z = torch.zeros(1,30,30)
            for i in range(probabilities.shape[1]):
                for j in range(probabilities.shape[1]):
                    z[0, i, j] = probabilities[0, i, j]
            zs.append(z)
        o = torch.concat(zs)


        # o, p = model([input_batch, att_batch])

        valid_len_batch = sample_batched['valid_len']               # label 加入终止符后的长度
        cur_batch_size = len(valid_len_batch)

        pred = torch.tensor([max_index(probs) for example in o for probs in example], dtype=torch.int64)
        pred = pred.view(cur_batch_size, max_len)
        if USE_CUDA:
            pred = pred.cuda()
        for i in range(len(valid_len_batch)):

            # print("valid length: ")
            # print(valid_len_batch[i])
            # print("prediction")
            # print(pred[i])
            # print('-'*40)
            # print('truth')
            # print(target_batch[i])
            # print('*'*20)

            if pred[i][0:valid_len_batch[i]].equal(target_batch[i][0:valid_len_batch[i]]):
                batched_lines = sample_batched['lines']
                batched_resolve = sample_batched['resolve']
                # print('*'*10 + 'ALL LINES' + '*'*10)
                # for line in [t[i] for t in batched_lines]:
                #     print(line)

                # print('*'*10 + 'GROUND TRUTH' + '*'*10)
                # for line in [t[i] for t in batched_resolve]:
                #     print(line)
                
                # print('*'*10 + 'PREDICTION' + '*'*10)
                # print(pred[i][0:valid_len_batch[i]])
                # print(target_batch[i][0:valid_len_batch[i]])

                batch_acc += 1

        mask_tensor = torch.zeros(size=(o.size()[:2]))
        for i in range(o.size()[0]):
            mask_tensor[i][0:valid_len_batch[i]] = 1

        if USE_CUDA:
            mask_tensor = mask_tensor.cuda()
            o = o.cuda()

        mask_tensor = mask_tensor.view(-1)

        o = o.contiguous().view(-1, o.size()[-1])
        target_batch = target_batch.view(-1)

        loss = CCE(o, target_batch)

        loss = torch.mul(loss, mask_tensor)
        loss = loss.sum() / valid_len_batch.sum()

        losses.append(loss.data)
        batch_loss.append(loss.data.cpu())

        model_optim.zero_grad()
        loss.backward()
        model_optim.step()

        iterator.set_postfix(loss='{}'.format(loss.data))

    print('Epoch {0} / {1}, average loss : {2} , average accuracy : {3}'.
          format(epoch + 1, params.nof_epoch, np.average(batch_loss), batch_acc / len(dataset)))

torch.save(model.state_dict(), params.save_path)
