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


params = dict()
params['batch_size'] = 32
params['nof_epoch'] = 30
params['lr'] = 0.01
params['gpu'] = True
params['embedding_size'] = 768
params['hiddens'] = 1024
params['nof_lstms'] = 4
params['dropout'] = 0.4
params['bidir'] = True
params['embed_batch'] = 1
params['bert_grad'] = False
params['codeBERT'] = './bert/CodeBERTa-small-v1'
params['save_path'] = './output/finalModel4MergeBertData.pt'
params['dataset_path'] = './output/tokenized_output'                # tokenized dataset path

# 加载数据集
dataset = load_from_disk(params['dataset_path']).with_format(type='torch')
# dataset.train_test_split(test_size=0.1)
dataset = dataset.shuffle(seed=random.randint(0, 100))['train']     # 注意这里和师兄代码有改动，师兄代码加载的DatasetDict没有splits
dataloader = DataLoader(dataset,
                        batch_size=params['batch_size'],
                        shuffle=True,
                        num_workers=0)


embed_model = RobertaModel.from_pretrained(params['codeBERT'])
if not params['bert_grad']:
    for param in embed_model.parameters():
        param.requires_grad_(False)

model = PointerNet(params['embedding_size'],
                   params['hiddens'],
                   params['nof_lstms'],
                   params['dropout'],
                   params['embed_batch'],
                   embed_model,
                   params['bidir'])

dataset_path = './output/tokenized_output'
dataset = load_from_disk(dataset_path).with_format(type='torch')
# dataset.train_test_split(test_size=0.1)

# dataset = dataset.shuffle(seed=random.randint(0, 100))['train']     # 注意这里和师兄代码有改动，师兄代码加载的DatasetDict没有splits
dataset = dataset.shuffle(seed=1)['train']     # 注意这里和师兄代码有改动，师兄代码加载的DatasetDict没有splits

if params['gpu'] and torch.cuda.is_available():
    USE_CUDA = True
    model.cuda()
    net = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True
    print('Using GPU, %i devices.' % torch.cuda.device_count())
else:
    USE_CUDA = False
    print('Using CPU')


CCE = torch.nn.CrossEntropyLoss()
model_optim = optim.Adam(filter(lambda p: p.requires_grad,
                                model.parameters()),
                         lr=params['lr'])


# 开始训练
losses = []
max_len = 30 
for epoch in range(params['nof_epoch']):
    batch_loss = []
    batch_acc = 0
    iterator = tqdm(dataloader, unit='Batch')

    for i_batch, sample_batched in enumerate(iterator):
        iterator.set_description('Epoch %i/%i' % (epoch + 1, params['nof_epoch']))

        input_batch = sample_batched['input_ids']         # 32 * 30 * 512     tokenized 后的每一行
        att_batch = sample_batched['attention_mask']      # 32 * 30 * 512     tokenized 后的每一行 token 中有效的部分（01矩阵）
        target_batch = sample_batched['label']            # 32 * 30          resolution 的行号排序
        
       # 改成不用batch试试
        model = PointerNet(params['embedding_size'],
                    params['hiddens'],
                    params['nof_lstms'],
                    params['dropout'],
                    params['embed_batch'],
                    embed_model,
                    params['bidir'], )
        if USE_CUDA:
            input_batch = input_batch.cuda()
            att_batch = att_batch.cuda()
            target_batch = target_batch.cuda()
            model = model.cuda()


        # 拿出每个batch每行的概率矩阵
        zs = []
        for i in range(len(sample_batched['label'])):   # 0-32
            max_lines = max(sample_batched['label'][i]) # 拿到的是 lines 中 <eos> 的索引
            # 只传入每个冲突块 a+b+<eos> 的总行数的embedding表示（其余的行都一样
            probabilities, indices = model([input_batch[i:i+1, 0:max_lines+1], att_batch[i:i+1, 0:max_lines+1]])

            z = torch.zeros(1,30,30)
            for j in range(probabilities.shape[1]):
                for k in range(probabilities.shape[1]):
                    z[0, j, k] = probabilities[0, j, k]
            zs.append(z)
        batch_probabilities = torch.concat(zs)    # 32, 30, 30


        valid_len_batch = sample_batched['valid_len']               # batch 中每个冲突块实际resolved结果的长度（包括<eos>
        a_b_line_list = sample_batched['lines'] # a+b+eos的所有行，是list，且是30*32形状
        eos_index = [ [row[i] for row in a_b_line_list].index('<eos>') for i in range(len(a_b_line_list[0]))]

        cur_batch_size = len(valid_len_batch)

        # 第一处修改是改成下标+1（行号从1开始），第二处修改是 valid_len+1后直接改成0
        pred = torch.tensor([torch.argmax(probs)+1 for example in batch_probabilities for probs in example], dtype=torch.int64)
        for i in range(cur_batch_size):    # 0-32
            metEOS = False
            for j in range(max_len):       # 0-30
                if metEOS:
                    pred[i*max_len + j] = 0
                    continue
                if pred[i*max_len + j] == eos_index[i]+1:   # to check if bug
                    metEOS = True
        


        

        pred = pred.view(cur_batch_size, max_len)   # 预测结果
        if USE_CUDA:
            pred = pred.cuda()
        for i in range(len(valid_len_batch)):
            if pred[i][0:valid_len_batch[i]].equal(target_batch[i][0:valid_len_batch[i]]):  # 预测与实际相符
                batch_acc += 1

        # 如果训练的时候都告诉他实际长度，实际inference阶段怎么办？
        mask_tensor = torch.zeros(size=(batch_probabilities.size()[:2]))  # 32 * 30
        for i in range(batch_probabilities.size()[0]):
            mask_tensor[i][0:valid_len_batch[i]] = 1

        if USE_CUDA:
            mask_tensor = mask_tensor.cuda()
            batch_probabilities = batch_probabilities.cuda()

        # mask_tensor = mask_tensor.view(-1)

        # batch_probabilities = batch_probabilities.contiguous().view(-1, batch_probabilities.size()[-1])
        # target_batch = target_batch.view(-1)


        loss = torch.sum(torch.square(target_batch - pred), dtype=float)
        loss.requires_grad = True
        losses.append(loss.data)
        batch_loss.append(loss.data.cpu())

        # 可能存在的问题，valid_len后所有概率都为0，loss始终为3.4
        # loss = CCE(batch_probabilities, target_batch)

        # # 下面这两行代码相当于什么也没干
        # loss = torch.mul(loss, mask_tensor)
        # loss = loss.sum() / valid_len_batch.sum()

        # losses.append(loss.data)
        # batch_loss.append(loss.data.cpu())

        model_optim.zero_grad()
        loss.backward()
        model_optim.step()

        iterator.set_postfix(loss='{}'.format(loss.data))

    print('Epoch {0} / {1}, average loss : {2} , average accuracy : {3}%'.
          format(epoch + 1, params['nof_epoch'], np.average(batch_loss), batch_acc / len(dataset) * 100))

torch.save(model.state_dict(), params['save_path'])
