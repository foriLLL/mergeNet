from PtrNet import PointerNet
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from datasets import load_from_disk, DatasetDict, Dataset
from transformers import RobertaModel
import numpy as np
from config import max_len


params = dict()
params['batch_size'] = 32
params['nof_epoch'] = 200
params['lr'] = 0.001
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
# tokenized dataset path
params['dataset_path'] = './output/tokenized_output'

# 加载数据集
dataset = load_from_disk(params['dataset_path']
                         )
dataset = dataset.with_format('torch')
if not isinstance(dataset, DatasetDict):
    raise TypeError("dataset must be an instance of DatasetDict")
dataset = dataset['train']

# dataset.train_test_split(test_size=0.1)
# dataset = dataset.shuffle(seed=random.randint(0, 100))     # 注意这里和师兄代码有改动，师兄代码加载的DatasetDict没有splits

dataloader = DataLoader(dataset,    # type: ignore
                        batch_size=params['batch_size'],
                        shuffle=False,
                        num_workers=0)


embed_model = RobertaModel.from_pretrained(params['codeBERT'])
if not isinstance(embed_model, RobertaModel):
    raise TypeError("embed_model must be an instance of RobertaModel")

if not params['bert_grad']:
    for param in embed_model.parameters():
        param.requires_grad_(False)

model = PointerNet(params['embedding_size'],        # 768
                   params['hiddens'],               # 1024
                   params['nof_lstms'],             # 4
                   params['dropout'],               # 0.4
                   params['embed_batch'],           # 1
                   embed_model,
                   params['bidir'])                 # True

if params['gpu'] and torch.cuda.is_available():
    USE_CUDA = True
    model.cuda()
    model = torch.nn.DataParallel(
        model, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True
    print('Using GPU, %i devices.' % torch.cuda.device_count())
else:
    USE_CUDA = False
    print('Using CPU')


CCE = torch.nn.CrossEntropyLoss()
model_optim = optim.Adam(filter(lambda p: p.requires_grad,
                                model.parameters()),
                         lr=params['lr'])


print(params)
# 开始训练
losses = []
for epoch in range(params['nof_epoch']):
    batch_loss = []
    acc_ans = 0
    # iterator = tqdm(dataloader, unit='Batch')

    for i_batch, sample_batched in enumerate(dataloader):
        batch_acc = 0
        # iterator.set_description('Epoch %i/%i' % (epoch + 1, params['nof_epoch']))

        # 32 * 30 * 512     tokenized 后的每一行
        input_batch = sample_batched['input_ids']
        # 32 * 30 * 512     tokenized 后的每一行 token 中有效的部分（01矩阵）
        att_batch = sample_batched['attention_mask']
        # 32 * 30          resolution 的行号排序
        target_batch = sample_batched['label']

       # 改成不用batch试试

        if USE_CUDA:
            input_batch = input_batch.cuda()
            att_batch = att_batch.cuda()
            target_batch = target_batch.cuda()

        # 拿出每个batch每行的概率矩阵
        zs = []
        for i in range(len(sample_batched['label'])):   # 0-32
            # 拿到的是 lines 中 <eos> 的索引
            max_lines = max(sample_batched['label'][i])
            # 只传入每个冲突块 a+b+<eos> 的总行数的embedding表示（其余的行都一样
            probabilities, indices = model(
                [input_batch[i:i+1, 0:max_lines], att_batch[i:i+1, 0:max_lines]])   # 这里注意，每次只训练一个元素
            

            z = torch.zeros(1, 30, 30)
            for j in range(probabilities.shape[1]):
                for k in range(probabilities.shape[1]):
                    z[0, j, k] = probabilities[0, j, k]
            zs.append(z)
        batch_probabilities = torch.concat(zs)    # 32, 30, 30

        # batch 中每个冲突块实际resolved结果的长度（包括<eos>
        valid_len_batch = sample_batched['valid_len']
        a_b_line_list = sample_batched['lines']  # a+b+eos的所有行，是list，且是30*32形状
        eos_index = [[row[i] for row in a_b_line_list].index(
            '<eos>')+1 for i in range(len(a_b_line_list[0]))]

        cur_batch_size = len(valid_len_batch)

        # 第一处修改是改成下标+1（行号从1开始），第二处修改是 valid_len+1后直接改成0
        pred = torch.tensor([torch.argmax(
            probs)+1 for example in batch_probabilities for probs in example], dtype=torch.int64)
        for i in range(cur_batch_size):    # 0-32
            metEOS = False
            for j in range(max_len):       # 0-30
                if metEOS:
                    pred[i*max_len + j] = 0
                    continue
                if pred[i*max_len + j] == eos_index[i]:
                    metEOS = True

        if USE_CUDA:
            batch_probabilities = batch_probabilities.cuda()

        batch_probabilities = batch_probabilities.contiguous(
        ).view(-1, batch_probabilities.size()[-1])  # [32*30, 30]

        # 将 batch_probabilities 在 eos 后的 0 都改为 1
        for i in range(len(batch_probabilities)):
            if pred[i] == 0:
                batch_probabilities[i][:] = 0
                batch_probabilities[i][0] = 1

        pred = pred.view(cur_batch_size, max_len)   # 预测结果
        if USE_CUDA:
            pred = pred.cuda()
        for i in range(len(valid_len_batch)):
            # 预测与实际相符
            if pred[i][0:valid_len_batch[i]].equal(target_batch[i][0:valid_len_batch[i]]):
                acc_ans += 1
                batch_acc += 1
                # print("accurate: ",batch_acc)
                # print(pred[i][0:valid_len_batch[i]])
                # print(target_batch[i][0:valid_len_batch[i]])
                # print('*' * 10)

        target_batch = target_batch.view(-1)
        loss = CCE(batch_probabilities, target_batch)
        losses.append(loss.data)
        batch_loss.append(loss.data.cpu())

        if i_batch % 1 == 0:
            print(
                f"training {i_batch} batches in epoch {epoch}, loss = {loss.data}, batch_acc = {batch_acc} / {len(sample_batched['label'])}")

        model_optim.zero_grad()
        loss.backward()
        model_optim.step()

        # iterator.set_postfix(loss='{}'.format(loss.data))

    print('Epoch {0} / {1}, average loss : {2} , average accuracy : {3}%'.
          format(epoch + 1, params['nof_epoch'], np.average(batch_loss), acc_ans / len(dataset) * 100))

torch.save(model.state_dict(), params['save_path'])
