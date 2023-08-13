import torch
import torch.nn as nn
from torch.nn import Parameter

class Encoder(nn.Module):
    """
    Encoder class for Pointer-Net
    """

    def __init__(self, embedding_dim,
                 hidden_dim,
                 n_layers,
                 dropout,
                 bidir):
        """
        Initiate Encoder

        :param Tensor embedding_dim: Number of embbeding channels
        :param int hidden_dim: Number of hidden units for the LSTM
        :param int n_layers: Number of layers for LSTMs
        :param float dropout: Float between 0-1
        :param bool bidir: Bidirectional
        """

        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim // 2 if bidir else hidden_dim
        self.n_layers = n_layers * 2 if bidir else n_layers         # 考虑了是否双向
        self.bidir = bidir
        self.lstm = nn.LSTM(embedding_dim,
                            self.hidden_dim,
                            n_layers,
                            dropout=dropout,
                            bidirectional=bidir)

        # Used for propagating .cuda() command
        self.h0 = Parameter(torch.zeros(1), requires_grad=False)
        self.c0 = Parameter(torch.zeros(1), requires_grad=False)
        # ChatGPT：
        # 使用全零的张量作为 LSTM 的初始状态是一种常见的做法，因为它可以帮助模型更好地捕捉序列中的长期依赖关系。
        # 在训练过程中，LSTM 会根据输入数据和当前状态计算下一个状态，并将其作为下一次计算的初始状态。
        # 因此，初始状态的值并不会对模型的最终输出产生影响，而只是为了帮助模型更好地捕捉序列中的长期依赖关系。

    def forward(self, embedded_inputs,
                hidden):
        """
        Encoder - Forward-pass

        :param Tensor embedded_inputs: Embedded inputs of Pointer-Net
        :param Tensor hidden: Initiated hidden units for the LSTMs (h, c)
        :return: LSTMs outputs and hidden units (h, c)
        """

        embedded_inputs = embedded_inputs.permute(1, 0, 2)  # [30 * 32 * 768]， 因为输入是 [seq_len, batch_size, input_size]

        outputs, hidden = self.lstm(embedded_inputs, hidden)
        # LSTM  input   -> (L, N, H_in), (h_0, c_0)
        #       output  -> (L, N, D*H_out), (h_n, c_n)

        return outputs.permute(1, 0, 2), hidden
        # outputs.permute: [32, 30, 1024]
        # hidden: (h_n, c_n)

    def init_hidden(self, embedded_inputs):
        """
        Initiate hidden units

        :param Tensor embedded_inputs: The embedded input of Pointer-NEt
        :return: Initiated hidden units for the LSTMs (h, c)
        """

        batch_size = embedded_inputs.size(0)

        # Reshaping (Expanding)
        h0 = self.h0.unsqueeze(0).unsqueeze(0).repeat(self.n_layers,
                                                      batch_size,
                                                      self.hidden_dim)
        c0 = self.h0.unsqueeze(0).unsqueeze(0).repeat(self.n_layers,
                                                      batch_size,
                                                      self.hidden_dim)

        return h0, c0


class Attention(nn.Module):
    """
    Attention model for Pointer-Net
    """

    def __init__(self, input_dim, hidden_dim):
        """
        Initiate Attention

        :param int input_dim: Input's dimention                         1024
        :param int hidden_dim: Number of hidden units in the attention  1024
        """

        super(Attention, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.input_linear = nn.Linear(input_dim, hidden_dim)
        self.context_linear = nn.Conv1d(input_dim, hidden_dim, 1, 1)
        self.V = Parameter(torch.FloatTensor(hidden_dim), requires_grad=True)
        self._inf = Parameter(torch.FloatTensor([float('-inf')]), requires_grad=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

        # Initialize vector V
        nn.init.uniform_(self.V, -1, 1)

    def forward(self, input,    # h_t
                context,
                mask):
        """
        Attention - Forward-pass

        :param Tensor input: Hidden state h
        :param Tensor context: Attention context        decoder 中 forward 的 context 始终为 encoder 最后一个输出
        :param ByteTensor mask: Selection mask
        :return: tuple of - (Attentioned hidden state, Alphas)
        """

        # (batch, hidden_dim, seq_len)
        inp = self.input_linear(input).unsqueeze(2).expand(-1, -1, context.size(1))

        # (batch, hidden_dim, seq_len)
        context = context.permute(0, 2, 1)
        ctx = self.context_linear(context)

        # (batch, 1, hidden_dim)
        V = self.V.unsqueeze(0).expand(context.size(0), -1).unsqueeze(1)

        # (batch, seq_len)
        att = torch.bmm(V, self.tanh(inp + ctx)).squeeze(1)
        if len(att[mask]) > 0:
            att[mask] = self.inf[mask]
        alpha = self.softmax(att)

        hidden_state = torch.bmm(ctx, alpha.unsqueeze(2)).squeeze(2)

        return hidden_state, alpha

    def init_inf(self, mask_size):
        self.inf = self._inf.unsqueeze(1).expand(*mask_size)


class Decoder(nn.Module):
    """
    Decoder model for Pointer-Net
    """

    def __init__(self, embedding_dim,
                 hidden_dim):
        """
        Initiate Decoder

        :param int embedding_dim: Number of embeddings in Pointer-Net
        :param int hidden_dim: Number of hidden units for the decoder's RNN
        """

        super(Decoder, self).__init__()
        self.embedding_dim = embedding_dim          # 768
        self.hidden_dim = hidden_dim                # 1024

        self.input_to_hidden = nn.Linear(embedding_dim, 4 * hidden_dim) # 768 -> 1024 * 4
        self.hidden_to_hidden = nn.Linear(hidden_dim, 4 * hidden_dim)   # 1024 -> 1024 * 4
        self.hidden_out = nn.Linear(hidden_dim * 2, hidden_dim)         # 1024 * 2 -> 1024
        self.att = Attention(hidden_dim, hidden_dim)

        # Used for propagating .cuda() command
        self.mask = Parameter(torch.ones(1), requires_grad=False)
        self.runner = Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, embedded_inputs,
                decoder_input,
                hidden,
                context):
        """
        Decoder - Forward-pass

        :param Tensor embedded_inputs: Embedded inputs of Pointer-Net   # [1, 3, 768]
        :param Tensor decoder_input: First decoder's input
        :param Tensor hidden: First decoder's hidden states
        :param Tensor context: Encoder's outputs
        :return: (Output probabilities, Pointers indices), last hidden state
        """

        batch_size = embedded_inputs.size(0)       # 1
        input_length = embedded_inputs.size(1)     # 3

        # (batch, seq_len)
        mask = self.mask.repeat(input_length).unsqueeze(0).repeat(batch_size, 1)    # [1, 3] tensor([[1., 1., 1.,]])
        self.att.init_inf(mask.size())      # Attention 中的 inf 变为 [1, 3] tensor([[-inf, -inf, -inf]])

        # Generating arang(input_length), broadcasted across batch_size
        runner = self.runner.repeat(input_length)   # [1, 3] tensor([0., 0., 0.])
        for i in range(input_length):
            runner.data[i] = i          # [1, 3] tensor([0., 1., 2.])
        
        # 可以换成这个，但是有 device 问题 runner = torch.arange(input_length, dtype=torch.float)  # [1, 3] tensor([0., 1., 2.])
        runner = runner.unsqueeze(0).expand(batch_size, -1).long()  # [1, 3] tensor([[0, 1, 2]])

        outputs = []
        pointers = []

        def step(x, hidden):
            """
            Recurrence step function

            :param Tensor x: Input at time t            [1, 768]
            :param tuple(Tensor, Tensor) hidden: Hidden states at time t-1
            :return: Hidden states at time t (h, c), Attention probabilities (Alpha)
            """

            # Regular LSTM
            h, c = hidden   # h: [1, 1024]  c: [1, 1024]

            gates = self.input_to_hidden(x) + self.hidden_to_hidden(h)  # [1, 1024 * 4]
            input, forget, cell, out = gates.chunk(4, 1)    # [1, 1024]

            input = torch.sigmoid(input)
            forget = torch.sigmoid(forget)
            cell = torch.tanh(cell)
            out = torch.sigmoid(out)    # 概率

            c_t = (forget * c) + (input * cell) # 时间步t输出的c
            h_t = out * torch.tanh(c_t)  # 概率乘 c_t，也就是attention pooling 后的结果

            # Attention section
            hidden_t, output = self.att(h_t, context, torch.eq(mask, 0)) # Attention 机制, output 为输出概率
            hidden_t = torch.tanh(self.hidden_out(torch.cat((hidden_t, h_t), 1)))   # 拼接 hidden_t 和 h_t

            return hidden_t, c_t, output

        # Recurrence loop
        for _ in range(input_length):
            h_t, c_t, outs = step(decoder_input, hidden)
            hidden = (h_t, c_t)

            # Masking selected inputs
            masked_outs = outs * mask                   # [[0.32, 0.34, 0.33]] * [[1, 1, 1]] = [[0.32, 0.34, 0.33]]

            # Get maximum probabilities and indices
            max_probs, indices = masked_outs.max(1)
            one_hot_pointers = (runner == indices.unsqueeze(1).expand(-1, outs.size()[1])).float()  # [[0., 1., 0.]]

            # Update mask to ignore seen indices
            mask = mask * (1 - one_hot_pointers)

            # Get embedded inputs by max indices
            embedding_mask = one_hot_pointers.unsqueeze(2).expand(-1, -1, self.embedding_dim).bool()    # [1, 3, 768] 每次有3行里有1行全是True，其他行全False
            decoder_input = embedded_inputs[embedding_mask.data].view(batch_size, self.embedding_dim)   # [1, 768] 从 embedded_inputs 中取出 3 行中的 1 行

            outputs.append(outs.unsqueeze(0))
            pointers.append(indices.unsqueeze(1))

        outputs = torch.cat(outputs).permute(1, 0, 2)
        pointers = torch.cat(pointers, 1)

        return (outputs, pointers), hidden


class Embedding(nn.Module):
    def __init__(self, codeBERT, batch_size):
        super(Embedding, self).__init__()
        self.bert = codeBERT
        self.batch = batch_size     # embed_batch，这里是1

    # 两个参数都是 batchsize * max_line * token后长度  (32*30*512)
    def forward(self, input_ids, attention_mask):
        return self.bert(input_ids, attention_mask)['pooler_output']


class PointerNet(nn.Module):
    """
    Pointer-Net
    """

    def __init__(self, embedding_dim,
                 hidden_dim,
                 lstm_layers,
                 dropout,
                 embed_batch,
                 bert,
                 bidir=False,):
        """
        Initiate Pointer-Net

        :param int embedding_dim: Number of embbeding channels      768
        :param int hidden_dim: Encoders hidden units                1024
        :param int lstm_layers: Number of layers for LSTMs
        :param float dropout: Float between 0-1
        :param bool bidir: Bidirectional
        """

        super(PointerNet, self).__init__()
        self.embedding_dim = embedding_dim
        self.bidir = bidir
        # self.embedding = nn.Linear(2, embedding_dim)
        self.bert = bert
        self.embed_batch = embed_batch
        self.embedding = Embedding(self.bert, embed_batch)
        self.encoder = Encoder(embedding_dim,
                               hidden_dim,      # 1024
                               lstm_layers,
                               dropout,
                               bidir)
        self.decoder = Decoder(embedding_dim, hidden_dim)
        self.decoder_input0 = Parameter(torch.FloatTensor(embedding_dim), requires_grad=False)  # [768]

        # Initialize decoder_input0
        nn.init.uniform_(self.decoder_input0, -1, 1)

    def forward(self, inputs):
        """
        PointerNet - Forward-pass

        :param Tensor inputs: Input sequence
        :return: Pointers probabilities and indices
        """

        batch_size = inputs[0].size(0)

        # 针对批量化的处理
        decoder_input0 = self.decoder_input0.unsqueeze(0).expand(batch_size, -1)

        input_ids = inputs[0]       # 现在的做法是没有批量化，一次一个[1, a+b+<eoos>, 512]  如 [1, 3, 512]
        att_mask = inputs[1]        # [1, a+b+<eoos>, 512]

        # embedding
        input_ids = input_ids.view(-1, input_ids.size()[-1])    # 32*30*512 -> 960 * 512    1, 3, 512 -> 3, 512
        att_mask = att_mask.view(-1, att_mask.size()[-1])       # 32*30*512 -> 960 * 512    1, 3, 512 -> 3, 512
        embedded_inputs = self.embedding(input_ids, att_mask)   # 960 * 512 -> 960 * 768    1, 3, 512 -> 3, 768
        embedded_inputs = embedded_inputs.view(batch_size, -1, embedded_inputs.size()[-1])  # 960 * 768 -> 32 * 30 * 768    3, 768 -> 1, 3, 768

        # encoder
        encoder_hidden0 = self.encoder.init_hidden(embedded_inputs)     #([8, 1, 512], [8, 1, 512])
            # 输出   encoder_outputs [1, 3, D * H_out]  D = 2 if bidirection    H_out = 512
            #       encoder_hidden (h_n, c_n)    ([8, 1, 512], [8, 1, 512]) 四层双向 因为是双向，所以 1024 // 2
        encoder_outputs, encoder_hidden = self.encoder(embedded_inputs,
                                                       encoder_hidden0)

        # decoder 第一个decoder的隐状态是最后一个encoder输出的隐状态
        if self.bidir:
            decoder_hidden0 = (torch.cat((encoder_hidden[0][-2], encoder_hidden[0][-1]), dim=-1),
                               torch.cat((encoder_hidden[1][-2], encoder_hidden[1][-1]), dim=-1))       # todo: 这里为什么不是[1][-2]   分别是 decoder LSTM 的 h, c，为什么不用 1
                            # ([1, 1024], [1, 1024])
        else:
            decoder_hidden0 = (encoder_hidden[0][-1],
                               encoder_hidden[1][-1])


        (outputs, pointers), decoder_hidden = self.decoder(embedded_inputs,
                                                           decoder_input0,      # decoder 第一个输入    -1, 1 均匀分布随机数
                                                           decoder_hidden0,     # decoder 第一个hidden state， 为什么是一个 tuple
                                                           encoder_outputs)     # context 输出

        return outputs, pointers
