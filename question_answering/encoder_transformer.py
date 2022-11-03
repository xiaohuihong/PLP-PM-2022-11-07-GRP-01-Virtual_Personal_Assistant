import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import spacy
from question_answering.preprocess import *
nlp = spacy.load('en_core_web_sm')
import os
dir_path = os.path.dirname(os.path.realpath(__file__))


class DepthwiseSeparableConvolution(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, dim=1):
        super().__init__()
        self.dim = dim
        if dim == 2:
            self.depthwise_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                                            kernel_size=kernel_size, groups=in_channels, padding=kernel_size // 2)
            self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        else:
            self.depthwise_conv = nn.Conv1d(in_channels=in_channels, out_channels=in_channels,
                                            kernel_size=kernel_size, groups=in_channels, padding=kernel_size // 2,
                                            bias=False)
            self.pointwise_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        # x = [bs, seq_len, emb_dim]
        if self.dim == 1:
            x = x.transpose(1, 2)
            x = self.pointwise_conv(self.depthwise_conv(x))
            x = x.transpose(1, 2)
        else:
            x = self.pointwise_conv(self.depthwise_conv(x))
        # print("DepthWiseConv output: ", x.shape)
        return x


class HighwayLayer(nn.Module):

    def __init__(self, layer_dim, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        self.flow_layers = nn.ModuleList([nn.Linear(layer_dim, layer_dim) for _ in range(num_layers)])
        self.gate_layers = nn.ModuleList([nn.Linear(layer_dim, layer_dim) for _ in range(num_layers)])

    def forward(self, x):
        # print("Highway input: ", x.shape)
        for i in range(self.num_layers):
            flow = self.flow_layers[i](x)
            gate = torch.sigmoid(self.gate_layers[i](x))
            x = gate * flow + (1 - gate) * x
        # print("Highway output: ", x.shape)
        return x


class EmbeddingLayer(nn.Module):

    def __init__(self, char_vocab_dim, char_emb_dim, kernel_size, device):
        super().__init__()
        self.device = device
        self.weights_matrix = np.load(dir_path + '/dataset/qaglove_vt.npy')
        self.char_embedding = nn.Embedding(char_vocab_dim, char_emb_dim)
        self.word_embedding = self.get_glove_word_embedding()
        self.conv2d = DepthwiseSeparableConvolution(char_emb_dim, char_emb_dim, kernel_size, dim=2)
        self.highway = HighwayLayer(self.word_emb_dim + char_emb_dim)

    def get_glove_word_embedding(self):
        num_embeddings, embedding_dim = self.weights_matrix.shape
        self.word_emb_dim = embedding_dim
        embedding = nn.Embedding.from_pretrained(torch.FloatTensor(self.weights_matrix).to(self.device), freeze=True)
        return embedding

    def forward(self, x, x_char):
        # x = [bs, seq_len]
        # x_char = [bs, seq_len, word_len(=16)]
        word_emb = self.word_embedding(x)
        # word_emb = [bs, seq_len, word_emb_dim]
        word_emb = F.dropout(word_emb, p=0.1)
        char_emb = self.char_embedding(x_char)
        # char_embed = [bs, seq_len, word_len, char_emb_dim]
        char_emb = F.dropout(char_emb.permute(0, 3, 1, 2), p=0.05)
        # [bs, char_emb_dim, seq_len, word_len] == [N, Cin, Hin, Win]
        conv_out = F.relu(self.conv2d(char_emb))
        # [bs, char_emb_dim, seq_len, word_len]
        # the depthwise separable conv does not change the shape of the input
        char_emb, _ = torch.max(conv_out, dim=3)
        # [bs, char_emb_dim, seq_len]
        char_emb = char_emb.permute(0, 2, 1)
        # [bs, seq_len, char_emb_dim]
        concat_emb = torch.cat([char_emb, word_emb], dim=2)
        # [bs, seq_len, char_emb_dim + word_emb_dim]
        emb = self.highway(concat_emb)
        # [bs, seq_len, char_emb_dim + word_emb_dim]
        # print("Embedding output: ", emb.shape)
        return emb


class MultiheadAttentionLayer(nn.Module):

    def __init__(self, hid_dim, num_heads, device):
        super().__init__()
        self.num_heads = num_heads
        self.device = device
        self.hid_dim = hid_dim
        self.head_dim = self.hid_dim // self.num_heads
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, x, mask):
        # x = [bs, len_x, hid_dim]
        # mask = [bs, len_x]
        batch_size = x.shape[0]
        Q = self.fc_q(x)
        K = self.fc_k(x)
        V = self.fc_v(x)
        # Q = K = V = [bs, len_x, hid_dim]
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        # [bs, len_x, num_heads, head_dim ]  => [bs, num_heads, len_x, head_dim]
        K = K.permute(0, 1, 3, 2)
        # [bs, num_heads, head_dim, len_x]
        energy = torch.matmul(Q, K) / self.scale
        # (bs, num_heads){[len_x, head_dim] * [head_dim, len_x]} => [bs, num_heads, len_x, len_x]
        mask = mask.unsqueeze(1).unsqueeze(2)
        # [bs, 1, 1, len_x]
        # print("Mask: ", mask)
        # print("Energy: ", energy)
        energy = energy.masked_fill(mask == 1, -1e10)
        # print("energy after masking: ", energy)
        alpha = torch.softmax(energy, dim=-1)
        #  [bs, num_heads, len_x, len_x]
        # print("energy after smax: ", alpha)
        alpha = F.dropout(alpha, p=0.1)
        a = torch.matmul(alpha, V)
        # [bs, num_heads, len_x, head_dim]
        a = a.permute(0, 2, 1, 3)
        # [bs, len_x, num_heads, hid_dim]
        a = a.contiguous().view(batch_size, -1, self.hid_dim)
        # [bs, len_x, hid_dim]
        a = self.fc_o(a)
        # [bs, len_x, hid_dim]
        # print("Multihead output: ", a.shape)
        return a


class PositionEncoder(nn.Module):

    def __init__(self, model_dim, device, max_length=400):
        super().__init__()
        self.device = device
        self.model_dim = model_dim
        pos_encoding = torch.zeros(max_length, model_dim)
        for pos in range(max_length):
            for i in range(0, model_dim, 2):
                pos_encoding[pos, i] = math.sin(pos / (10000 ** ((2 * i) / model_dim)))
                pos_encoding[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / model_dim)))
        pos_encoding = pos_encoding.unsqueeze(0).to(device)
        self.register_buffer('pos_encoding', pos_encoding)

    def forward(self, x):
        # print("PE shape: ", self.pos_encoding.shape)
        # print("PE input: ", x.shape)
        x = x + Variable(self.pos_encoding[:, :x.shape[1]], requires_grad=False)
        # print("PE output: ", x.shape)
        return x


class EncoderBlock(nn.Module):

    def __init__(self, model_dim, num_heads, num_conv_layers, kernel_size, device):
        super().__init__()
        self.num_conv_layers = num_conv_layers
        self.conv_layers = nn.ModuleList([DepthwiseSeparableConvolution(model_dim, model_dim, kernel_size)
                                          for _ in range(num_conv_layers)])
        self.multihead_self_attn = MultiheadAttentionLayer(model_dim, num_heads, device)
        self.position_encoder = PositionEncoder(model_dim, device)
        self.pos_norm = nn.LayerNorm(model_dim)
        self.conv_norm = nn.ModuleList([nn.LayerNorm(model_dim) for _ in range(self.num_conv_layers)])
        self.feedfwd_norm = nn.LayerNorm(model_dim)
        self.feed_fwd = nn.Linear(model_dim, model_dim)

    def forward(self, x, mask):
        # x = [bs, len_x, model_dim]
        # mask = [bs, len_x]
        out = self.position_encoder(x)
        # [bs, len_x, model_dim]
        res = out
        out = self.pos_norm(out)
        # [bs, len_x, model_dim]
        for i, conv_layer in enumerate(self.conv_layers):
            out = F.relu(conv_layer(out))
            out = out + res
            if (i + 1) % 2 == 0:
                out = F.dropout(out, p=0.1)
            res = out
            out = self.conv_norm[i](out)
        out = self.multihead_self_attn(out, mask)
        # [bs, len_x, model_dim]
        out = F.dropout(out + res, p=0.1)
        res = out
        out = self.feedfwd_norm(out)
        out = F.relu(self.feed_fwd(out))
        # [bs, len_x, model_dim]
        out = F.dropout(out + res, p=0.1)
        # [bs, len_x, model_dim]
        # print("Encoder block output: ", out.shape)
        return out


class ContextQueryAttentionLayer(nn.Module):

    def __init__(self, model_dim):
        super().__init__()
        self.W0 = nn.Linear(3 * model_dim, 1, bias=False)

    def forward(self, C, Q, c_mask, q_mask):
        # C = [bs, ctx_len, model_dim]
        # Q = [bs, qtn_len, model_dim]
        # c_mask = [bs, ctx_len]
        # q_mask = [bs, qtn_len]
        c_mask = c_mask.unsqueeze(2)
        # [bs, ctx_len, 1]
        q_mask = q_mask.unsqueeze(1)
        # [bs, 1, qtn_len]
        ctx_len = C.shape[1]
        qtn_len = Q.shape[1]
        C_ = C.unsqueeze(2).repeat(1, 1, qtn_len, 1)
        # [bs, ctx_len, qtn_len, model_dim]
        Q_ = Q.unsqueeze(1).repeat(1, ctx_len, 1, 1)
        # [bs, ctx_len, qtn_len, model_dim]
        C_elemwise_Q = torch.mul(C_, Q_)
        # [bs, ctx_len, qtn_len, model_dim]
        S = torch.cat([C_, Q_, C_elemwise_Q], dim=3)
        # [bs, ctx_len, qtn_len, model_dim*3]
        S = self.W0(S).squeeze()
        # print("Simi matrix: ", S.shape)
        # [bs, ctx_len, qtn_len, 1] => # [bs, ctx_len, qtn_len]
        S_row = S.masked_fill(q_mask == 1, -1e10)
        S_row = F.softmax(S_row, dim=2)
        S_col = S.masked_fill(c_mask == 1, -1e10)
        S_col = F.softmax(S_col, dim=1)
        A = torch.bmm(S_row, Q)
        # (bs)[ctx_len, qtn_len] X [qtn_len, model_dim] => [bs, ctx_len, model_dim]
        B = torch.bmm(torch.bmm(S_row, S_col.transpose(1, 2)), C)
        # [ctx_len, qtn_len] X [qtn_len, ctx_len] => [bs, ctx_len, ctx_len]
        # [ctx_len, ctx_len] X [ctx_len, model_dim ] => [bs, ctx_len, model_dim]
        model_out = torch.cat([C, A, torch.mul(C, A), torch.mul(C, B)], dim=2)
        # [bs, ctx_len, model_dim*4]
        # print("C2Q output: ", model_out.shape)
        return F.dropout(model_out, p=0.1)


class OutputLayer(nn.Module):

    def __init__(self, model_dim):
        super().__init__()
        self.W1 = nn.Linear(2 * model_dim, 1, bias=False)
        self.W2 = nn.Linear(2 * model_dim, 1, bias=False)

    def forward(self, M1, M2, M3, c_mask):
        start = torch.cat([M1, M2], dim=2)
        start = self.W1(start).squeeze()
        p1 = start.masked_fill(c_mask == 1, -1e10)
        # p1 = F.log_softmax(start.masked_fill(c_mask==1, -1e10), dim=1)
        end = torch.cat([M1, M3], dim=2)
        end = self.W2(end).squeeze()
        p2 = end.masked_fill(c_mask == 1, -1e10)
        # p2 = F.log_softmax(end.masked_fill(c_mask==1, -1e10), dim=1)
        # print("preds: ", [p1.shape,p2.shape])
        return p1, p2


class EncoderTransformer(nn.Module):

    def __init__(self, char_vocab_dim, char_emb_dim, word_emb_dim, kernel_size, model_dim, num_heads, device):
        super().__init__()
        self.embedding = EmbeddingLayer(char_vocab_dim, char_emb_dim, kernel_size, device)
        self.ctx_resizer = DepthwiseSeparableConvolution(char_emb_dim + word_emb_dim, model_dim, 5)
        self.qtn_resizer = DepthwiseSeparableConvolution(char_emb_dim + word_emb_dim, model_dim, 5)
        self.embedding_encoder = EncoderBlock(model_dim, num_heads, 4, 5, device)
        self.c2q_attention = ContextQueryAttentionLayer(model_dim)
        self.c2q_resizer = DepthwiseSeparableConvolution(model_dim * 4, model_dim, 5)
        self.model_encoder_layers = nn.ModuleList([EncoderBlock(model_dim, num_heads, 2, 5, device)
                                                   for _ in range(7)])
        self.output = OutputLayer(model_dim)
        self.device = device

    def forward(self, ctx, qtn, ctx_char, qtn_char):
        c_mask = torch.eq(ctx, 1).float().to(self.device)
        q_mask = torch.eq(qtn, 1).float().to(self.device)
        ctx_emb = self.embedding(ctx, ctx_char)
        # [bs, ctx_len, ch_emb_dim + word_emb_dim]
        ctx_emb = self.ctx_resizer(ctx_emb)
        #  [bs, ctx_len, model_dim]
        qtn_emb = self.embedding(qtn, qtn_char)
        # [bs, ctx_len, ch_emb_dim + word_emb_dim]
        qtn_emb = self.qtn_resizer(qtn_emb)
        # [bs, qtn_len, model_dim]
        C = self.embedding_encoder(ctx_emb, c_mask)
        # [bs, ctx_len, model_dim]
        Q = self.embedding_encoder(qtn_emb, q_mask)
        # [bs, qtn_len, model_dim]
        C2Q = self.c2q_attention(C, Q, c_mask, q_mask)
        # [bs, ctx_len, model_dim*4]
        M1 = self.c2q_resizer(C2Q)
        # [bs, ctx_len, model_dim]
        for layer in self.model_encoder_layers:
            M1 = layer(M1, c_mask)
        M2 = M1
        # [bs, ctx_len, model_dim]
        for layer in self.model_encoder_layers:
            M2 = layer(M2, c_mask)
        M3 = M2
        # [bs, ctx_len, model_dim]
        for layer in self.model_encoder_layers:
            M3 = layer(M3, c_mask)
        p1, p2 = self.output(M1, M2, M3, c_mask)
        return p1, p2


class qa_model():
    def __init__(self):
        self.word2idx = np.load(dir_path + '/dataset/qa_word2idx.npy', allow_pickle=True).item()
        self.idx2word = np.load(dir_path + '/dataset/qa_idx2word.npy', allow_pickle=True).item()
        self.char2idx = np.load(dir_path + '/dataset/qa_char2idx.npy', allow_pickle=True).item()
        self.char_vocab_dim = len(self.char2idx)
        self.char_emb_dim = 200
        self.word_emb_dim = 300
        self.kernel_size = 5
        self.model_dim = 128
        self.num_attention_heads = 8
        self.device = torch.device('cpu')
        self.model = EncoderTransformer(self.char_vocab_dim,
                                        self.char_emb_dim,
                                        self.word_emb_dim,
                                        self.kernel_size,
                                        self.model_dim,
                                        self.num_attention_heads,
                                        self.device).to(self.device)
        self.model.load_state_dict(torch.load(dir_path + '/model/model_encoder_transformer.h5', map_location ='cpu'))

    def make_char_vector(self, max_sent_len, sentence, max_word_len=16):
        char_vec = torch.zeros(max_sent_len, max_word_len).type(torch.LongTensor)
        for i, word in enumerate(nlp(sentence, disable=['parser', 'tagger', 'ner'])):
            for j, ch in enumerate(word.text):
                if j == max_word_len:
                    break
                char_vec[i][j] = self.char2idx.get(ch, 0)
        return char_vec

    def predict(self, context, question):
        self.model.eval()
        input_list = [context, question]
        input_df = pd.DataFrame([input_list])
        input_df.columns = ["context", "question"]

        input_df['context_ids'] = input_df.context.apply(context_to_ids, word2idx=self.word2idx)
        input_df['question_ids'] = input_df.question.apply(question_to_ids, word2idx=self.word2idx)

        # input_err = get_error_indices(input_df, self.idx2word)
        # input_df.drop(input_err, inplace=True)

        max_context_len = max([len(ctx) for ctx in input_df.context_ids])
        padded_context = torch.LongTensor(len(input_df), max_context_len).fill_(1)

        max_question_len = max([len(ques) for ques in input_df.question_ids])
        padded_question = torch.LongTensor(len(input_df), max_question_len).fill_(1)

        for i, ctx in enumerate(input_df.context_ids):
            padded_context[i, :len(ctx)] = torch.LongTensor(ctx)

        for i, ques in enumerate(input_df.question_ids):
            padded_question[i, : len(ques)] = torch.LongTensor(ques)

        max_word_ctx = 16
        char_ctx = torch.zeros(len(input_df), max_context_len, max_word_ctx).type(torch.LongTensor)
        for i, context in enumerate(input_df.context):
            char_ctx[i] = self.make_char_vector(max_context_len, context)

        max_word_ques = 16
        char_ques = torch.zeros(len(input_df), max_question_len, max_word_ques).type(torch.LongTensor)
        for i, question in enumerate(input_df.question):
            char_ques[i] = self.make_char_vector(max_question_len, question)

        p1, p2 = self.model(padded_context, padded_question, char_ctx, char_ques)

        batch_size, c_len = p1.size()
        ls = nn.LogSoftmax(dim=1)
        mask = (torch.ones(c_len, c_len) * float('-inf')).to(self.device).tril(-1).unsqueeze(0).expand(batch_size, -1, -1)

        score = (ls(p1).unsqueeze(2) + ls(p2).unsqueeze(1)) + mask
        score, s_idx = score.max(dim=1)
        score, e_idx = score.max(dim=1)
        s_idx = torch.gather(s_idx, 1, e_idx.view(-1, 1)).squeeze()

        # stack predictions
        # predictions = {}
        # for i in range(batch_size):
        #     pred = input_df.context_ids[i][s_idx[i]:e_idx[i] + 1]
        #     pred = ' '.join([idx2word[idx.item()] for idx in pred])
        #     predictions[i] = pred

        pred = input_df.context_ids[0][s_idx.item():e_idx.item() + 1]
        answer = ' '.join([self.idx2word[idx] for idx in pred])
        print(f"Context: {context} \nQuestion: {question} \nAnswer: {answer}".format(context=context, question=question, answer=answer))

        return answer


if __name__ == '__main__':
    context = "Today is John's birthday."
    question = "What day is it today?"
    model = qa_model()
    pred = model.predict(context, question)
    print("Get the answer: ", pred)















