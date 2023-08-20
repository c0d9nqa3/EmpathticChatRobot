import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math
from Model.common_layer import EncoderLayer, DecoderLayer, MultiHeadAttention, Conv, PositionwiseFeedForward, LayerNorm , _gen_bias_mask ,_gen_timing_signal, share_embedding, LabelSmoothing, NoamOpt, _get_attn_subsequent_mask
#from utils import config
import random
# from numpy import random
import os
import pprint
from tqdm import tqdm

pp = pprint.PrettyPrinter(indent=1)
import os
import time
from copy import deepcopy
from sklearn.metrics import accuracy_score
import pdb
from utils.common import get_emotion_words

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Semantic_Encoder(nn.Module):
    """
    A Transformer Encoder module.
    Inputs should be in the shape [batch_size, length, hidden_size]
    Outputs will have the shape [batch_size, length, hidden_size]
    Refer Fig.1 in https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(self, embedding_size, hidden_size, num_layers, num_heads, total_key_depth, total_value_depth,
                 filter_size, max_length=1000, input_dropout=0.0, layer_dropout=0.0,
                 attention_dropout=0.0, relu_dropout=0.0, use_mask=False, universal=False, concept=False):
        """
        Parameters:
            embedding_size: Size of embeddings
            hidden_size: Hidden size
            num_layers: Total layers in the Encoder  2
            num_heads: Number of attention heads   2
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head   40
            total_value_depth: Size of last dimension of values. Must be divisible by num_head  40
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN  50
            max_length: Max sequence length (required for timing signal)
            input_dropout: Dropout just after embedding
            layer_dropout: Dropout for each layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
            use_mask: Set to True to turn on future value masking
        """

        super(Semantic_Encoder, self).__init__()
        self.universal = universal
        self.num_layers = num_layers
        self.timing_signal = _gen_timing_signal(max_length, hidden_size)

        if (self.universal):
            ## for t
            self.position_signal = _gen_timing_signal(num_layers, hidden_size)

        params = (hidden_size,
                  total_key_depth or hidden_size,
                  total_value_depth or hidden_size,
                  filter_size,
                  num_heads,
                  _gen_bias_mask(max_length) if use_mask else None,
                  layer_dropout,
                  attention_dropout,
                  relu_dropout)

        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)
        if (self.universal):
            self.enc = EncoderLayer(*params)
        else:
            self.enc = nn.ModuleList([EncoderLayer(*params) for _ in range(num_layers)])

        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)

    def forward(self, inputs, mask):
        # Add input dropout
        x = self.input_dropout(inputs)

        # Project to hidden size
        x = self.embedding_proj(x)

        if (self.universal):
            if (False):
                x, (self.remainders, self.n_updates) = self.act_fn(x, inputs, self.enc, self.timing_signal,
                                                                   self.position_signal, self.num_layers)
                y = self.layer_norm(x)
            else:
                for l in range(self.num_layers):
                    x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)
                    x += self.position_signal[:, l, :].unsqueeze(1).repeat(1, inputs.shape[1], 1).type_as(inputs.data)
                    x = self.enc(x, mask=mask)
                y = self.layer_norm(x)
        else:
            # Add timing signal
            x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)

            for i in range(self.num_layers):
                x = self.enc[i](x, mask)

            y = self.layer_norm(x)
        return y


class Emotion_Encoder(nn.Module):
    """
    A Transformer Encoder module.
    """

    def __init__(self, embedding_size, hidden_size, num_layers, num_heads, total_key_depth, total_value_depth,
                 filter_size, max_length=1000, input_dropout=0.0, layer_dropout=0.0,
                 attention_dropout=0.0, relu_dropout=0.0, use_mask=False, universal=False, concept=False):
        """
        Parameters:
            embedding_size: Size of embeddings
            hidden_size: Hidden size
            num_layers: Total layers in the Encoder  2
            num_heads: Number of attention heads   2
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head   40
            total_value_depth: Size of last dimension of values. Must be divisible by num_head  40
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN  50
            max_length: Max sequence length (required for timing signal)
            input_dropout: Dropout just after embedding
            layer_dropout: Dropout for each layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
            use_mask: Set to True to turn on future value masking
        """

        super(Emotion_Encoder, self).__init__()
        self.universal = universal
        self.num_layers = num_layers
        self.timing_signal = _gen_timing_signal(max_length, hidden_size)

        if (self.universal):
            ## for t
            self.position_signal = _gen_timing_signal(num_layers, hidden_size)

        params = (hidden_size,
                  total_key_depth or hidden_size,
                  total_value_depth or hidden_size,
                  filter_size,
                  num_heads,
                  _gen_bias_mask(max_length) if use_mask else None,
                  layer_dropout,
                  attention_dropout,
                  relu_dropout)

        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)
        if (self.universal):
            self.enc = EncoderLayer(*params)
        else:
            self.enc = nn.ModuleList([EncoderLayer(*params) for _ in range(num_layers)])

        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)

    def forward(self, inputs, mask):
        # Add input dropout
        x = self.input_dropout(inputs)

        # Project to hidden size
        x = self.embedding_proj(x)

        if (self.universal):
            if (False):
                x, (self.remainders, self.n_updates) = self.act_fn(x, inputs, self.enc, self.timing_signal,
                                                                   self.position_signal, self.num_layers)
                y = self.layer_norm(x)
            else:
                for l in range(self.num_layers):
                    x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)
                    x += self.position_signal[:, l, :].unsqueeze(1).repeat(1, inputs.shape[1], 1).type_as(inputs.data)
                    x = self.enc(x, mask=mask)
                y = self.layer_norm(x)
        else:
            # Add timing signal
            x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)

            for i in range(self.num_layers):
                x = self.enc[i](x, mask)

            y = self.layer_norm(x)
        return y


class Decoder(nn.Module):
    """
    A Transformer Decoder module.
    """

    def __init__(self, embedding_size, hidden_size, num_layers, num_heads, total_key_depth, total_value_depth,
                 filter_size, max_length=1000, input_dropout=0.0, layer_dropout=0.0,
                 attention_dropout=0.0, relu_dropout=0.0, universal=False):
        """
        Parameters:
            embedding_size: Size of embeddings
            hidden_size: Hidden size
            num_layers: Total layers in the Encoder
            num_heads: Number of attention heads
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN
            max_length: Max sequence length (required for timing signal)
            input_dropout: Dropout just after embedding
            layer_dropout: Dropout for each layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
        """

        super(Decoder, self).__init__()
        self.universal = universal
        self.num_layers = num_layers
        self.timing_signal = _gen_timing_signal(max_length, hidden_size)

        if (self.universal):
            ## for t
            self.position_signal = _gen_timing_signal(num_layers, hidden_size)

        self.mask = _get_attn_subsequent_mask(max_length)

        params = (hidden_size,
                  total_key_depth or hidden_size,
                  total_value_depth or hidden_size,
                  filter_size,
                  num_heads,
                  _gen_bias_mask(max_length),  # mandatory
                  layer_dropout,
                  attention_dropout,
                  relu_dropout)

        if (self.universal):
            self.dec = DecoderLayer(*params)
        else:
            self.dec = nn.Sequential(*[DecoderLayer(*params) for l in range(num_layers)])

        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)
        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)

    def forward(self, inputs, encoder_output, mask=None):
        mask_src, mask_trg = mask
        dec_mask = torch.gt(mask_trg.bool() + self.mask[:, :mask_trg.size(-1), :mask_trg.size(-1)].bool(), 0).to(device)
        # Add input dropout
        x = self.input_dropout(inputs)
        x = self.embedding_proj(x)

        if (self.universal):
            if (False):
                x, attn_dist, (self.remainders, self.n_updates) = self.act_fn(x, inputs, self.dec, self.timing_signal,
                                                                              self.position_signal, self.num_layers,
                                                                              encoder_output, decoding=True)
                y = self.layer_norm(x)

            else:
                x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)
                for l in range(self.num_layers):
                    x += self.position_signal[:, l, :].unsqueeze(1).repeat(1, inputs.shape[1], 1).type_as(inputs.data)
                    x, _, attn_dist, _ = self.dec(
                        (x, encoder_output, [], (mask_src, dec_mask)))
                y = self.layer_norm(x)
        else:
            # Add timing signal
            x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)

            # Run decoder
            y, _, attn_dist, _ = self.dec((x, encoder_output, [], (mask_src, dec_mask)))

            # Final layer normalization
            y = self.layer_norm(y)

        return y, attn_dist


class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)
        self.emo_proj = nn.Linear(2 * d_model, vocab)
        self.p_gen_linear = nn.Linear(300, 1)

    def forward(self, x, attn_dist=None, enc_batch_extend_vocab=None,
                max_oov_length=None, temp=1, beam_search=False, attn_dist_db=None):
        if False:
            p_gen = self.p_gen_linear(x)
            alpha = torch.sigmoid(p_gen)

        logit = self.proj(x)  # x: (bsz, tgt_len, emb_dim)

        if False:
            vocab_dist = F.softmax(logit / temp, dim=2)
            vocab_dist_ = alpha * vocab_dist

            attn_dist = F.softmax(attn_dist / temp, dim=-1)
            attn_dist_ = (1 - alpha) * attn_dist
            enc_batch_extend_vocab_ = torch.cat([enc_batch_extend_vocab.unsqueeze(1)] * x.size(1),1)  ## extend for all seq

            extra_zeros = Variable(torch.zeros((logit.size(0), max_oov_length))).to(device)
            if extra_zeros is not None:
                extra_zeros = torch.cat([extra_zeros.unsqueeze(1)] * x.size(1), 1)
                vocab_dist_ = torch.cat([vocab_dist_, extra_zeros], 2)

            logit = torch.log(vocab_dist_.scatter_add(2, enc_batch_extend_vocab_, attn_dist_) + 1e-18)

            return logit
        else:
            return F.log_softmax(logit, dim=-1)


class Empdg_G(nn.Module):
    def __init__(self, vocab, emotion_number, model_file_path=None, is_eval=False, load_optim=False):
        '''
        :param decoder_number: the number of emotion labels, i.e., 32
        '''
        super(Empdg_G, self).__init__()
        self.vocab = vocab
        self.vocab_size = vocab.n_words

        self.embedding = share_embedding(self.vocab, False)
        self.semantic_und = Semantic_Encoder(300, 300, num_layers=1,
                                             num_heads=2,
                                             total_key_depth=40, total_value_depth=40,
                                             filter_size=50, universal=False)
        self.emotion_pec = Emotion_Encoder(300, 300, num_layers=1,
                                           num_heads=2,
                                           total_key_depth=40, total_value_depth=40,
                                           filter_size=50, universal=False)
        self.map_emo = {0: 'surprised', 1: 'excited', 2: 'annoyed', 3: 'proud',
                        4: 'angry', 5: 'sad', 6: 'grateful', 7: 'lonely', 8: 'impressed',
                        9: 'afraid', 10: 'disgusted', 11: 'confident', 12: 'terrified',
                        13: 'hopeful', 14: 'anxious', 15: 'disappointed', 16: 'joyful',
                        17: 'prepared', 18: 'guilty', 19: 'furious', 20: 'nostalgic',
                        21: 'jealous', 22: 'anticipating', 23: 'embarrassed', 24: 'content',
                        25: 'devastated', 26: 'sentimental', 27: 'caring', 28: 'trusting',
                        29: 'ashamed', 30: 'apprehensive', 31: 'faithful'}

        ## emotional signal distilling
        self.identify = nn.Linear(300, emotion_number, bias=False)
        self.identify_new = nn.Linear(2*300, emotion_number, bias=False)
        self.activation = nn.Softmax(dim=1)

        ## decoders
        self.emotion_embedding = nn.Linear(emotion_number, 300)
        self.decoder = Decoder(300, hidden_size=300, num_layers=1,
                               num_heads=2,
                               total_key_depth=40, total_value_depth=40,
                               filter_size=50)

        self.decoder_key = nn.Linear(300, emotion_number, bias=False)
        self.generator = Generator(300, self.vocab_size)

        if False:
            # Share the weight matrix between target word embedding & the final logit dense layer
            self.generator.proj.weight = self.embedding.lut.weight

        self.criterion = nn.NLLLoss(ignore_index=1)
        if False:
            self.criterion = LabelSmoothing(size=self.vocab_size, padding_idx=1, smoothing=0.1)
            self.criterion_ppl = nn.NLLLoss(ignore_index=1)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        if False:
            self.optimizer = NoamOpt(300, 1, 8000,
                                     torch.optim.Adam(self.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

        if model_file_path is not None:
            print("loading weights")
            state = torch.load(model_file_path, map_location=lambda storage, location: storage)
            self.encoder.load_state_dict(state['encoder_state_dict'])
            self.decoder.load_state_dict(state['decoder_state_dict'])
            self.generator.load_state_dict(state['generator_dict'])
            self.embedding.load_state_dict(state['embedding_dict'])
            self.decoder_key.load_state_dict(state['decoder_key_state_dict'])
            if load_optim:
                self.optimizer.load_state_dict(state['optimizer'])
            self.eval()

        self.model_dir = "save/EmpDG/"
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.best_path = ""

    def save_model(self, running_avg_ppl, iter, f1_g, f1_b, ent_g, ent_b):
        state = {
            'iter': iter,
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'generator_dict': self.generator.state_dict(),
            'decoder_key_state_dict': self.decoder_key.state_dict(),
            'embedding_dict': self.embedding.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'current_loss': running_avg_ppl
        }
        model_save_path = os.path.join(self.model_dir,
                                       'model_{}_{:.4f}_{:.4f}_{:.4f}_{:.4f}_{:.4f}'.format(iter, running_avg_ppl, f1_g,
                                                                                            f1_b, ent_g, ent_b))
        self.best_path = model_save_path
        torch.save(state, model_save_path)

    def train_one_batch(self, batch, iter, train=True, loss_from_d=0.0):
        enc_batch = batch["context_batch"]
        enc_batch_ext = batch["context_ext_batch"]
        enc_emo_batch = batch['emotion_context_batch']
        enc_emo_batch_ext = batch["emotion_context_ext_batch"]

        oovs = batch["oovs"]
        max_oov_length = len(sorted(oovs, key=lambda i: len(i), reverse=True)[0])

        dec_batch = batch["target_batch"]

        if False:
            self.optimizer.optimizer.zero_grad()
        else:
            self.optimizer.zero_grad()

        ## Semantic Understanding
        mask_semantic = enc_batch.data.eq(1).unsqueeze(1)  # (bsz, src_len)->(bsz, 1, src_len)
        sem_emb_mask = self.embedding(batch["mask_context"])  # dialogue state  E_d
        sem_emb = self.embedding(enc_batch) + sem_emb_mask  # E_w+E_d
        sem_encoder_outputs = self.semantic_und(sem_emb, mask_semantic)  # C_u  (bsz, sem_w_len, emb_dim)

        ## Multi-resolution Emotion Perception (understanding & predicting)
        mask_emotion = enc_emo_batch.data.eq(1).unsqueeze(1)
        # emo_emb_mask = self.embedding(batch["mask_emotion_context"])
        # emo_emb = self.embedding(enc_emo_batch) + emo_emb_mask
        emo_encoder_outputs = self.emotion_pec(self.embedding(enc_emo_batch), mask_emotion)  # C_e  (bsz, emo_w_len, emb_dim)

        # emotion_logit = self.identify(emo_encoder_outputs[:,0,:])  # (bsz, emotion_number)
        emotion_logit = self.identify_new(torch.cat((emo_encoder_outputs[:,0,:],sem_encoder_outputs[:,0,:]), dim=-1))  # (bsz, emotion_number)
        loss_emotion = nn.CrossEntropyLoss(reduction='sum')(emotion_logit, batch['emotion_label'])
        pred_emotion = np.argmax(emotion_logit.detach().cpu().numpy(), axis=1)
        emotion_acc = accuracy_score(batch["emotion_label"].cpu().numpy(), pred_emotion)


        ## Combine Two Contexts
        src_emb = torch.cat((sem_encoder_outputs, emo_encoder_outputs), dim=1)  # (bsz, src_len, emb_dim)
        mask_src = torch.cat((mask_semantic, mask_emotion), dim=2)  # (bsz, 1, src_len)

        ## Empathetic Response Generation
        sos_emb = self.emotion_embedding(emotion_logit).unsqueeze(1)
        dec_emb = self.embedding(dec_batch[:, :-1])
        dec_emb = torch.cat((sos_emb, dec_emb), dim=1)  # (bsz, 1+tgt_len, emb_dim)

        mask_trg = dec_batch.data.eq(1).unsqueeze(1)
        # inputs, encoder_output, pred_emotion=None, emotion_contexts=None, mask=None
        pre_logit, attn_dist = self.decoder(dec_emb, src_emb, (mask_src, mask_trg))

        ## compute output dist
        enc_ext_batch = torch.cat((enc_batch_ext, enc_emo_batch_ext), dim=1)
        logit = self.generator(pre_logit, attn_dist, enc_ext_batch if False else None,
                               max_oov_length, attn_dist_db=None)
        # logit = F.log_softmax(logit,dim=-1) #fix the name later
        ## loss: NNL if ptr else Cross entropy
        loss = self.criterion(logit.contiguous().view(-1, logit.size(-1)), dec_batch.contiguous().view(-1))

        loss += loss_emotion
        loss += loss_from_d

        if False:
            loss_ppl = self.criterion_ppl(logit.contiguous().view(-1, logit.size(-1)),
                                          dec_batch.contiguous().view(-1)).item()

        if train:
            loss.backward()
            self.optimizer.step()

        if False:
            return loss_ppl, math.exp(min(loss_ppl, 100)), loss_emotion.item(), emotion_acc
        else:
            return loss.item(), math.exp(min(loss.item(), 100)), 0, 0

    def compute_act_loss(self, module):
        R_t = module.remainders
        N_t = module.n_updates
        p_t = R_t + N_t
        avg_p_t = torch.sum(torch.sum(p_t, dim=1) / p_t.size(1)) / p_t.size(0)
        loss = 0.001 * avg_p_t.item()
        return loss

#贪婪解码器来对输入序列进行逐步预测并生成输出序列
    def decoder_greedy(self, batch, max_dec_step=30):
        enc_batch_ext, extra_zeros = None, None
        #经过编码器编码后的上下文输入序列以及扩展版本
        enc_batch = batch["context_batch"]
        #表达情感的输入序列
        enc_batch_ext = batch["context_ext_batch"]
        enc_emo_batch = batch['emotion_context_batch']
        enc_emo_batch_ext = batch["emotion_context_ext_batch"]
        #在enc_batch和enc_emo_batch中出现的OOV单词的列表
        oovs = batch["oovs"]
        #在batch的OOV单词列表中最长的OOV单词的长度。
        max_oov_length = len(sorted(oovs, key=lambda i: len(i), reverse=True)[0])

        if False:
            self.optimizer.optimizer.zero_grad()
        else:
            self.optimizer.zero_grad()

        ## Semantic Understanding
        #构建上下文输入序列的掩码，用于在编码器中去标记pad符号
        mask_semantic = enc_batch.data.eq(1).unsqueeze(1)  # (bsz, src_len)->(bsz, 1, src_len)
        #从self.embedding中获取上下文输入序列和状态掩码（即dialogue state）的词向量
        sem_emb_mask = self.embedding(batch["mask_context"])  # dialogue state  E_d
        sem_emb = self.embedding(enc_batch) + sem_emb_mask  # E_w+E_d
        #得到的词向量作为输入传递给语义理解模型（self.semantic_und），并获取其输出（即C_u）
        sem_encoder_outputs = self.semantic_und(sem_emb, mask_semantic)  # C_u  (bsz, sem_w_len, emb_dim)


        # Multi-resolution Emotion Perception (understanding & predicting)
        #构建上下文输入序列的掩码，用于在编码器中标记pad符号
        mask_emotion = enc_emo_batch.data.eq(1).unsqueeze(1)
        # emo_emb_mask = self.embedding(batch["mask_emotion_context"])
        # emo_emb = self.embedding(enc_emo_batch) + emo_emb_mask
        #得到的词向量作为输入传递给多分辨率情感感知模型（self.emotion_pec），并获取其输出（即C_e）
        emo_encoder_outputs = self.emotion_pec(self.embedding(enc_emo_batch), mask_emotion)  # C_e  (bsz, emo_w_len, emb_dim)


        ## Identify
        # emotion_logit = self.identify(emo_encoder_outputs[:,0,:])  # (bsz, emotion_number)
        #情感识别
        #提取emo_encoder的最后一层的第一个时间步的输出作为情感信息的输入，并将其传递识别情绪
        #tensor
        emotion_logit = self.identify_new(torch.cat((emo_encoder_outputs[:,0,:],sem_encoder_outputs[:,0,:]), dim=-1))  # (bsz, emotion_number)

        ## Combine Two Contexts
        #通过联合编码将上下文输入序列和表达情感的输入序列进行组合
        #得到编码器的输入掩码和输入源
        src_emb = torch.cat((sem_encoder_outputs, emo_encoder_outputs), dim=1)  # (bsz, src_len, emb_dim)
        mask_src = torch.cat((mask_semantic, mask_emotion), dim=2)  # (bsz, 1, src_len)
        #将context seq 扩展版和情感表达输入扩展凭借
        enc_ext_batch = torch.cat((enc_batch_ext, enc_emo_batch_ext), dim=1)
        #起始的目标文本序列，初始化生成过程，长度1，value-3
        ys = torch.ones(1, 1).fill_(3).long()
        #情感标签经过嵌入层转化为embedding向量
        #生成过程迭代输入，拼接，将对应的embedding向量拼接到ys_emb中去，实现了输入先前的输出作为输入的循环过程
        ys_emb = self.emotion_embedding(emotion_logit).unsqueeze(1)  # (bsz, 1, emb_dim)
        if False:
            ys = ys.cuda()
        mask_trg = ys.data.eq(1).unsqueeze(1)
        decoded_words = []
        '''
        通过循环进行解码生成，每次迭代中生成的结果都被加入到decoded_words列表中。
        在每次迭代中，输入是源文本序列(src_emb)和上一步生成的目标文本序列的embedding向量(ys_emb)，
        同时还需要提供掩码(mask_src和mask_trg)，用于控制注意力机制的使用。
        在每次迭代中，输出是对应目标序列位置的概率分布(prob)，其由生成器(generator)给出。
        然后，从概率分布(prob)中选择概率最大的词作为下一步的生成结果，
        同时在ys中添加下一步的生成结果，将其转化为embedding向量，并更新mask_trg
        '''
        for i in range(max_dec_step + 1):
            if False:
                out, attn_dist = self.decoder(self.embedding_proj_in(ys_emb), self.embedding_proj_in(src_emb),
                                              (mask_src, mask_trg))
            else:
                out, attn_dist = self.decoder(ys_emb, src_emb, (mask_src, mask_trg))

            prob = self.generator(out, attn_dist, enc_ext_batch, max_oov_length, attn_dist_db=None)
            # logit = F.log_softmax(logit,dim=-1) #fix the name later
            # filtered_logit = top_k_top_p_filtering(logit[:, -1], top_k=0, top_p=0, filter_value=-float('Inf'))
            # Sample from the filtered distribution
            # next_word = torch.multinomial(F.softmax(filtered_logit, dim=-1), 1).squeeze()
            _, next_word = torch.max(prob[:, -1], dim=1)

            batch_words = []
            for i_batch, ni in enumerate(next_word.view(-1)):
                if ni.item() == 2:
                    batch_words.append('<EOS>')
                elif ni.item() in self.vocab.index2word:
                    batch_words.append(self.vocab.index2word[ni.item()])
                else:
                    batch_words.append(oovs[i_batch][ni.item() - self.vocab.n_words])
                    next_word[i_batch] = 0
            decoded_words.append(batch_words)
            next_word = next_word.data[0]

            if False:
                ys = torch.cat([ys, torch.ones(1, 1).long().fill_(next_word).cuda()], dim=1)
                ys = ys.cuda()
                ys_emb = torch.cat((ys_emb, self.embedding(torch.ones(1, 1).long().fill_(next_word).cuda())), dim=1)
            else:
                ys = torch.cat([ys, torch.ones(1, 1).long().fill_(next_word)], dim=1)
                ys_emb = torch.cat((ys_emb, self.embedding(torch.ones(1, 1).long().fill_(next_word))), dim=1)
            mask_trg = ys.data.eq(1).unsqueeze(1)

        sent = []
        for _, row in enumerate(np.transpose(decoded_words)):
            st = ''
            for e in row:
                if e == '<EOS>':
                    break
                else:
                    st += e + ' '
            sent.append(st)
        return sent

    def predict(self, batch, max_dec_step=30):
        enc_batch_ext, extra_zeros = None, None
        enc_batch = batch["context_batch"]
        enc_batch_ext = batch["context_ext_batch"]
        enc_emo_batch = batch['emotion_context_batch']
        enc_emo_batch_ext = batch["emotion_context_ext_batch"]

        oovs = batch["oovs"]
        max_oov_length = len(sorted(oovs, key=lambda i: len(i), reverse=True)[0])

        ## Semantic Understanding
        mask_semantic = enc_batch.data.eq(1).unsqueeze(1)  # (bsz, src_len)->(bsz, 1, src_len)
        sem_emb_mask = self.embedding(batch["mask_context"])  # dialogue state  E_d
        sem_emb = self.embedding(enc_batch) + sem_emb_mask  # E_w+E_d
        sem_encoder_outputs = self.semantic_und(sem_emb, mask_semantic)  # C_u  (bsz, sem_w_len, emb_dim)

        # Multi-resolution Emotion Perception (understanding & predicting)
        mask_emotion = enc_emo_batch.data.eq(1).unsqueeze(1)
        # emo_emb_mask = self.embedding(batch["mask_emotion_context"])
        # emo_emb = self.embedding(enc_emo_batch) + emo_emb_mask
        emo_encoder_outputs = self.emotion_pec(self.embedding(enc_emo_batch), mask_emotion)  # C_e  (bsz, emo_w_len, emb_dim)

        ## Identify
        # emotion_logit = self.identify(emo_encoder_outputs[:,0,:])  # (bsz, emotion_number)
        emotion_logit = self.identify_new(torch.cat((emo_encoder_outputs[:,0,:],sem_encoder_outputs[:,0,:]), dim=-1))  # (bsz, emotion_number)

        ## Combine Two Contexts
        src_emb = torch.cat((sem_encoder_outputs, emo_encoder_outputs), dim=1)  # (bsz, src_len, emb_dim)
        mask_src = torch.cat((mask_semantic, mask_emotion), dim=2)  # (bsz, 1, src_len)
        enc_ext_batch = torch.cat((enc_batch_ext, enc_emo_batch_ext), dim=1)

        ys = torch.ones(enc_batch.size(0), 1).fill_(3).long()
        ys_emb = self.emotion_embedding(emotion_logit).unsqueeze(1)  # (bsz, 1, emb_dim)
        if False:
            ys = ys.cuda()
        mask_trg = ys.data.eq(1).unsqueeze(1)
        decoded_words = []
        for i in range(max_dec_step + 1):
            if False:
                out, attn_dist = self.decoder(self.embedding_proj_in(ys_emb), self.embedding_proj_in(src_emb),
                                              (mask_src, mask_trg))
            else:
                out, attn_dist = self.decoder(ys_emb, src_emb, (mask_src, mask_trg))

            prob = self.generator(out, attn_dist, enc_ext_batch, max_oov_length, attn_dist_db=None)
            # logit = F.log_softmax(logit,dim=-1) #fix the name later
            # filtered_logit = top_k_top_p_filtering(logit[:, -1], top_k=0, top_p=0, filter_value=-float('Inf'))
            # Sample from the filtered distribution
            # next_word = torch.multinomial(F.softmax(filtered_logit, dim=-1), 1).squeeze()
            _, next_word = torch.max(prob[:, -1], dim=1)
            decoded_words.append(['<EOS>' if ni.item() == 2 else self.vocab.index2word[ni.item()] for ni in
                                  next_word.view(-1)])
            next_word = next_word.data[0]

            if False:
                ys = torch.cat([ys, torch.ones(enc_batch.size(0), 1).long().fill_(next_word).cuda()], dim=1)
                ys = ys.cuda()
                ys_emb = torch.cat((ys_emb, self.embedding(torch.ones(enc_batch.size(0), 1).long().fill_(next_word).cuda())), dim=1)
            else:
                ys = torch.cat([ys, torch.ones(enc_batch.size(0), 1).long().fill_(next_word)], dim=1)
                ys_emb = torch.cat((ys_emb, self.embedding(torch.ones(enc_batch.size(0), 1).long().fill_(next_word))), dim=1)
            mask_trg = ys.data.eq(1).unsqueeze(1)

        sent = []
        for _, row in enumerate(np.transpose(decoded_words)):
            st = ''
            for e in row:
                if e == '<EOS>':
                    break
                else:
                    st += e + ' '
            sent.append(st)
        return sent

    def g_for_d(self, batch, max_dec_step=30):
        enc_batch_ext, extra_zeros = None, None
        enc_batch = batch["context_batch"]
        enc_batch_ext = batch["context_ext_batch"]
        enc_emo_batch = batch['emotion_context_batch']
        enc_emo_batch_ext = batch["emotion_context_ext_batch"]

        oovs = batch["oovs"]
        max_oov_length = len(sorted(oovs, key=lambda i: len(i), reverse=True)[0])

        ## Semantic Understanding
        mask_semantic = enc_batch.data.eq(1).unsqueeze(1)  # (bsz, src_len)->(bsz, 1, src_len)
        sem_emb_mask = self.embedding(batch["mask_context"])  # dialogue state  E_d
        sem_emb = self.embedding(enc_batch) + sem_emb_mask  # E_w+E_d
        sem_encoder_outputs = self.semantic_und(sem_emb, mask_semantic)  # C_u  (bsz, sem_w_len, emb_dim)

        # Multi-resolution Emotion Perception (understanding & predicting)
        mask_emotion = enc_emo_batch.data.eq(1).unsqueeze(1)
        # emo_emb_mask = self.embedding(batch["mask_emotion_context"])
        # emo_emb = self.embedding(enc_emo_batch) + emo_emb_mask
        emo_encoder_outputs = self.emotion_pec(self.embedding(enc_emo_batch), mask_emotion)  # C_e  (bsz, emo_w_len, emb_dim)

        ## Identify
        # emotion_logit = self.identify(emo_encoder_outputs[:,0,:])  # (bsz, emotion_number)
        emotion_logit = self.identify_new(torch.cat((emo_encoder_outputs[:,0,:],sem_encoder_outputs[:,0,:]), dim=-1))  # (bsz, emotion_number)

        ## Combine Two Contexts
        src_emb = torch.cat((sem_encoder_outputs, emo_encoder_outputs), dim=1)  # (bsz, src_len, emb_dim)
        mask_src = torch.cat((mask_semantic, mask_emotion), dim=2)  # (bsz, 1, src_len)
        enc_ext_batch = torch.cat((enc_batch_ext, enc_emo_batch_ext), dim=1)

        ys = torch.ones(enc_batch.size(0), 1).fill_(3).long()
        ys_emb = self.emotion_embedding(emotion_logit).unsqueeze(1)  # (bsz, 1, emb_dim)
        if False:
            ys = ys.cuda()
        mask_trg = ys.data.eq(1).unsqueeze(1)
        decoded_words = []
        for i in range(max_dec_step + 1):
            if False:
                out, attn_dist = self.decoder(self.embedding_proj_in(ys_emb), self.embedding_proj_in(src_emb),
                                              (mask_src, mask_trg))
            else:
                out, attn_dist = self.decoder(ys_emb, src_emb, (mask_src, mask_trg))

            prob = self.generator(out, attn_dist, enc_ext_batch, max_oov_length, attn_dist_db=None)
            # logit = F.log_softmax(logit,dim=-1) #fix the name later
            # filtered_logit = top_k_top_p_filtering(logit[:, -1], top_k=0, top_p=0, filter_value=-float('Inf'))
            # Sample from the filtered distribution
            # next_word = torch.multinomial(F.softmax(filtered_logit, dim=-1), 1).squeeze()
            _, next_word = torch.max(prob[:, -1], dim=1)
            decoded_words.append(['<EOS>' if ni.item() == 2 else self.vocab.index2word[ni.item()] for ni in
                                  next_word.view(-1)])
            next_word = next_word.data[0]

            if False:
                ys = torch.cat([ys, torch.ones(enc_batch.size(0), 1).long().fill_(next_word).cuda()], dim=1)
                ys = ys.cuda()
                ys_emb = torch.cat((ys_emb, self.embedding(torch.ones(enc_batch.size(0), 1).long().fill_(next_word).cuda())), dim=1)
            else:
                ys = torch.cat([ys, torch.ones(enc_batch.size(0), 1).long().fill_(next_word)], dim=1)
                ys_emb = torch.cat((ys_emb, self.embedding(torch.ones(enc_batch.size(0), 1).long().fill_(next_word))), dim=1)
            mask_trg = ys.data.eq(1).unsqueeze(1)

        sent = []
        sent_emo = []
        for _, row in enumerate(np.transpose(decoded_words)):
            st = []
            for e in row:
                if e == '<EOS>':
                    break
                else:
                    st.append(e)
            sent.append(st)
            sent_emo.append(get_emotion_words(st))
        return sent, sent_emo, sem_encoder_outputs[:, 0, :], emo_encoder_outputs[:, 0, :]

