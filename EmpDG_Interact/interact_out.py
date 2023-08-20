# Use this script to interact with the trained model.
import pdb
import torch
import torch.utils.data as data
from collections import deque

from utils.data_loader import prepare_data_seq
import json
import nltk
from Model.Empdg_G import Empdg_G
#调用 prepare_data_seq 函数来准备数据集，返回四个数据集迭代器，一个词汇表和情感类别数。其中，batch_size 是每个迭代器返回的数据批量大小。
data_loader_tra, data_loader_val, data_loader_tst, vocab, program_number = prepare_data_seq(batch_size=16)
#定义一个 Empdg_G 模型，并将词汇表和情感类别数作为参数传入
model = Empdg_G(vocab, emotion_number=program_number)
# print("config_model"+config.model)
#调用 PyTorch 的 load 函数从已经保存的模型参数文件中加载模型的权重。
checkpoint = torch.load('result/EmpDG_best.tar', map_location=lambda storage, location: storage)
weights_best = checkpoint['models_g']
model.load_state_dict({name: weights_best[name] for name in weights_best})
print(model)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)
#将模型设为评估模式。
model.eval()

print('Let\'s chat')
#设置对话框大小和上下文对话队列。
DIALOG_SIZE = 5
context = deque(DIALOG_SIZE * ['None'], maxlen=DIALOG_SIZE)
#加载了一个情感词典文件 EMODICT，该文件存储了一些与情感相关的词汇
EMODICT = json.load(open('empathetic-dialogue/NRCDict.json'))[0]

'''
该函数的作用是从一个句子中提取情感单词。该函数接受一个参数 utt_words，该参数是一个由单词组成的列表。该函数首先创建一个空列表 emo_ws，
然后对 utt_words 中的每个单词进行遍历，如果该单词在 EMODICT（即一个保存情感单词的字典） 中出现
，则将其添加到 emo_ws 列表中。最后，该函数返回列表 emo_ws。
'''
def get_emotion_words(utt_words):
    emo_ws = []
    for u in utt_words:
        for w in u.split():
            if w in EMODICT:
                emo_ws.append(w)
    return emo_ws

#字典将缩写词替换为完整形式，以便更好地理解和处理句子
word_pairs = {"it's": "it is", "don't": "do not", "doesn't": "does not", "didn't": "did not", "you'd": "you would",
                  "you're": "you are", "you'll": "you will", "i'm": "i am", "they're": "they are", "that's": "that is",
                  "what's": "what is", "couldn't": "could not", "i've": "i have", "we've": "we have", "can't": "cannot",
                  "i'd": "i would", "i'd": "i would", "aren't": "are not", "isn't": "is not", "wasn't": "was not",
                  "weren't": "were not", "won't": "will not", "there's": "there is", "there're": "there are"}


#将输入的句子进行预处理和清洗，使其更适合用于自然语言处理模型的输入,使用nltk库的word_tokenize()方法将句子分词，得到一个单词列表。最后，函数返回分词后的句子列表
def clean(sentence, word_pairs):
    sentence = sentence.lower()
    for k, v in word_pairs.items():
        sentence = sentence.replace(k,v)
    sentence = nltk.word_tokenize(sentence)
    return sentence

class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, data, vocab):
        """Reads source and target sequences from txt files."""
        self.vocab = vocab
        self.data = data
        self.emo_map = {
            'surprised': 0, 'excited': 1, 'annoyed': 2, 'proud': 3, 'angry': 4, 'sad': 5, 'grateful': 6, 'lonely': 7,
            'impressed': 8, 'afraid': 9, 'disgusted': 10, 'confident': 11, 'terrified': 12, 'hopeful': 13,
            'anxious': 14, 'disappointed': 15,
            'joyful': 16, 'prepared': 17, 'guilty': 18, 'furious': 19, 'nostalgic': 20, 'jealous': 21,
            'anticipating': 22, 'embarrassed': 23,
            'content': 24, 'devastated': 25, 'sentimental': 26, 'caring': 27, 'trusting': 28, 'ashamed': 29,
            'apprehensive': 30, 'faithful': 31}
        self.map_emo = {0: 'surprised', 1: 'excited', 2: 'annoyed', 3: 'proud',
                        4: 'angry', 5: 'sad', 6: 'grateful', 7: 'lonely', 8: 'impressed',
                        9: 'afraid', 10: 'disgusted', 11: 'confident', 12: 'terrified',
                        13: 'hopeful', 14: 'anxious', 15: 'disappointed', 16: 'joyful',
                        17: 'prepared', 18: 'guilty', 19: 'furious', 20: 'nostalgic',
                        21: 'jealous', 22: 'anticipating', 23: 'embarrassed', 24: 'content',
                        25: 'devastated', 26: 'sentimental', 27: 'caring', 28: 'trusting',
                        29: 'ashamed', 30: 'apprehensive', 31: 'faithful'}

#返回一个数据样本，将数据处理为模型所需的格式并返回一个字典
    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        item = {}

        item["context_text"] = [x for x in self.data if x!="None"]
        item["emotion_context_text"] = get_emotion_words(item["context_text"])

        inputs = self.preprocess([item["context_text"],
                                  item["emotion_context_text"]])

        item["context"], item["context_ext"], item["oovs"], item["context_mask"], \
        item["emotion_context"], item["emotion_context_ext"], item["emotion_context_mask"] = inputs

        return item

    def __len__(self):
        return 1
#将目标序列中的 oov 单词转换为模型可读的格式（ids）
#对于已知的词汇表中不存在的单词，使用 oovs 列表中的 id，否则返回该单词的 id。
#最后，将序列的结尾添加一个结束标记符 2
    def target_oovs(self, target, oovs):  #
        ids = []
        for w in target:
            if w not in self.vocab.word2index:
                if w in oovs:
                    ids.append(len(self.vocab.word2index) + oovs.index(w))
                else:
                    ids.append(0)
            else:
                ids.append(self.vocab.word2index[w])
        ids.append(2)
        return torch.LongTensor(ids)

    '''函数用于将上下文和情感上下文中的
    oov(out-of-vocabulary)在词汇表中不存在的单词，
    如果某个单词在词汇表中不存在，我们就需要对它进行特殊处理。
    在这个代码中，当遇到oov时，将它添加到一个列表中，
    并用它的索引表示在词汇表中的位置。
    这些oov索引可以与其他单词的索引一起输入模型，
    使其能够处理这些不在词汇表中的单词。
    单词转换为模型可读的格式（ids)'''
    def process_oov(self, context, emotion_context):  # oov for input
        ids = []
        ids_e = []
        oovs = []
        for si, sentence in enumerate(context):
            sentence = clean(sentence, word_pairs)
            for w in sentence:
                if w in self.vocab.word2index:
                    i = self.vocab.word2index[w]
                    ids.append(i)
                else:
                    if w not in oovs:
                        oovs.append(w)
                    oov_num = oovs.index(w)
                    ids.append(len(self.vocab.word2index) + oov_num)

        for ew in emotion_context:
            if ew in self.vocab.word2index:
                i = self.vocab.word2index[ew]
                ids_e.append(i)
            elif ew in oovs:
                oov_num = oovs.index(ew)
                ids_e.append(len(self.vocab.word2index) + oov_num)
            else:
                oovs.append(ew)
                oov_num = oovs.index(w)
                ids_e.append(len(self.vocab.word2index) + oov_num)
        return ids, ids_e, oovs
    '''
    上下文中的句子拼接在一起，形成一个长的序列，
    并在序列的开头添加了一个特殊的标记符 6（X_dial）和一个结束标记符 2
    '''
    def preprocess(self, arr, anw=False):
        """Converts words to ids."""
        if anw:
            sequence = [self.vocab.word2index[word] if word in self.vocab.word2index else 0 for word in arr] + [2]
            return torch.LongTensor(sequence)
        else:
            #原始文本的上下文转化为ids
            context = arr[0]
            #情感上下文
            emotion_context = arr[1]
            #序列开头
            X_dial = [6]
            X_dial_ext = [6]
            X_dial_mask = [6]

            X_emotion = [7]
            X_emotion_ext = [7]
            X_emotion_mask = [7]


            for i, sentence in enumerate(context):  # concat sentences in context
                sentence = clean(sentence, word_pairs)
                X_dial += [self.vocab.word2index[word] if word in self.vocab.word2index else 0 for word in sentence]
                spk = self.vocab.word2index["USR"] if i % 2 == 0 else self.vocab.word2index["SYS"]
                X_dial_mask += [spk for _ in range(len(sentence))]

            for i, ew in enumerate(emotion_context):
                X_emotion += [self.vocab.word2index[ew] if ew in self.vocab.word2index else 0]
                X_emotion_mask += [self.vocab.word2index["LAB"]]

            X_ext, X_e_ext, X_oovs = self.process_oov(context, emotion_context)
            X_dial_ext += X_ext
            X_emotion_ext += X_e_ext

            assert len(X_dial) == len(X_dial_mask) == len(X_dial_ext)
            assert len(X_emotion) == len(X_emotion_ext) == len(X_emotion_mask)

            return X_dial, X_dial_ext, X_oovs, X_dial_mask, X_emotion, X_emotion_ext, X_emotion_mask

    def preprocess_emo(self, emotion, emo_map):
        program = [0]*len(emo_map)
        program[emo_map[emotion]] = 1
        return program, emo_map[emotion]  # one

#将一个batch的数据整理成模型需要的格式
def collate_fn(batch_data):
    ## input - context
    context_batch = torch.LongTensor([batch_data[0]['context']])
    context_ext_batch = torch.LongTensor([batch_data[0]['context_ext']])
    mask_context = torch.LongTensor([batch_data[0]['context_mask']])  # (bsz, max_context_len) dialogue state

    ## input - emotion_context
    emotion_context_batch = torch.LongTensor([batch_data[0]['emotion_context']])
    emotion_context_ext_batch = torch.LongTensor([batch_data[0]['emotion_context_ext']])
    mask_emotion_context = torch.LongTensor([batch_data[0]['emotion_context_mask']])

    d = {}
    ##input
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d["context_batch"] = context_batch.to(device)  # (bsz, max_context_len)
    d["context_ext_batch"] = context_ext_batch.to(device)  # (bsz, max_context_len)
    d["mask_context"] = mask_context.to(device)

    d["emotion_context_batch"] = emotion_context_batch.to(device)  # (bsz, max_emo_context_len)
    d["emotion_context_ext_batch"] = emotion_context_ext_batch.to(device)  # (bsz, max_emo_context_len)
    d["mask_emotion_context"] = mask_emotion_context.to(device)

    ##text
    d["context_txt"] = [batch_data[0]['context_text']]
    d["emotion_context_txt"] = [batch_data[0]['emotion_context_text']]
    d["oovs"] = [batch_data[0]["oovs"]]
    return d

#根据给定的输入数据和词汇表生成一个批次的数据,如果需要多個批次，可以多次調用
def make_batch(inp,vacab):
    d = Dataset(inp,vacab)
    loader = torch.utils.data.DataLoader(dataset=d, batch_size=1, shuffle=False, collate_fn=collate_fn)
    return iter(loader).__next__()


def interact(input_message):

    while True:
        #ipt = input(">> User: ")
        if (len(str(input_message).strip()) != 0):
            context.append(str(input_message).rstrip().lstrip())
            #转换为模型输入的格式
            batch = make_batch(context, vocab)
            #生成回复
            sent_g = model.decoder_greedy(batch, max_dec_step=30)
            #print("{}: ".format(config.model), sent_g[0])
            context.append(sent_g[0])
            with open('output_EmpDG.txt', 'w') as f:
                for element in context:
                    f.write(str(element) + '\n')
            f.close()
            return sent_g[0]



