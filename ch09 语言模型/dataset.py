import os
import json

class Dataset(object):
    def __init__(self):
        self.ptb_dir = "PTB_dataset"
        self.ptb_train = "ptb.train.txt"
        self.ptb_test = "ptb.test.txt"
        self.ptb_valid = "ptb.valid.txt"
        if os.path.exists("word2id.txt"):
            self.word2id = json.load(open("word2id.txt", encoding="utf-8"))
        else:
            self.create_vocab()
        if os.path.exists("id2word.txt"):
            self.id2word = json.load(open("id2word.txt", encoding="utf-8"))
        else:
            self.create_vocab()

    def create_vocab(self):
        vocab = {"EOF"}
        with open(os.path.join(self.ptb_dpythir, self.ptb_train), encoding="utf-8") as f:
            for line in f.readlines():
                vocab = vocab.union(set(line.strip().split()))
        print(len(vocab))
        self.word2id = dict(zip(vocab, range(len(vocab))))
        self.id2word = dict(zip(range(len(vocab)), vocab))
        with open("word2id.txt", "w", encoding="utf-8") as f:
            f.write(json.dumps(self.word2id))
        with open("id2word.txt", "w", encoding="utf-8") as f:
            f.write(json.dumps(self.id2word))
    
    def W2ID(self, sentence):
        return list(map(lambda word:self.word2id[word], sentence))
    
    def ID2W(self, ids):
        return list(map(lambda id:self.id2word[str(id)], ids))
    
    def convert(self, source, target):
        fin = open(os.path.join(self.ptb_dir, source))
        fout = open(os.path.join(self.ptb_dir, target), 'w', encoding="utf-8")
        for line in fin.readlines():
            words = line.strip().split() + ['EOF']
            # 将每个单词替换为词汇表中的编号
            ids = self.W2ID(words)
            out_line = " ".join([str(id) for id in ids])
            fout.write(out_line+"\n")
        fin.close()
        fout.close()
        


if __name__ ==  "__main__":
    dataset = Dataset()

    dataset.convert("ptb.train.txt", "train_ids.txt")
    dataset.convert("ptb.valid.txt", "valid_ids.txt")
    dataset.convert("ptb.test.txt", "test_ids.txt")
    '''
    sent = " mr. <unk> is chairman of <unk> n.v. the dutch publishing group "
    sent = sent.split()
    ids = dataset.W2ID(sent)
    sents = dataset.ID2W(ids)
    print(ids)
    print(sents)
    '''