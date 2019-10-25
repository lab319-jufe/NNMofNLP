class Config():
    ptb_dir = "E:/OneDrive/Documents/Ma&St-learning/NLP/simple-examples/data"
    ptb_train = "ptb.train.txt"
    ptb_test = "ptb.test.txt"
    ptb_valid = "ptb.valid.txt"
    vocab_size = 10000
    embedding_size = 300
    num_step = 30
    keep_prob = 0.5
    number_layers = 2
    hidden_size = 512
    batch_size = 128
    lr = 1e-2
    epoches = 20
    Training = True
    share_emb_and_softmax = False
    save_path = "model"
    model_name = "nnlm.ckpt"