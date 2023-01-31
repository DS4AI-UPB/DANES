# coding: utf-8

__author__      = "Ciprian-Octavian Truică"
__copyright__   = "Copyright 2021, University Politehnica of Bucharest"
__license__     = "GNU GPL"
__version__     = "0.1"
__email__       = "ciprian.truica@upb.ro"
__status__      = "Development"

import pandas as pd
from scipy import io as sio
from wordembeddings import WordEmbeddings
from tokenization import Tokenization
import numpy as np

from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import sys

tkn = Tokenization()

def processElement(elem):
    id_line = elem[0]
    text = elem[1]
    text = tkn.createCorpus(text, remove_stopwords=False)
    return id_line, text

if __name__ == "__main__":
    fn = sys.argv[1] # filename
    df = pd.read_csv(fn, sep=',')
    # columns = ['id', 'content', 'label', 'num_reactions', 'num_comments', 'num_shares', 'num_likes', 'num_loves', 'num_wows', 'num_hahas', 'num_sads', 'num_angrys' ]
    print(df)

    
    labels = df['label'].unique()
    num_classes = len(labels)
    id2label = {}
    counts = {}
    idx = 0
    label2id = {'mostly true': 0, 'mixture of true and false': 1, 'no factual content': 1, 'mostly false': 1}
    for label in labels:
        df.loc[df['label'] == label, 'label'] = label2id[label]
        idx += 1


    y = df['label'].astype(int).to_list()

    sio.savemat('labels.mat', {'y': y})
        
    X_network = df[['num_comments', 'num_shares', 'num_likes', 'num_loves', 'num_wows', 'num_hahas', 'num_sads', 'num_angrys']].to_numpy()
    scaler_std = StandardScaler()
    X_net_std = scaler_std.fit_transform(X_network)
    X_net_std = X_net_std.reshape((X_net_std.shape[0], 1, X_net_std.shape[1]))
    print(X_network)
    # print(X_net_mm)
    print(X_net_std)

    sio.savemat('network.mat', {'X_net_std': X_net_std})

    print("Start Tokenization")
    texts = df[['id', 'content']].to_numpy().tolist()
    corpus = [None] * len(texts)
    no_threads = cpu_count() - 1
    with ProcessPoolExecutor(max_workers=no_threads) as worker:
        for result in worker.map(processElement, texts):
            if result:
                corpus[result[0]] = result[1]

    idx = 0
    for doc in corpus:
        print(idx, doc)
        idx += 1

    print("Start Document Tokenization")
    X_docs = []
    we = WordEmbeddings(corpus)
    documents = we.preprareDocuments()
    vocabulary_size = we.no_words
    max_size = we.max_size
    print("Vocabulary size", vocabulary_size)
    print("Max Documnet size", max_size)
    for document in documents:
        doc_size = len(document)
        X_docs.append(document + [0] * (max_size - doc_size) )
    X_docs = np.array(X_docs)

    sio.savemat('corpus.mat', {'X': X_docs})

    print("Start W2V CBOW")
    w2v_cbow = we.word2vecEmbedding(sg=0)
    sio.savemat('w2v_cbow.mat', {'w2v_cbow': w2v_cbow})

    print("Start W2V SG")
    w2v_sg = we.word2vecEmbedding(sg=1)
    sio.savemat('w2v_sg.mat', {'w2v_sg': w2v_sg})

    print("Start FT CBOW")
    ft_cbow = we.word2FastTextEmbeddings(sg=0)
    sio.savemat('ft_cbow.mat', {'ft_cbow': ft_cbow})

    print("Start FT SG")
    ft_sg = we.word2FastTextEmbeddings(sg=1)
    sio.savemat('ft_sg.mat', {'ft_sg': ft_sg})

    print("Start GLOVE")
    glove = we.word2GloVeEmbedding()
    sio.savemat('glove.mat', {'glove': glove})

    print("Start MITTENS")
    mittens = we.word2MittensEmbedding()
    sio.savemat('mittens.mat', {'mittens': mittens})

