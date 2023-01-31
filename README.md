## DANES

### Packages

Python >= 3.9
- SciPy
- Pandas
- numpy
- SciKit-Learn
- matplotlib
- tensorflow
- stop_words
- nltk
- SpaCy

### Utilization

To process the text and create both word embeddings and social context embeddings use

`python create_embeddings.py FILE_NAME`

The FILE_NAME is a csv file with the followind columns \['id', 'content', 'label', 'num_reactions', 'num_comments', 'num_shares', 'num_likes', 'num_loves', 'num_wows', 'num_hahas', 'num_sads', 'num_angrys'\].
The output of this script is
- corpus.mat - the tokenize corpus 
- network.mat -  the social context embeddings
- w2v_cbow.mat - the Word2Vec CBWO embeddings
- w2v_sg.mat - the Word2Vec Skip-Gram embeddings
- ft_cbow.mat - the FastText CBOW embeddings
- ft_sg.mat - the FastText Skip-Gram embeddings
- glove.mat - the GloVe embeddings
- mittens.mat - the Mittens embeddings

To train the \[Bi\]GRU DANES vesion use

`python danes_gru.py`


To train the \[Bi\]LSTM DANES vesion use

`python danes_lstm.py`


