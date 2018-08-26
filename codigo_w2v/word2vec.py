import gensim
import os
import glob
import re
import pathlib
import logging

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO)

def removeacentos(texto):
    txt = re.sub(r"\s+", " ", texto).lower()
    acentos = [('áâàãä', 'a'),
               ('éèêë', 'e'),
               ('íìîï', 'i'),
               ('óòôöõ', 'o'),
               ('úùüû', 'u'),
               ('ç', 'c'),
               ('ñ', 'n')]
    for de, para in acentos:
        txt = re.sub(r"[{}]".format(de), para, txt)
    return txt

def read_texts():
    logging.info("Procurando os arquivos .... ")
    arqs = [f for f in glob.glob(r".\textos\*.txt")]
    logging.info("Lendo {} arquivos .... ".format(len(arqs)))
    for arq in arqs:
        with open(arq, encoding='utf-8', errors='ignore') as f:
            logging.info(" - {}".format(arq))
            lines = f.read().splitlines()
            yield gensim.utils.simple_preprocess(removeacentos(''.join(lines)))

if __name__ == '__main__':
    out_loc = os.getcwd()+'/vectors'
    pathlib.Path(out_loc).mkdir(parents=True, exist_ok=True)
    documents = list(read_texts())
    logging.info("Documentos carregados")

    logging.info('Gerando vetores dos {} documentos...'.format(len(documents)))
    model = gensim.models.Word2Vec(
        documents,
        size=300,
        window=10,
        min_count=2,
        workers=10)
    model.train(documents, total_examples=len(documents), epochs=10)

    # gravando o modelo em formato texto para ser convertido para o spacy
    logging.info('Gravando o modelo...')
    model.wv.save_word2vec_format("./vectors/vetores.txt", binary=False)

    logging.info('Realizando testes...')

    w1 = "tribunal"
    try:
        print("\nTermo mais similar a {}".format(w1), model.wv.most_similar(positive=w1))
    except:
        print('Erro para: ',w1)

    w1 = ["ministro"]
    try:
        print(
            "\nTermos mais similares a {}".format(w1),
            model.wv.most_similar( positive=w1, topn=6))
    except:
        print('Erro para: ',w1)

    w1 = ["pena"]
    w2 = ["advogado"]
    try:
        print("\nMost similar to {} e não a {}".format(w1,w2),
            model.wv.most_similar(positive=w1, negative=w2, topn=10))
    except:
        print('Erro para: ',w1,' ou ', w2)

