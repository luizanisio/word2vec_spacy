# word2vec_spacy
Passo a passo para criar um word2vec no gensim e importar no Spacy

A criação de um word2vec aumenta a acurácia na comparação de documentos e termos.

O Spacy tem um modelo treinado para o português, mas a similaridade pode ser melhorada para domínios específicos com a criação de vetores próprios para esses domínios. 

Códigos na pasta <b>codigo_w2v</b>

## Passo a passo
Este tutorial foi baseado nos links abaixo e em alguns fóruns pela web.
- http://kavita-ganesan.com/gensim-word2vec-tutorial-starter-code/#.W4MNt-hKi02
- https://github.com/kavgan/nlp-text-mining-working-examples/blob/master/word2vec/scripts/word2vec.py

### 1. Juntar um volume de textos brutos (txt) do domínio desejado.
Podem ser textos médicos, jurídicos, de psicologia, de uma área técnica qualquer, receitas etc. 
É difícil de definir o número de documentos necessários para a criação dos vetores, mas alguns milhares parece um bom número para começar. Uma prova de conceito com 10 mil documentos jurídicos mostrou resultados bons para começar.
- o exemplo aqui foi gerado com uma base de 25 textos jurídicos diversos baixados na interent. Não é um volume representativo, mas já apresenta alguns resultados interessantes.

### 2. A geração do modelo
Foram realizados testes com o código de <b>Kavita Ganesan</b> que pode ser baixado no Github https://github.com/kavgan/nlp-text-mining-working-examples/blob/master/word2vec/scripts/word2vec.py
- na versão disponibilizada aqui foram feitos alguns ajustes para facilitar a importação do modelo para o Spacy e ler todos os arquivos que estão na pasta <b>textos</b>
- para usar o código exemplo, basta criar uma pasta <b>textos</b> e colocar quantos documentos achar necessário, e rodar o código. Ele vai carregar todos os documentos texto da pasta para gerar o modelo.
- ao final, será criado o arquivo <b>\vectors\vetores.txt</b> que será convertido para o formato do Spacy.

### 3. Convertendo o modelo para usar no Spacy

A conversão do modelo gerado pelo Gensim para o formato do Spacy é feita pela linha de comando:
```bat
python -m spacy init-model pt vectors_spacy --vectors-loc .\vectors\vetores.txt
```

Será criada a pasta <b>vectors_spacy<b> que poderá ser carregada pelo Spacy, como no exemplo abaixo.
```py
from spacy import util as spc_util
import pt_core_news_sm

nlp = pt_core_news_sm.load()

doc=nlp('justiça tribunal')
tk=[k for k in doc]
print('Semelhança entre justiça e tribunal = ',tk[0].similarity(tk[1]))

pathw2v = './vectors_spacy'
spc_util.load_model(pathw2v, vocab=nlp.vocab)

doc=nlp('justiça tribunal')
tk=[k for k in doc]
print('Semelhança entre justiça e tribunal = ',tk[0].similarity(tk[1]))
```

Como exemplo, termos a saída das comparações dos termos <b>justiça</b> e <b>tribunal</b> com e sem o modelo gerado pelos 25 documentos:
- <b>Sem o uso do modelo gerado:</b> Semelhança entre justiça e tribunal =  0.31015217
- <b>Com o uso do modelo gerado:</b> Semelhança entre justiça e tribunal =  0.8437208
 
### 4. Utilizando o <b>tensorboard</b> para visualizar o modelo criado e testar algumas similaridades

O tensorboard é uma ferramenta excelente para visualizar o modelo criado com a rederização de uma nuvem de vetores onde os termos ficam próximos dos seus semelhantes, o exemplo <b>vectors_tensorboard.py</b> foi baseado no código abaixo:
- https://github.com/explosion/spaCy/blob/master/examples/vectors_tensorboard.py

Após gerar os arquivos de visualização na pasta <b>tensorboard_out</b>, é necessário rodar a linha abaixo para abrir o servidor e acessar o browser (dica: use o chrome) no endereço http://localhost:6006
```bat
tensorboard --host=0.0.0.0 --logdir=.\tensorboard_out
```



