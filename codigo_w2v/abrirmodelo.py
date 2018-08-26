from spacy import util as spc_util
import pt_core_news_sm

nlp = pt_core_news_sm.load()

doc=nlp('justica tribunal')
tk=[k for k in doc]
print('Semelhança entre justica e tribunal = ',tk[0].similarity(tk[1]))

pathw2v = './vectors_spacy'
spc_util.load_model(pathw2v, vocab=nlp.vocab)

doc=nlp('justica tribunal')
tk=[k for k in doc]
print('Semelhança entre justica e tribunal = ',tk[0].similarity(tk[1]))

