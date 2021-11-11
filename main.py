import spacy
import neuralcoref
from nltk import Tree
# from transformers import pipeline
#
# nlp2 = pipeline("ner")
# import nltk
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('maxent_ne_chunker')
# nltk.download('words')
# from allennlp_models import pretrained
# al = pretrained.load_predictor("tagging-fine-grained-transformer-crf-tagger")
# from polyglot.text import Text

def to_nltk_tree(node):
    if node.n_lefts + node.n_rights > 0:
        return Tree(node.orth_, [to_nltk_tree(child) for child in node.children])
    else:
        return node.orth_

nlp = spacy.load("en_core_web_lg")           # load package "en_core_web_sm"
neuralcoref.add_to_pipe(nlp)

baseDir = "D:\\Downloads\\Documents\\CS699\\AC3R-Demo-AC3RPlus\\accidentCases\\accidentCase0-6\\accidentCases\\"

with open(baseDir+'Natural06.xml') as sFile:
    text3 = sFile.read()
print(text3)
# Process whole documents
text = ("It's a two-way, three-lane street with two southbound lanes and one northbound lane. The street has a speed limit of 45mph. There's a junction nearby with a eastbound road. The traffic signal is red. "
        "The ego-vehicle is stopped near the junction on the North branch of the road. "
        "A 2013 black toyota camry has stopped near the signal while going South. "
        "A white 2012 honda accord is going east through the junction at a speed of 30mph. "
        "Another red 2015 ford focus is turning left from the eastbound road onto the southbound road, through the junction, at a speed of 35mph.")
text2 = ("It is a two-way, double-lane southbound residential street with pavement on the right side. The speed limit is 40mph. The weather is clear and dry."
         "One white hatchback and one black sedan is parked on the right side."
         " A white coupe is going North at a speed of 30mph and about to cross the ego-vehicle."
         " One black SUV is going South at a speed of 25mph."
         " One red sedan is going South at a speed of 35mph. It is leading the black SUV by 200ft. "
         "The distance between the black SUV and ego-vehicle is 300ft.")
doc = nlp(text3)

# Analyze syntax
# print("Noun phrases:", [chunk.text for chunk in doc.noun_chunks])
# print("Verbs:", [token.lemma_ for token in doc if token.pos_ == "VERB"])
#
# for token in doc:
#         print(token,token.ent_type_,token.lemma_)

# Find named entities, phrases and concepts
for entity in doc.ents:
    print(entity.text, entity.label_)
print(doc._.has_coref)
print(doc._.coref_clusters)
# print(nlp2(text))
# txt = Text(text)
# for sent in txt.sentences:
#   print(sent, "\n")
#   for entity in sent.entities:
#     print(entity.tag, entity)

# {print(' '.join(c[0] for c in chunk), chunk.label() ) for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(text2))) if hasattr(chunk, 'label') }
# tmp = al.predict(sentence=text)
# for i in range(len(tmp['words'])):
#     if(tmp['tags'][i] != 0):
#         print(tmp['words'][i],tmp['tags'][i])
[to_nltk_tree(sent.root).pretty_print() for sent in doc.sents]