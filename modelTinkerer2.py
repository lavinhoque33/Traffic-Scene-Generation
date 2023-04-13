import sys
import spacy
from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER
from spacy.lang.char_classes import CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS
from spacy.tokenizer import Tokenizer
from spacy.util import compile_infix_regex
from word2number import w2n
from urllib.parse import unquote
from nltk import Tree
from pathlib import Path
from owlready2 import *
from tkinter.filedialog import askopenfilename
from transformers import AutoTokenizer
import copy

sys.path.append('fast-coref/src')
from inference.model_inference import Inference

onto = get_ontology("Ontology/CarOntology.rdf").load()
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

vs = []
vTypes = []
vActions = []
vActL = []
vPhrases = []
vDir = []

for clas in onto.classes():
    if (clas.name == "car_type-leaflvl"):
        for instance in clas.instances():
            vTypes.append(instance.name)
    if (clas.name == "car_makemodel-leaflvl"):
        for instance in clas.instances():
            vs.append(unquote(instance.name))
    if (clas.name == "vehicle_direction-leaflvl"):
        for instance in clas.instances():
            vDir.append(unquote(instance.name))
    if (clas.name == "vehicle_action-leaflvl"):
        for instance in clas.instances():
            vActL.append(unquote(instance.name))
            vActions.append(
                {'action': unquote(instance.name), 'properties': [prop.name for prop in instance.get_properties()]})


def to_nltk_tree(node):
    if node.n_lefts + node.n_rights > 0:
        return Tree(node.orth_, [to_nltk_tree(child) for child in node.children])
    else:
        return node.orth_


def checkVerb(tree):
    if (tree.pos_ == 'VERB'):
        return tree
    for child in tree.children:
        tmp = checkVerb(child)
        if (tmp is not None):
            return tmp


def getVerbs(sent):
    ret = []
    for token in sent:
        if (token.pos_ == 'VERB'):
            ret.append(token)
    return ret


def checkAction2(sent, vRoot, vPhrase):
    vbs = getVerbs(sent)
    for token in sent:
        if (token.text == vRoot and vPhrase in " ".join(tok.text for tok in list(token.subtree)).split(' and ')):
            for vehicle in vehicles:
                if (token.head.text == vehicle['root']):
                    token = token.head
                    break
            if (token in token.head.lefts):
                # print('HERE1', token.text, token.head.lemma_)
                if (token.head.lemma_ in vActL):
                    if (token.head.pos_ == 'VERB'):
                        return token.head
                else:
                    # print('HERE', token.text, token.head, list(token.head.rights))
                    for tok in token.head.rights:
                        tmp = checkVerb(tok)
                        if (tmp is not None):
                            return tmp
    return False


def getHead(token):
    for vehicle in vehicles:
        if (token.head.text == vehicle['root']):
            token = token.head


def checkAction(sent, vRoot, vPhrase):
    vbs = getVerbs(sent)
    for token in sent:
        # print(" ".join(tok.text for tok in list(token.subtree)).split(' and '), "HE")
        t1 = []
        t2 = " ".join(tok.text for tok in list(token.subtree)).split(' and ')
        for i in range(len(t2)):
            for tm in t2[i].split(' , '):
                t1.append(tm)
        if (token.text == vRoot and vPhrase in t1):
            for vehicle in vehicles:
                if (token.head.text == vehicle['root']):
                    token = token.head
                    break
            if (token in token.head.lefts):
                # print('HERE1', token.text, token.head.lemma_)
                if (token.head.lemma_ in vActL):
                    if (token.head.pos_ == 'VERB'):
                        return token.head
                else:
                    # print('HERE', token.text, token.head, list(token.head.rights))
                    for tok in token.head.rights:
                        tmp = checkVerb(tok)
                        if (tmp is not None):
                            return tmp
    return False


def checkActor(actor, ltree):
    for root in ltree:
        # print(" ".join(tok.text for tok in list(root.subtree)))
        if (" ".join(tok.text for tok in list(root.subtree)) == actor):
            return True
    return False


def checkActed(rtree):
    for root in rtree:
        if (root.dep_ == 'prep'):
            return checkActed(root.children)
        ret = " ".join(tok.text for tok in list(root.subtree)).strip()
        if (ret in vPhrases):
            for vehicle in vehicles:
                if vehicle['phrase'] == ret:
                    return vehicle


def checkActed2(rtree):
    for root in rtree:
        if (root.pos_ == 'NOUN'):
            ret = " ".join(tok.text for tok in list(root.subtree)).strip()
            if (ret in vPhrases):
                for vehicle in vehicles:
                    if vehicle['phrase'] == ret:
                        return vehicle


def checkRelPos(posActions, prevSent):
    for action in posActions:
        if (action.lemma_ == 'follow'):
            tmp = checkActed2(action.rights)
            if (tmp is not None):
                return tmp
            else:
                return sentVehicle[prevSent][0]
        else:
            for token in action.rights:
                if (token.lemma_ in ('ahead', 'behind')):
                    tmp = checkActed(token.children)
                    if (tmp is not None):
                        return tmp
                    else:
                        return sentVehicle[prevSent][0]
    return False


def checkLaTemp(root):
    if (root.lemma_ in ['lane', 'side']):
        for child in root.children:
            if (child.text.lower() in ('left', 'middle', 'mid', 'right')):
                return child.text.lower()
    else:
        for child in root.children:
            tmp = checkLaTemp(child)
            if (tmp):
                return tmp


def checkLane(posActions, prevSent):
    for action in posActions:
        # print(list(action.rights))
        for token in action.rights:
            if (token.lemma_ in ['lane', 'side']):
                for child in token.children:
                    if (child.text.lower() in ('left', 'middle', 'mid', 'right')):
                        return child.text.lower()
            else:
                for child in token.children:
                    tmp = checkLaTemp(child)
                    if (tmp is not None):
                        return tmp
    return False


def checkVehicle(token):
    for car in vs:
        if (token.lower() == ' '.join(car.split()[1:]).lower()):
            return car
        elif (token.lower() == car.split()[0].lower()):
            return car.split()[0]
    return False


def getSpeed(tree):
    for root in tree:
        if (root.text in ('mph', 'kmph', 'kmh', 'miles/hour', 'm/h', 'kilometres/hour', 'km/h')):
            return root.nbor(-1).text + root.text
        tmp = getSpeed(root.children)
        if (tmp is not None):
            return tmp


def getDirection(tree):
    for root in tree:
        # print(" ".join(tok.lemma_ for tok in list(root.subtree)))
        tmp = " ".join(tok.text.lower() for tok in list(root.subtree))
        for direction in vDir:
            if (direction in tmp):
                print('Found direction ', direction)
                return direction
    print('No Direction')
    return False


def checkVehicleType(token):
    for car in vTypes:
        if (token.lower() == car.lower()):
            return car
    return False


def getChild(token, veh):
    for child in token.children:
        # print(child.text, child.dep_)
        if (checkVehicleType(child.text)):
            continue
        if (child.dep_ == 'amod' or child.dep_ == 'compound'):
            if (child.text.lower() not in veh.lower()):
                coly[0] = child.text
        if (child.dep_ == 'nummod'):
            try:
                if (int(child.text) > 1000):
                    coly[1] = child.text
            except:
                print('Not a year')
                coly[2] = child.text
        getChild(child, veh)


def getActors():
    vehicles = []
    # print("Noun phrases:", "chunk text", "chunk root text", "chunk root dep_", "chunk root head text")
    # print('HERE',list(doc.noun_chunks))
    for chunk in doc.noun_chunks:
        try:
            if (nPhrases[chunk.text] == 1):
                continue
        except KeyError:
            # print(KeyError)
            nPhrases[chunk.text] = 1
        # print("Noun phrases:", chunk.text, chunk.root.text, chunk.root.dep_)
        # if('ego' in chunk.text):
        #     vPhrases.append(chunk.text)
        #     vehicles.append({'makemodel': 'Ego-vehicle', 'root': 'ego-vehicle','phrase': chunk.text, 'actions': []})
        #     continue
        tmp = checkVehicleType(chunk.root.lemma_)
        if (not tmp):
            tmp = checkVehicle(chunk.root.lemma_)
        if (tmp):
            vcl = {}
            vcl['makemodel'] = tmp
            vcl['root'] = chunk.root.text
            for i in range(len(coly)):
                coly[i] = False
            getChild(chunk.root, tmp)
            if (coly[0]):
                vcl['color'] = coly[0]
            if (coly[1]):
                vcl['year'] = coly[1]
            vcl['phrase'] = chunk.text
            vPhrases.append(chunk.text)
            vcl['actions'] = []
            cnt = 1
            if (coly[2]):
                cnt = w2n.word_to_num(coly[2])
            for i in range(cnt):
                vehicles.append(copy.deepcopy(vcl))
    return vehicles


nlp = spacy.load("en_core_web_sm")
infixes = (
        LIST_ELLIPSES
        + LIST_ICONS
        + [
            r"(?<=[0-9])[+\-\*^](?=[0-9-])",
            r"(?<=[{al}{q}])\.(?=[{au}{q}])".format(
                al=ALPHA_LOWER, au=ALPHA_UPPER, q=CONCAT_QUOTES
            ),
            r"(?<=[{a}]),(?=[{a}])".format(a=ALPHA),
            # ✅ Commented out regex that splits on hyphens between letters:
            # r"(?<=[{a}])(?:{h})(?=[{a}])".format(a=ALPHA, h=HYPHENS),
            r"(?<=[{a}0-9])[:<>=/](?=[{a}])".format(a=ALPHA),
        ]
)

infix_re = compile_infix_regex(infixes)
output_dir = Path('./content/')
# Add special case
# nlp.tokenizer.add_special_case("ego-vehicle", [{"ORTH": "ego-vehicle"}])
nlp.tokenizer.infix_finditer = infix_re.finditer
# exceptions = nlp.Defaults.tokenizer_exceptions
# filtered_exceptions = {k:v for k,v in exceptions.items() if "'" not in k and "’" not in k and "‘" not in k}
# nlp.tokenizer = Tokenizer(nlp.vocab, rules = filtered_exceptions)
# Download the rest of the model parameters
# !gdown --id 1CQxUq2zvCHc1mJUEZ_Zy6WSJQqFz76Pw
# Initialize the inference module. This will also download the Longformer model finetuned for OntoNotes from Huggingface
inference_model = Inference("./", encoder_name="shtoshni/longformer_coreference_ontonotes", spac=nlp)

# For joint model
# !gdown --id 1c_X-iDJNr4BM9iAN4YjUMVJUS741eA_Z
# inference_model = Inference("./", encoder_name="shtoshni/longformer_coreference_joint")

baseDir = "D:\\Downloads\\Documents\\CS699\\SpaceTraffic2\\Annotated Data\\"
filename = askopenfilename()
print(filename)
# with open(baseDir+'Natural08.xml') as sFile:
with open(filename) as sFile:
    text = sFile.read()
print(text)
# Process whole documents

from spacy.matcher import Matcher,PhraseMatcher

# matcher = Matcher(nlp.vocab)
matcher = PhraseMatcher(nlp.vocab)

def replace_word(docs, wrd, replacement):
    matcher.add(wrd, [nlp(wrd)])
    doc = nlp(docs)
    match_id, start, end = matcher(doc)[0]  # assuming only one match replacement

    return doc[:start].text_with_ws + f"{replacement}" +doc[end-1].whitespace_ + doc[end:-1].text_with_ws


output = inference_model.perform_coreference(text)

for cluster in output["clusters"]:
    if len(cluster) > 1:
        print(cluster)
        for i in range(1, len(cluster)):
            text = replace_word(text, cluster[i][1], cluster[0][1])
            # text = text.replace(cluster[i][1], cluster[0][1])

print("Coreference Resolved DOC: ")
print(text)
# [to_nltk_tree(sent.root).pretty_print() for sent in doc.sents]

doc = nlp(text)
print(doc[0],doc[1])
coly = [False] * 3
nPhrases = {}
# Analyze syntax


vehicles = getActors()
# print(vehicles)
sentVehicle = {}
# Analyze actions
prevSent = ''
for vehicle in vehicles:
    print(vehicle)
    for sent in doc.sents:
        if (sentVehicle.get(sent) is None):
            sentVehicle[sent] = []
        if (vehicle['phrase'] in sent.text):
            action = {}
            # print(':::::', sent.text.strip())
            # print(':::::::', sent.root.lemma_)
            # for token in sent:
            #     print(':::::::::', token.text, token.head.text, token.head.pos_, token.dep_, token.pos_)
            sentence_tokens = []
            sentence_tokens.append([token.text for token in sent])
            act = checkAction(sent, vehicle['root'], vehicle['phrase'])
            print('Action is: ', act)
            if (not act):
                continue
            action['action'] = act.lemma_
            if (act.lemma_ in ('go', 'drive', 'travel', 'enter', 'exit', 'turn', 'pass', 'follow')):
                # Get Speed of Action
                tmp = getSpeed(act.rights)
                if (tmp is not None):
                    action['speed'] = tmp
            if (act.lemma_ == 'pass'):
                action['actedon'] = checkActed(act.rights)['makemodel']
            # Get Direction of Action
            tmp = getDirection(act.rights)
            if (tmp):
                action['direction'] = tmp
            tmp = checkRelPos([act, sent.root], prevSent)
            if (tmp):
                print('Relative Position Found')
                action['relatedto'] = tmp['makemodel']
                if (tmp['actions'][0].get('direction') is not None):
                    action['direction'] = tmp['actions'][0]['direction']
            cnt = False
            if (vehicle in sentVehicle[sent]):
                cnt = True
            tmp = checkLane([act, sent.root], prevSent)
            if (tmp):
                action['lane'] = tmp
            vehicle['actions'].append(action)
            if (not cnt):
                sentVehicle[sent].append(vehicle)
        prevSent = sent

for vehicle in vehicles:
    print(vehicle)
