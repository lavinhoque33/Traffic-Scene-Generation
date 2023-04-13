import json
import sys
import spacy
from transformers import T5Tokenizer, T5ForConditionalGeneration
from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER
from spacy.lang.char_classes import CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS
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
    if(clas.name == "car_type-leaflvl"):
        for instance in clas.instances():
            vTypes.append(instance.name)
    if(clas.name == "car_makemodel-leaflvl"):
        for instance in clas.instances():
            vs.append(unquote(instance.name))
    if (clas.name == "vehicle_direction-leaflvl"):
        for instance in clas.instances():
            vDir.append(unquote(instance.name))
    if (clas.name == "vehicle_action-leaflvl"):
        for instance in clas.instances():
            vActL.append(unquote(instance.name))
            vActions.append({'action': unquote(instance.name), 'properties': [prop.name for prop in instance.get_properties()]})

def to_nltk_tree(node):
    if node.n_lefts + node.n_rights > 0:
        return Tree(node.orth_, [to_nltk_tree(child) for child in node.children])
    else:
        return node.orth_

def checkVerb(tree):
    if(tree.pos_ == 'VERB'):
        return tree
    for child in tree.children:
        tmp = checkVerb(child)
        if(tmp is not None):
            return tmp

def getVerbs(sent):
    ret = []
    for token in sent:
        if(token.pos_ == 'VERB'):
            ret.append(token)
    return ret

def checkAction2(sent, vRoot, vPhrase):
    vbs = getVerbs(sent)
    for token in sent:
        if(token.text == vRoot and vPhrase in " ".join(tok.text for tok in list(token.subtree)).split(' and ')):
            for vehicle in vehicles:
                if(token.head.text == vehicle['root']):
                    token = token.head
                    break
            if(token in token.head.lefts):
                # print('HERE1', token.text, token.head.lemma_)
                if(token.head.lemma_ in vActL):
                    if(token.head.pos_ == 'VERB'):
                        return token.head
                else:
                    # print('HERE', token.text, token.head, list(token.head.rights))
                    for tok in token.head.rights:
                        tmp = checkVerb(tok)
                        if(tmp is not None):
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
        if(token.text == vRoot and vPhrase in t1):
            for vehicle in vehicles:
                if(token.head.text == vehicle['root']):
                    token = token.head
                    break
            if(token in token.head.lefts):
                # print('HERE1', token.text, token.head.lemma_)
                if(token.head.lemma_ in vActL):
                    if(token.head.pos_ == 'VERB'):
                        return token.head
                else:
                    # print('HERE', token.text, token.head, list(token.head.rights))
                    for tok in token.head.rights:
                        tmp = checkVerb(tok)
                        if(tmp is not None):
                            return tmp
    return False

def checkActor(actor, ltree):
    for root in ltree:
        # print(" ".join(tok.text for tok in list(root.subtree)))
        if(" ".join(tok.text for tok in list(root.subtree)) == actor):
            return True
    return False

def checkActed(rtree):
    for root in rtree:
        if(root.dep_ == 'prep'):
            return checkActed(root.children)
        ret = " ".join(tok.text for tok in list(root.subtree)).strip()
        if (ret in vPhrases):
            for vehicle in vehicles:
                if vehicle['phrase'] == ret:
                    return vehicle

def checkActed2(rtree):
    for root in rtree:
        if(root.pos_ == 'NOUN'):
            ret = " ".join(tok.text for tok in list(root.subtree)).strip()
            if (ret in vPhrases):
                for vehicle in vehicles:
                    if vehicle['phrase'] == ret:
                        return vehicle

def checkRelPos(posActions, prevSent):
    for action in posActions:
        if(action.lemma_ == 'follow'):
            tmp = checkActed2(action.rights)
            if (tmp is not None):
                return tmp
            else:
                return sentVehicle[prevSent][0],'follow'
        else:
            for token in action.rights:
                if(token.lemma_ in ('ahead', 'behind')):
                    tmp = checkActed(token.children)
                    if(tmp is not None):
                        return tmp
                    else:
                        return sentVehicle[prevSent][0],token.lemma_
    return False

def checkLaTemp(root):
    if(root.lemma_ in ['lane','side']):
        for child in root.children:
            if (child.text.lower() in ('left', 'middle', 'mid', 'right')):
                return child.text.lower()
    else:
        for child in root.children:
            tmp = checkLaTemp(child)
            if(tmp):
                return tmp

def checkLane(posActions, prevSent):
    for action in posActions:
        # print(list(action.rights))
        for token in action.rights:
            if(token.lemma_ in ['lane','side']):
                for child in token.children:
                    if(child.text.lower() in ('left', 'middle', 'mid', 'right')):
                        return child.text.lower()
            else:
                for child in token.children:
                    tmp = checkLaTemp(child)
                    if(tmp is not None):
                        return tmp
    return False

def checkVehicle(token):
    for car in vs:
        if(token.lower() == ' '.join(car.split()[1:]).lower()):
            return car
        elif(token.lower() == car.split()[0].lower()):
            return car.split()[0]
    return False

def getSpeed(tree):
    for root in tree:
        if(root.text in ('mph', 'kmph', 'kmh', 'miles/hour', 'm/h', 'kilometres/hour', 'km/h')):
            return root.nbor(-1).text+root.text
        tmp = getSpeed(root.children)
        if(tmp is not None):
            return tmp

def getDirection(tree):
    for root in tree:
        # print(" ".join(tok.lemma_ for tok in list(root.subtree)))
        tmp = " ".join(tok.text.lower() for tok in list(root.subtree))
        for direction in vDir:
            if(direction in tmp):
                # print('Found direction ', direction)
                return direction
    # print('No Direction')
    return False

def checkVehicleType(token):
    for car in vTypes:
        if(token.lower() == car.lower()):
            return car
    return False

def getChild(token,veh):
    for child in token.children:
        # print(child.text, child.dep_)
        if (checkVehicleType(child.text)):
            continue
        if(child.dep_ == 'amod' or child.dep_ == 'compound'):
            if(child.text.lower() not in veh.lower()):
                coly[0] = child.text
        if (child.dep_ == 'nummod'):
            try:
                if(int(child.text)>1000):
                    coly[1] = child.text
            except:
                # print('Not a year')
                coly[2] = child.text
        getChild(child,veh)

def getActors():
    vehicles = []
    # print("Noun phrases:", "chunk text", "chunk root text", "chunk root dep_", "chunk root head text")
    # print('HERE',list(doc.noun_chunks))
    for chunk in doc.noun_chunks:
        try:
            if(nPhrases[chunk.text] == 1):
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
        if(not tmp):
            tmp = checkVehicle(chunk.root.lemma_)
        if(tmp):
            vcl = {}
            vcl['makemodel'] = tmp
            vcl['root'] = chunk.root.text
            for i in range(len(coly)):
                coly[i] = False
            getChild(chunk.root,tmp)
            if(coly[0]):
                vcl['color'] = coly[0]
            if(coly[1]):
                vcl['year'] = coly[1]
            vcl['phrase'] = chunk.text
            vPhrases.append(chunk.text)
            vcl['actions'] = []
            cnt = 1
            if(coly[2]):
                cnt = w2n.word_to_num(coly[2])
            for i in range(cnt):
                vehicles.append(copy.deepcopy(vcl))
    return vehicles

# model_name = "allenai/unifiedqa-v2-t5-3b-1363200"
model_name = "allenai/unifiedqa-t5-3b" # you can specify the model size here
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def run_model(input_string, **generator_args):
    input_ids = tokenizer.encode(input_string, return_tensors="pt")
    res = model.generate(input_ids, **generator_args)
    return input_ids,tokenizer.batch_decode(res, skip_special_tokens=True)

def run_model2(q, opts=None):
    if (opts):
        ids,ans = run_model(q + ' \\n ' + opts + ' \\n ' + text2)
    else:
        ids,ans = run_model(q + ' \\n ' + text2)
    return ids,ans

def compute_loss(input_ids, answer):
    labels = tokenizer(answer, return_tensors='pt').input_ids
    res = model.forward(input_ids, labels=labels)
    return res.loss.tolist()


def formOpts(opts):
    op2 = ''
    op3 = []
    c = 0
    for i in opts.split(','):
        op2 += '(' + chr(ord('a') + c) + ')' + i.strip() + ' '
        c += 1
        op3.append(i.strip())
    return op2, op3

def getQA(q,ops=None):
    if(ops):
        opts, optlist = formOpts(ops)
        ids, ans = run_model2(q, opts)
        los = compute_loss(ids, ans)
    else:
        ids, ans = run_model2(q)
        los = compute_loss(ids, ans)
    return q,ans,los

nlp = spacy.load("en_core_web_trf")
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
# Download the rest of the model parameters
# !gdown --id 1CQxUq2zvCHc1mJUEZ_Zy6WSJQqFz76Pw
# Initialize the inference module. This will also download the Longformer model finetuned for OntoNotes from Huggingface
inference_model = Inference("./", encoder_name="shtoshni/longformer_coreference_ontonotes", spac = nlp)

from spacy.matcher import Matcher,PhraseMatcher
# matcher = Matcher(nlp.vocab)
matcher = PhraseMatcher(nlp.vocab)

def replace_word(docs, wrd, replacement):
    matcher.add(wrd, [nlp(wrd)])
    doc = nlp(docs)
    ind = 0
    match_id_string = ''
    t = matcher(doc)
    while(match_id_string != wrd):
        match_id, start, end = t[ind]  # assuming only one match replacement
        match_id_string = nlp.vocab.strings[match_id]
        ind += 1
    return doc[:start].text_with_ws + f"{replacement}" +doc[end-1].whitespace_ + doc[end:].text_with_ws
# For joint model
# !gdown --id 1c_X-iDJNr4BM9iAN4YjUMVJUS741eA_Z
# inference_model = Inference("./", encoder_name="shtoshni/longformer_coreference_joint")
otp = ''
baseDir = "D:\\Downloads\\Documents\\CS699\\SpacyTraffic3.9\\Annotated Data\\"
# filename = askopenfilename()
# print(filename)
# with open(baseDir+'Natural08.xml') as sFile:
for sam in range(15,16):
    print('Case: ', sam)
    otp+= f"Case:{sam}" + '\n'
    try:
        with open(baseDir+str(sam)+'\\Case'+str(sam)+'.xml') as sFile:
            text2 = sFile.read()
    except:
        print("No Files")
        continue
    # print(text)
    # Process whole documents
    text = text2
    output = inference_model.perform_coreference(text2)



    for cluster in output["clusters"]:
      if len(cluster) > 1:
        print(cluster)
        if(len(cluster[0][1].split()) <= 1):
            print("Skipping")
            continue
        for i in range(1, len(cluster)):
            text = replace_word(text, cluster[i][1], cluster[0][1])
          # text = text2.replace(cluster[i][1], cluster[0][1])

    # print(text)

    print("Coreference Resolved DOC: ")
    print(text)
    otp+= '\n'+ text + '\n'
    doc = nlp(text)

    # [to_nltk_tree(sent.root).pretty_print() for sent in doc.sents]

    coly = [False]*3
    nPhrases = {}
    # Analyze syntax


    vehicles = getActors()
    # print(vehicles)
    sentVehicle = {}
    roads = {}
    #Analyze actions
    prevSent = ''

    # Road Property Questions
    q, ans, los = getQA('is the road a junction?', 'yes,no')
    if(ans[0] == 'yes' and los < 0.05):
        roads['type'] = 'junction'
        roads['junc'] = {}
        q, ans, los = getQA('is the junction a t-junction?', 'yes,no')
        if (ans[0] == 'yes' and los < 0.05):
            roads['junc']['type'] = 'T'
        else:
            q, ans, los = getQA('is the junction 4-way?', 'yes,no')
            if (ans[0] == 'yes' and los < 0.05):
                roads['junc']['type'] = '4-way'
            else:
                q, ans, los = getQA('is the junction 3-way?', 'yes,no')
                if (ans[0] == 'yes' and los < 0.05):
                    roads['junc']['type'] = '3-way'
        q, ans, los = getQA('is there a south branch?', 'yes,no')
        if (ans[0] == 'yes' and los < 0.05):
            roads['south'] = {}
            roads['south']['way'] = 2
            roads['south']['lane'] = 1
            roads['south']['divided'] = False
        q, ans, los = getQA('is there a north branch?', 'yes,no')
        if (ans[0] == 'yes' and los < 0.05):
            roads['north'] = {}
            roads['north']['way'] = 2
            roads['north']['lane'] = 1
            roads['north']['divided'] = False
        q, ans, los = getQA('is there a west branch?', 'yes,no')
        if (ans[0] == 'yes' and los < 0.05):
            roads['west'] = {}
            roads['west']['way'] = 2
            roads['west']['lane'] = 1
            roads['west']['divided'] = False
        q, ans, los = getQA('is there a east branch?', 'yes,no')
        if (ans[0] == 'yes' and los < 0.05):
            roads['east'] = {}
            roads['east']['way'] = 2
            roads['east']['lane'] = 1
            roads['east']['divided'] = False
    else:
        roads['type'] = 'regular'
        roads['main'] = {}
        q, ans, los = getQA('is the road one-way?', 'yes,no')
        if (ans[0] == 'yes' and los < 0.15):
            roads['main']['way'] = 1
            roads['main']['divided'] = False
        else:
            q, ans, los = getQA('is the road two-way?', 'yes,no')
            if (ans[0] == 'yes' and los < 0.15):
                roads['main']['way'] = 2
            q, ans, los = getQA('is the road divided?', 'yes,no')
            if (ans[0] == 'yes' and los < 0.1):
                roads['main']['divided'] = True
            else:
                roads['main']['divided'] = False
        q, ans, los = getQA('is it a one lane road?', 'yes,no')
        if (ans[0] == 'yes' and los < 0.15):
            roads['main']['lane'] = 1
        else:
            q, ans, los = getQA('is it a two lane road?', 'yes,no')
            if (ans[0] == 'yes' and los < 0.15):
                roads['main']['lane'] = 2
            else:
                q, ans, los = getQA('is it a three lane road?', 'yes,no')
                if (ans[0] == 'yes' and los < 0.15):
                    roads['main']['lane'] = 3
                else:
                    q, ans, los = getQA('is it a single lane road?', 'yes,no')
                    if (ans[0] == 'yes' and los < 0.15):
                        roads['main']['lane'] = 1

    for i in range(len(vehicles)):
        vehicle = vehicles[i]
        # print(vehicle)
        #QA approach start
        action = {}
        q, ans, los = getQA('is ' + vehicle['phrase'] + ' turning?', 'yes,no')
        if (ans[0] == 'yes' and los < 0.01):
            action['action'] = 'turn'
            q, ans, los = getQA('what is the speed of ' + vehicle['phrase'] + ' ?')
            if (los < 0.1):
                action['speed'] = ans[0]
            q, ans, los = getQA('is ' + vehicle['phrase'] + ' turning left?', 'yes,no')
            if (ans[0] == 'yes' and los < 0.05):
                action['reldir'] = 'left'
            else:
                q, ans, los = getQA('is ' + vehicle['phrase'] + ' turning right?', 'yes,no')
                if (ans[0] == 'yes' and los < 0.05):
                    action['reldir'] = 'right'
            if(roads['type'] == 'junc'):
                q, ans, los = getQA('is ' + vehicle['phrase'] + ' turning onto south branch?', 'yes,no')
                if (ans[0] == 'yes' and los < 0.05):
                    action['turndir'] = 'south'
                else:
                    q, ans, los = getQA('is ' + vehicle['phrase'] + ' turning onto north branch?', 'yes,no')
                    if (ans[0] == 'yes' and los < 0.05):
                        action['turndir'] = 'north'
                    else:
                        q, ans, los = getQA('is ' + vehicle['phrase'] + ' turning onto east branch?', 'yes,no')
                        if (ans[0] == 'yes' and los < 0.05):
                            action['turndir'] = 'east'
                        else:
                            q, ans, los = getQA('is ' + vehicle['phrase'] + ' turning onto west branch?', 'yes,no')
                            if (ans[0] == 'yes' and los < 0.05):
                                action['turndir'] = 'west'
        else:
            q, ans, los = getQA('is ' + vehicle['phrase'] + ' travelling?', 'yes,no')
            if (ans[0] == 'yes' and los < 0.1):
                action['action'] = 'travel'
                q, ans, los = getQA('what is the speed of ' + vehicle['phrase'] + ' ?')
                if (los < 0.1):
                    action['speed'] = ans[0]
                q, ans, los = getQA('is ' + vehicle['phrase'] + ' going south?', 'yes,no')
                if (ans[0] == 'yes' and los < 0.05):
                    action['direction'] = 'south'
                else:
                    q, ans, los = getQA('is ' + vehicle['phrase'] + ' going north?', 'yes,no')
                    if (ans[0] == 'yes' and los < 0.05):
                        action['direction'] = 'north'
                    else:
                        q, ans, los = getQA('is ' + vehicle['phrase'] + ' going east?', 'yes,no')
                        if (ans[0] == 'yes' and los < 0.05):
                            action['direction'] = 'east'
                        else:
                            q, ans, los = getQA('is ' + vehicle['phrase'] + ' going west?', 'yes,no')
                            if (ans[0] == 'yes' and los < 0.05):
                                action['direction'] = 'west'
            else:
                q, ans, los = getQA('is ' + vehicle['phrase'] + ' parked?', 'yes,no')
                chk = False
                if (ans[0] == 'yes' and los < 0.01):
                    chk = True
                if (chk):
                    action['action'] = 'park'
                    q, ans, los = getQA('is ' + vehicle['phrase'] + ' parked on the road?', 'yes,no')
                    if (ans[0] == 'yes' and los < 0.05):
                        action['parkloc'] = 'on-road'
                    else:
                        q, ans, los = getQA('is ' + vehicle['phrase'] + ' parked off-street?', 'yes,no')
                        if (ans[0] == 'yes' and los < 0.05):
                            action['parkloc'] = 'off-road'
                        else:
                            chk = False
                            action['action'] = ''
                    if(chk):
                        q, ans, los = getQA('is ' + vehicle['phrase'] + ' parked on the left side?', 'yes,no')
                        if (ans[0] == 'yes' and los < 0.05):
                            action['parkdir'] = 'left'
                        else:
                            q, ans, los = getQA('is ' + vehicle['phrase'] + ' parked on the right side?', 'yes,no')
                            if (ans[0] == 'yes' and los < 0.05):
                                action['parkdir'] = 'right'
                if(not chk):
                    q, ans, los = getQA('is ' + vehicle['phrase'] + ' stopped?', 'yes,no')
                    if (ans[0] == 'yes' and los < 0.05):
                        action['action'] = 'stop'
                        chk2 = True
                        if (roads['type'] == 'junc'):
                            q, ans, los = getQA('is ' + vehicle['phrase'] + ' on the south branch?', 'yes,no')
                            if (ans[0] == 'yes' and los < 0.05):
                                action['branch'] = 'south'
                                action['direction'] = 'north'
                                chk2 = False
                        else:
                            q, ans, los = getQA('is south the direction of ' + vehicle['phrase'] + '?', 'yes,no')
                            if (ans[0] == 'yes' and los < 0.05):
                                action['direction'] = 'south'
                                chk2 = False
                        if(chk2):
                            if (roads['type'] == 'junc'):
                                q, ans, los = getQA('is ' + vehicle['phrase'] + ' on the north branch?', 'yes,no')
                                if (ans[0] == 'yes' and los < 0.05):
                                    action['branch'] = 'north'
                                    action['direction'] = 'south'
                                    chk2 = False
                            else:
                                q, ans, los = getQA('is north the direction of ' + vehicle['phrase'] + '?', 'yes,no')
                                if (ans[0] == 'yes' and los < 0.05):
                                    action['direction'] = 'north'
                                    chk2 = False
                        if (chk2):
                            if (roads['type'] == 'junc'):
                                q, ans, los = getQA('is ' + vehicle['phrase'] + ' on the east branch?', 'yes,no')
                                if (ans[0] == 'yes' and los < 0.05):
                                    action['branch'] = 'east'
                                    action['direction'] = 'west'
                                    chk2 = False
                            else:
                                q, ans, los = getQA('is east the direction of ' + vehicle['phrase'] + '?', 'yes,no')
                                if (ans[0] == 'yes' and los < 0.05):
                                    action['direction'] = 'east'
                                    chk2 = False
                        if (chk2):
                            if (roads['type'] == 'junc'):
                                q, ans, los = getQA('is ' + vehicle['phrase'] + ' on the west branch?', 'yes,no')
                                if (ans[0] == 'yes' and los < 0.05):
                                    action['branch'] = 'west'
                                    action['direction'] = 'east'
                                    chk2 = False
                            else:
                                q, ans, los = getQA('is west the direction of ' + vehicle['phrase'] + '?', 'yes,no')
                                if (ans[0] == 'yes' and los < 0.05):
                                    action['direction'] = 'west'
                                    chk2 = False

        q, ans, los = getQA('is ' + vehicle['phrase'] + ' on the left lane?', 'yes,no')
        if (ans[0] == 'yes' and los < 0.005):
            action['lane'] = 'left'
        q, ans, los = getQA('is ' + vehicle['phrase'] + ' on the right lane?', 'yes,no')
        if (ans[0] == 'yes' and los < 0.005):
            action['lane'] = 'right'
        q, ans, los = getQA('is ' + vehicle['phrase'] + ' on the middle lane?', 'yes,no')
        if (ans[0] == 'yes' and los < 0.005):
            action['lane'] = 'middle'
        vehicle['actions'].append(action)



        #Rule Based Approach Start
        for sent in doc.sents:
            if(sentVehicle.get(sent) is None):
                sentVehicle[sent] = []
            if(vehicle['phrase'] in sent.text):
                action = {}
                # print(':::::', sent.text.strip())
                # print(':::::::', sent.root.lemma_)
                # for token in sent:
                #     print(':::::::::', token.text, token.head.text, token.head.pos_, token.dep_, token.pos_)
                sentence_tokens = []
                sentence_tokens.append([token.text for token in sent])
                act = checkAction(sent, vehicle['root'], vehicle['phrase'])
                # print('Action is: ', act)
                if(not act):
                    continue
                action['action'] = act.lemma_
                if(act.lemma_ in ('go','drive','travel','enter','exit','turn','pass','follow')):
                    #Get Speed of Action
                    tmp = getSpeed(act.rights)
                    if(tmp is not None):
                        action['speed'] = tmp
                        try:
                            if(tmp != vehicle['actions'][0]['speed']):
                                vehicle['actions'][0]['speed2'] = tmp
                        except:
                            vehicle['actions'][0]['speed'] = tmp

                if(act.lemma_ == 'pass'):
                    try:
                        action['actedon'] = checkActed(act.rights)['makemodel']
                        action['relaction'] = 'pass'
                        vehicle['actions'][0]['actedon'] = action['actedon']
                        vehicle['actions'][0]['relaction'] = 'pass'
                    except:
                        print('No Relative Actor')
                elif(act.lemma_ == 'follow'):
                    try:
                        action['actedon'] = checkActed(act.rights)['makemodel']
                        action['relaction'] = 'follow'
                        vehicle['actions'][0]['actedon'] = action['actedon']
                        vehicle['actions'][0]['relaction'] = 'follow'
                    except:
                        print('No Relative Actor')
                    # Get Direction of Action
                tmp = getDirection(act.rights)
                if(tmp):
                    action['direction'] = tmp
                    try:
                        if (not vehicle['actions'][0]['direction']):
                            vehicle['actions'][0]['direction'] = action['direction']
                    except:
                        vehicle['actions'][0]['direction'] = action['direction']
                try:
                    tmp,rela = checkRelPos([act, sent.root], prevSent)
                except:
                    tmp = None
                if(tmp):
                    # print('Relative Position Found')
                    action['relatedto'] = tmp['makemodel']
                    vehicle['actions'][0]['actedon'] = tmp['makemodel']
                    vehicle['actions'][0]['relaction'] = rela
                    try:
                        if(tmp['actions'][0].get('direction') is not None):
                            action['direction'] = tmp['actions'][0]['direction']
                            try:
                                if(not vehicle['actions'][0]['direction']):
                                    vehicle['actions'][0]['direction'] = action['direction']
                            except:
                                vehicle['actions'][0]['direction'] = action['direction']
                    except:
                        print("No action ")
                cnt = False
                if(vehicle in sentVehicle[sent]):
                    cnt = True
                tmp = checkLane([act, sent.root], prevSent)
                if(tmp):
                    action['lane'] = tmp
                if(len(vehicle['actions'])==0):
                    vehicle['actions'].append(action)
                elif(vehicle['actions'][0]):
                    vehicle['actions'][0] = action
                if(not cnt):
                    sentVehicle[sent].append(vehicle)
            prevSent = sent
    print(' ')
    otp+= '\nRoad Property\n'
    for road in roads:
        print(road, ' : ', roads[road])
        otp+= json.dumps(road) + ' : ' + json.dumps(roads[road]) + '\n'
    print(' ')
    otp+= '\nVehicles\n'
    for vehicle in vehicles:
        print(vehicle)
        otp += json.dumps(vehicle) + '\n'

    #Prep bench file
    with open(f"{baseDir}\\{sam}\\Case{sam}-NLPF.json", 'w') as outfile:
        outfile.write(otp)
    otp+= '\n\n\n'

with open('json_data.json', 'w') as outfile:
    outfile.write(otp)
