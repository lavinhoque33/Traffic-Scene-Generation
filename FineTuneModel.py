from datasets import load_dataset
import json

def getlabel(tag):
    if(tag == 1):
        ret = 'WORK_OF_ART'
    elif(tag ==2):
        ret = 'FAC'
    elif (tag == 3):
        ret = 'EVENT'
    elif (tag == 4):
        ret = 'LOC'
    elif (tag == 5):
        ret = 'ORG'
    elif (tag == 7):
        ret = 'PERSON'
    elif (tag == 8):
        ret = 'PRODUCT'
    else:
        ret = ''
    return ret

dataset = load_dataset("dfki-nlp/few-nerd", "supervised")
# print(dataset['train'].features['fine_ner_tags'])
train = []
for i in range(len(dataset['train'])):
    if(59 in dataset['train'][i]['fine_ner_tags']):
        sent2 = ' '.join(dataset['train'][i]['tokens'])
        sent = ' '.join(dataset['train'][i]['tokens'])
        enttags = []
        entities = {}
        for j in range(len(dataset['train'][i]['tokens'])):
            start = sent.index(dataset['train'][i]['tokens'][j])
            end = start + len(dataset['train'][i]['tokens'][j])
            sent = sent[0:start] + ' ' * len(dataset['train'][i]['tokens'][j]) + sent[end:]
            lab = getlabel(dataset['train'][i]['ner_tags'][j])
            if(len(lab)>0):
                enttags.append((start,end,lab))
        entities['entities'] = enttags
        train.append((sent2,entities))
        # print(dataset['train'][i])

for i in range(len(dataset['validation'])):
    if(59 in dataset['validation'][i]['fine_ner_tags']):
        sent2 = ' '.join(dataset['validation'][i]['tokens'])
        sent = ' '.join(dataset['validation'][i]['tokens'])
        enttags = []
        entities = {}
        for j in range(len(dataset['validation'][i]['tokens'])):
            start = sent.index(dataset['validation'][i]['tokens'][j])
            end = start + len(dataset['validation'][i]['tokens'][j])
            sent = sent[0:start] + ' ' * len(dataset['validation'][i]['tokens'][j]) + sent[end:]
            lab = getlabel(dataset['validation'][i]['ner_tags'][j])
            if(len(lab)>0):
                enttags.append((start,end,lab))
        entities['entities'] = enttags
        train.append((sent2,entities))

for i in range(len(dataset['test'])):
    if(59 in dataset['test'][i]['fine_ner_tags']):
        sent2 = ' '.join(dataset['test'][i]['tokens'])
        sent = ' '.join(dataset['test'][i]['tokens'])
        enttags = []
        entities = {}
        for j in range(len(dataset['test'][i]['tokens'])):
            start = sent.index(dataset['test'][i]['tokens'][j])
            end = start + len(dataset['test'][i]['tokens'][j])
            sent = sent[0:start] + ' '*len(dataset['test'][i]['tokens'][j]) + sent[end:]
            lab = getlabel(dataset['test'][i]['ner_tags'][j])
            if(len(lab)>0):
                enttags.append((start,end,lab))
        entities['entities'] = enttags
        train.append((sent2,entities))

textfile = open("train.txt", "w")
json.dump(train,textfile)
textfile.close()


# for i in range(len(dataset['validation'])):
#     if(59 in dataset['validation'][i]['fine_ner_tags']):
#         count += 1
#         # print(dataset['train'][i])
# print(count)
#
# for i in range(len(dataset['test'])):
#     if(59 in dataset['test'][i]['fine_ner_tags']):
#         count += 1
#         # print(dataset['train'][i])
# print(count)