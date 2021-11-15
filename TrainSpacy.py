# Load a spacy model and chekc if it has ner
import spacy
import json
import random
from spacy.util import minibatch, compounding
from pathlib import Path

nlp=spacy.load('en_core_web_lg')

# Getting the pipeline component
ner=nlp.get_pipe("ner")

with open('train.txt') as json_file:
    TRAIN_DATA = json.load(json_file)

# Disable pipeline components you dont need to change
pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
unaffected_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

# TRAINING THE MODEL
with nlp.disable_pipes(*unaffected_pipes):

  # Training for 30 iterations
  for iteration in range(40):

    # shuufling examples  before every iteration
    random.shuffle(TRAIN_DATA)
    losses = {}
    # batch up the examples using spaCy's minibatch
    batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
    for batch in batches:
        texts, annotations = zip(*batch)
        nlp.update(
                    texts,  # batch of texts
                    annotations,  # batch of annotations
                    drop=0.3,  # dropout - make it harder to memorise data
                    losses=losses,
                )
    print("Losses", losses)

# Save the  model to directory
output_dir = Path('./content/')
nlp.to_disk(output_dir)
print("Saved model to", output_dir)