# Rule-based Traﬀic-scene Generation from Natural Language Description

A Rule-based Traffic-scene generation tool. The purpose of this tool is to extract all actor(vehicle, pedestrian) and environment(road,road-type,traffic signal etc.) 
related information from a Natural Language description and store it/use it for next stage in the pipeline, which is to actually ue those extracted informations to generate
still image or a driving simulator(e.g. CARLA) environment/scene.

This tool is Python based since I needed to use lots of NLP tools like entity recognition, dependency parser, coreference resolution, semantic parser etc. Along with that I also used
a traffic scene based Ontology for general traffic scene related knowledge as basis. UnifiedQA model was used heavily as a QA tool to input scene and its extracted context,
while specific questions were asked to the QA model to ultimately extract scene related information.

## Sample Input

It is a one-way, double-lane city road.
Two black hatchbacks, one brown hatchback and one white hatchback are parked off-street on the left side. 
A pickup-truck and one red hatchback is parked on the right side of the road. 
A ego-vehicle is travelling east at a speed of 30mph through the right lane. A silver hatchback is ahead of the ego-vehicle on the right lane travelling at 40mph.
One black SUV is on the left lane travelling at 35mph. A brown hatchback is following the black SUV at 35mph.

## Extracted JSON Output

{"LOC": ["east"],

"Road Properties": ["one-way","double lane"],

"Relative LOC": ["left","right","right","right","left"],

"MISC": ["off-street","speed"],

"Color": ["Black","black","black","brown","white","silver"],

"Vehicles": ["Black hatchback","Black hatchback","brown hatchback","white hatchback","red hatchback","pickup-truck","silver hatchback","Black SUV","brown hatchback","ego-vehicle"],

"Cardinal": ["two","one","one","one","one"],

"Quantity": ["40mph","35mph","30mph","35mph"],
}


## How to Run

`$ python sceneExtract.py`

Need to provide the correct input(scene description) file


## Dependencies

- Python 3.9(Older versions may not work)
- Spacy 3.2.3
- UnifiedQA (3B version)(latest)
- Fastcoref 2.1.1


## Acknowledgements

This tools is heavily insipred by the work [Automatic Crash Constructor from Crash Report (A3CR)](https://github.com/TriHuynh00/AC3R-Demo)
