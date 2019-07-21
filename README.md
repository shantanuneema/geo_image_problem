# geo_image_problem
image similarity problem

## Image similarity search using FastAI and Locality Sensitive hashing (Annoy)
### On fine grain microscopic rock thin sections

__High-level approach__ -
1) Transfer learning from a ResNet-34 model(trained on ImageNet) to develop an image classifier for igneous and metamorphic rock thin sections.
2) Take the output of second last fully connected layer from trained to get embedding for ~30,000 images.
3) Use Locality Sensitive hashing using Annoy by Spotify to create LSH hashing which enables fast approximate nearest neighbor search
4) A notebook is created where a stored trained model can be utilized by any user to make choices and find top k similar images to the input image (provided under /geological_similarity data)

__Implementation__

A detail notebook using the given dataset to generate model: https://github.com/shantanuneema/geo_image_problem/blob/master/notebooks/develop_gi_similarity_model.ipynb

A user interface to find top k similar rocks by using different models, and distance types:
https://github.com/shantanuneema/geo_image_problem/blob/master/notebooks/run_gi_sim.ipynb
