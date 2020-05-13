# CGExpan

The source code used for paper "[Empower Entity Set Expansion via Language Model Probing](https://arxiv.org/abs/2004.13897)", published in ACL 2020.

## Data

You can download the [Wiki](https://www.dropbox.com/sh/8ij1xkwqrddy5ad/AACSpXCLfFn2XVxgPb-LTcNNa?dl=0) and [APR](https://www.dropbox.com/sh/c52m31w5zm5j3y7/AACx3UoBZJj4iXsip_HhGmOXa?dl=0) datasets from the following links:

https://www.dropbox.com/sh/7aejy7t1bi9cjdj/AABIK71EcGtI2YAU-IoikK0xa?dl=0

After downloading the dataset, put them under the folder "./data/"

## Run

For Wiki dataset, run 
```
python src/main.py -dataset data/wiki/ -m 2 -gen_thres 3
```

For APR dataset, run 
```
python src/main.py -dataset data/apr/ -m 1 -gen_thres 2
```

Results for each query will be saved under "./data/\[DATA\]/results"

## Pretrained Embedding

To get pre-trained embedding for your own dataset, you need to provide "entity2id.txt" and "sentences.json". Please refer to [SetExpan](https://github.com/mickeystroller/SetExpan) and [HiExpan](https://github.com/mickeystroller/HiExpan) for the preprocessing code.

After putting the required files in your dataset folder "./data/\[DATA\]", you can run the following command to get the pretrained embedding:
```
python src/PretrainedEmb.py -dataset data/[DATA]
```
