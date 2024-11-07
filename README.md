
# AMNCutter: Affinity-Attention-Guided Multi-View Normalized Cutter for Unsupervised Surgical Instrument Segmentation [[Paper Link](https://arxiv.org/abs/2411.03695)]


[Mingyu Sheng](https://mhamilton.net/)<sup>1</sup>,
[Jianan Fan](https://ieeexplore.ieee.org/author/37090057230)<sup>1</sup>,
[Dongnan Liu](https://scholar.google.com/citations?hl=en&user=JZzb8XUAAAAJ)<sup>1</sup>,
[Ron Kikinis](https://scholar.google.com/citations?user=n01L0mEAAAAJ&hl=en&oi=ao)<sup>2</sup>,
[Weidong Cai](https://scholar.google.com/citations?hl=en&user=N8qTc2AAAAAJ)<sup>1</sup>

<sup>1</sup> [The University of Sydney](https://www.sydney.edu.au/), 
<sup>2</sup> [Harvard Medical School](https://hms.harvard.edu/)

This paper was accepted by the 2025 IEEE/CVF Winter Conference on Applications of Computer Vision ([WACV 2025](https://wacv2025.thecvf.com/)).

## Method Overview

![](paperFigure/method_overview_white.png)

## Setup
* Recommended Environment: Python 3.10.0+, Cuda 12.0+
* Install dependencies: `pip install -r requirements.txt`.

## Dataset Directory
```
dataset
    └── DatasetName
        ├── groundTruth
        │   ├── binary
        │   │   ├── videoName1
        │   │   |   ├── frameName1.png
        │   │   |   └── frameName2.png
        │   │   └── videoName2
        │   ├── binaryClassIndicator.json
        │   ├── parts
        │   ├── partsClassIndicator.json
        │   ├── semantic
        │   ├── semanticClassIndicator.json
        │   ├── types
        │   └── typesClassIndicator.json
        └── inputX
            └── originImage
                ├── videoName1
                |   ├── frameName1.png
                |   └── frameName2.png
                └── videoName2
```
A demo dataset is uploaded and is named "Demo".

`inputX`: Origin input frames are stored.

`groundTruth`: Ground truth masks and class indicators (json files) for all segmentation tasks are stored. Masks are stored in gray scale. Mapping between gray scale values and labels is in the class indicator json files.

## Data Splitting

```bash
python dataPreprocessing/data_train_val_test_split.py --config_file dataPreprocessing/demo_dataset_split.json
```

All datasets and their configuration are listed in the config file, including the training and testing sets.

You can adjust the config file for different data splitting purposes.

It will generate a "data_split" directory including 'train.txt', and 'test.txt', under the "DatasetName" directory. `samples.txt` is used for visualization.

Or, you can manually set `txt` files.

## Run

The following command trains an AMNCutter model with the datasets claimed in the 'json' config file.
```bash
python main.py --mode train --config_file configFiles/demo_AMNCutter.json
```


The following command evaluates an AMNCutter model with the datasets claimed in the 'json' config file.
```bash
python main.py --mode test --config_file configFiles/demo_AMNCutter.json
```


The following command visualizes the prediction results and some intermediate feature maps, while it may be slower.
```bash
python main.py --mode test --vis --config_file configFiles/demo_AMNCutter.json
```

Change configurations in the config file as you want, for more experiments.

`python experiment.py` can run a list of training and testing tasks.

## Download Trained Model

Please download [our model weights](https://drive.google.com/drive/folders/1NSS5sTWtBFGevEb_sbWhYjAxE2nOQMLu?usp=drive_link), and merge it with `saved_models`.
Please adjust configuration files in `configFiles` to leverage the downloaded weights.

## Outputs

A directory `outputs` will be generated after running.
It contains csv files of evaluation and output pictures of visualization.


## Citation

Arxiv BibTex:
```
@misc{sheng2024amncutteraffinityattentionguidedmultiviewnormalized,
      title={AMNCutter: Affinity-Attention-Guided Multi-View Normalized Cutter for Unsupervised Surgical Instrument Segmentation}, 
      author={Mingyu Sheng and Jianan Fan and Dongnan Liu and Ron Kikinis and Weidong Cai},
      year={2024},
      eprint={2411.03695},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2411.03695}, 
}
```

WACV BibTex:
Coming soon.



