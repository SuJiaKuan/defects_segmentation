# Defects Segmentation

Tools about defects segmentation.

## Prerequisites

Plase make sure you already following software in your computer:
- Pyhton 3
- pip3

And install the depency packages:

```bash
pip3 install requirements.txt
```

Plase also download [mazu_7200.zip](https://drive.google.com/file/d/1Nxut1cgtWspubyvUSYKOqCMmZUO10-mK/view?usp=sharing) and extract it under the project root folder, its folder name is `mazu_7200`.

## Dataset Generation

The script `generate_dataset.py` accepts an input images folder and generate an dataset for semantic segmentation.

* Take 100 images from `mazu_7200` and output the dataset in folder `output`:

```bash
python3 generate_dataset.py mazu_7200 -l 100
```

* Take all images from `mazu_7200` instead of just 100 imags:

```bash
python3 generate_dataset.py mazu_7200
```

* Change the ratio of testing set from 0.1 to 0.2:

```bash
python3 generate_dataset.py mazu_7200 -t 0.2
```

* For more and detailed arguments:

```bash
python3 generate_dataset.py -h
```

## Development

The script `seg_dev.py` accepts an input image pair and visualize th result for semantic segmentation dataset generation.

* Take an image pair from `mazu_7200`, visualize the result, and save as `output.png`:

```bash
python3 seg_dev.py mazu_7200/clean/00000001.png mazu_7200/noisy/00000001.png
```
