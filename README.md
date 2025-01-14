## _Update: 2021.12.18_

This is an expansion of [pycococreator](https://github.com/waspinator/pycococreator).
For more convenient use, the repository combines [pycococreator](https://github.com/waspinator/pycococreator) with [Open Images Dataset](https://opensource.google/projects/open-images-dataset).


# pycococreator


Please site using: [![DOI](https://zenodo.org/badge/127320367.svg)](https://zenodo.org/badge/latestdoi/127320367)


pycococreator is a set of tools to help create [COCO](http://cocodataset.org) datasets. It includes functions to generate annotations in uncompressed RLE ("crowd") and polygons in the format COCO requires.

Read more here https://patrickwasp.com/create-your-own-coco-style-dataset/

![alt text](https://i.imgur.com/iQSPjeC.png "input files")
![alt text](https://i.imgur.com/py2aYK9.png "output")

# Install

`pip install git+git://github.com/waspinator/pycococreator.git@0.2.0`

If you need to install pycocotools for python 3, try the following:

```
sudo apt-get install python3-dev
pip install cython
pip install git+git://github.com/waspinator/coco.git@2.1.0
```
