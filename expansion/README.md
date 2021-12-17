## _Update: 2021.12.18_

This is an expansion of [pycococreator](https://github.com/waspinator/pycococreator).
For more convenient use, the repository combines [pycococreator](https://github.com/waspinator/pycococreator) with [Open Images Dataset](https://opensource.google/projects/open-images-dataset).


## Installation
### Requirements
- Python 3.6+
- [pycococreator](https://github.com/waspinator/pycococreator)
- [FiftyOne](https://github.com/voxel51/fiftyone)

## How does it work?
The following convenient functions have been provided in _**oiv6_tools.py**_:
- _**download_objs:**_ Download the specific categories from Google's Open Images Dataset(v6).
- _**inventory_dataset:**_ Invent the available data from downloaded dataset.
- _**split_dataset:**_ Split the dataset as Train and Validation.
- _**convert_cocoformat:**_ Convert images and masks to COCO format.

_For more detail, please see: **main.py**_

## Acknowledgements
- To the FiftyOne team for providing such an awesome framework
- To the pycococreator team for providing such convenient api
- To all python developers that made packages used in this repository
