# DIRT/mu 1.1
A software to automatically extract root hairs from microscopy images, developed by Peter Pietrzyk.

This software is written and tested in:
- Python 3.9.10 (https://www.python.org)
 
Updates in version 1.1
 - Upgraded to Python 3.9.10
 - Calculates Root Mean Square for distance and curvature metrics
 - Adds min, mean and std for rolling sum density
 - Adjust curvature when merging candidates
 - Improves normalization of metrics
 - Accepts images without root

# Installation
1. Clone the repository into your local machine
```bash
git clone https://github.com/Computational-Plant-Science/DIRTmu.git
```
2. Use conda to create the environment from the provided .yml file
```bash
conda env create -f dirtmu_environment_python3.yml
```


# Running the sample file
To process the sample image from the command line type (example):
```bash
python3 hairyroots.py -i sample/TAKFA3-n3-1_Classes.tiff -o sample/ --id_root 3 --id_roothair 1 --id_background 2 --pixel_size 1.72 -p 
```

# Help
To show a help messeage type:
```bash
python hairyroots.py -h
```
or
```bash
python hairyroots.py --help
```
