# DIRT/mu 1.0
A software to automatically extract root hairs from microscopy images, developed by Peter Pietrzyk.

This software is written and tested in:
- Python 2.7 (https://www.python.org)
 
# Installation
1. Clone the repository into your local machine
```bash
git clone https://github.com/Computational-Plant-Science/DIRTmu.git
```
2. Use conda to create the environment from the provided .yml file
```bash
conda env create -f hairyroots_py2718_64.yml
```


# Running the sample file
To process the sample image from the command line type:
```bash
python hairyroots.py -i /sample/TAKFA3-n3-1_Classes.tiff -o /sample/ --cost_type rms --w_mind 10 --w_len 5 --id_root 3 --id_roothair 1 --id_background 2 --pixel_size 1.72 -p 
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
