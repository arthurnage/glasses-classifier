# Glasses or no glasses classifier

### MTCNN (https://github.com/ipazc/mtcnn) face detector has been used to crop faces from input images and store then in ```{input_dir}_faces``` directory 

### The model (2.0 mb) consists of parts of a small pretrained mobilenetv3 (https://github.com/d-li14/mobilenetv3.pytorch)

### MeGlass_120x120.zip (https://github.com/cleardusk/MeGlass) dataset has been used for training (top validation accuracy is 0.998)

### main.py performs classification of images from ```{input_dir}```.
Run classification
```
python3 main.py {input_dir}
```
