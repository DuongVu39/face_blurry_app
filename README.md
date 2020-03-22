## Face Blurrer App
### Duong Vu
### December 2019

An app to blur face if detect any

## Usage

Using default photo:

``` python src/blur_face_in_img.py```

To use your own photo, put it in the data folder then use the input-img argument to specify the image name. 

```python src/blur_face_in_img.py --input-img data/image_name.png```

## Results

Result photo will be saved in result folder

Result will look like this:

![](results/Blurred_default_img.png)

## Required packages

See [requirements.txt](requirements.txt)

## To do:
- Find another face detector model to add on (YOLOv3)
- Implement loading weight file from local path
- Write more unit tests
- Set it as a service to send a picture to and it will spit out the result
