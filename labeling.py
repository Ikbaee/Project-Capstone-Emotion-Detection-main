from doctest import Example
import cv2
from shutil import copyfile

from urllib.request import urlopen
import numpy as np
from deepface.DeepFace import analyze
from os import path, makedirs
from glob import glob

rawdir = 'dataset/raw/'
cleandir = 'dataset/clean-labeled/'
labels = ['angry', 'sad', 'happy', 'fear', 'disgust', 'surprise', 'neutral', 'unidentified']

def makedirs_raw_clean():
    makedirs(rawdir,exist_ok=True)
    makedirs(cleandir,exist_ok=True)

def makedirs_label():
    for label in labels:
        dir_label = path.join(*[cleandir, label])
        makedirs(dir_label,exist_ok=True)

def url_to_image(url, readFlag=cv2.IMREAD_COLOR):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    resp = urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, readFlag)

    # return the image
    return image

def crop_image(image, bounding_box):
    x, y, w, h = bounding_box[0]
    img_cropped = image[y:y + h, x:x + w]
    return img_cropped

def image_detection(img):
    gray_img = cv2.cvtColor(img ,cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    face_bb = face_cascade.detectMultiScale(gray_img, 1.1, 4)
    if len(face_bb) == 0:
        return None
    crop_img = crop_image(img, face_bb)
    return crop_img

def face_expression(img):
    analysis = analyze(img, actions = ["emotion","dominant_emotion"], detector_backend='mtcnn')
    return analysis

def image_expression_prediction(img):
    crop_face = image_detection(img)
    if crop_face is None:
        return None
    analysis = face_expression(crop_face)
    return analysis, crop_face

def labeling_image():
    for img in glob(path.join(*[rawdir, '*.jpg'])):
        imgname = path.basename(img)
        image = cv2.imread(img)
        
        try:
            result = image_expression_prediction(image)
        except ValueError:
            print('image name: ', imgname)
            savepath = path.join(*[cleandir,'unidentified',imgname])
            copyfile(img, savepath)
            continue   
        if result is None:
            savepath = path.join(*[cleandir,'unidentified',imgname])
            copyfile(img, savepath)
            continue
        analysis, crop_face = result
        label = analysis['dominant_emotion']
        savepath = path.join(*[cleandir,label,imgname])
        cv2.imwrite(savepath,crop_face)

# makedirs_raw_clean()
# makedirs_label()
labeling_image()