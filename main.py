"""
The MIT License (MIT)

Copyright (c) 2014 Douglas Fernandes Silva

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


__author__ = 'Douglas Fernandes Silva'
__email__ = 'douglasfs.ti@gmail.com'
__license__ = 'MIT'
__version__ = '0.02'

import os
import sys
import cv2
from time import sleep
import serial
import requests
import argparse
import numpy as np
from threading import Thread

URL = 'http://127.0.0.1:8080/?action=snapshot'
NAMES = {}
IMAGE_PATH = 'data/images'
MODEL_FILE = 'data/model.mdl'
CASCADE_FACE = 'data/haarcascade_frontalface_alt.xml'
ARDUINO = serial.Serial('/dev/ttyACM0', 9600)


class Facerec(Thread):

    def __init__(self, tries=100):
        Thread.__init__(self)
        self.tries = tries

    def run(self):
        print 'Please wait, processing images.'
        percent, name = prediction()
        if percent > 80:
            print 'You was identified, you shall pass.'
            ARDUINO.write('1')
        else:
            print 'You wasn\'t identified. You shall not pass.'
        print 'Name: {0} | Accuracy level: {1}%'.format(name, percent)
        sleep(5)
        ARDUINO.write('2')
        print 'Press space to quit.'


class Facedetect(Thread):

    def __init__(self):
        Thread.__init__(self)

    def run(self):
        while True:
            get_snapshot('det.jpg')
            image = cv2.flip(cv2.imread('det.jpg'), 1)
            gray = cv2.flip(cv2.imread('det.jpg', cv2.IMREAD_GRAYSCALE), 1)
            coords = detect_face(gray)
            if len(coords) > 0:
                coords *= 2
                marker(image, coords)
            cv2.imshow('Face Recognizer', image)
            if 0xFF & cv2.waitKey(10) == 32:
                break


def read_images(path=IMAGE_PATH, sz=None):
    """
    Reads images in a given folder, resizes images on the fly if size is given.

    :param path: Path to a folder with subfolders representing the subjects persons. - data/images
    :param sz: A tuple with the sizes - (100,100)
    :return:
        - images: the images, which is a Python list of numpy arrays.
        - labels: the corresponding labels (the unique number of the subject person) in a Python list os numpy arrays.
    """

    index, images, labels = 0, [], []

    for dirname, dirnames, filenames in os.walk(path):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                try:
                    image = cv2.imread(os.path.join(subject_path, filename), cv2.IMREAD_GRAYSCALE)
                    if sz is not None:
                        image = cv2.resize(image, sz)
                    images.append(np.asarray(image, dtype=np.uint8))
                    labels.append(index)
                    if index not in NAMES:
                        NAMES[index] = subdirname
                except IOError, (errno, strerror):
                    print 'I/O error({0}): {1}'.format(errno, strerror)
                except:
                    print 'Unexpected error:', sys.exc_info()[0]
                    raise
            index += 1
    images, labels = np.asarray(images), np.asarray(labels, dtype=np.int32)
    return images, labels


def train_model(images, labels):
    try:
        model = cv2.createFisherFaceRecognizer()
        model.train(images, labels)
        model.save(MODEL_FILE)
    except:
        print 'Unexpected error:', sys.exc_info()[0]
        raise


def recognizer(image):
    model = cv2.createFisherFaceRecognizer()
    model.load(MODEL_FILE)
    return model.predict(image)


def normalize_image(image):
    image = cv2.equalizeHist(image)
    image = cv2.bilateralFilter(image, 0, 20., 2.)
    return image


def shrink_image(image, w=320):
    ratio = float(w) / image.shape[1]
    dimensions = (w, int(image.shape[0] * ratio))
    return cv2.resize(image, dimensions, interpolation=cv2.INTER_AREA)


def detect(image, cascade):
    coords = cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30),
                                      flags=cv2.CASCADE_FIND_BIGGEST_OBJECT)
    if len(coords) > 0:
        coords[:, 2:] += coords[:, :2]
        return coords
    return []


def detect_face(image):
    image = normalize_image(shrink_image(image))
    cascade = cv2.CascadeClassifier(CASCADE_FACE)
    return detect(image, cascade)


def prediction(tries=101):
    RESULT = {}
    for i in range(tries):
        get_snapshot('rec.jpg')
        image = cv2.flip(cv2.imread('rec.jpg', cv2.IMREAD_GRAYSCALE), 1)
        coords = detect_face(image)
        if len(coords) > 0:
            roi = normalize_image(crop_face(image, coords * 2))
            roi = cv2.resize(roi, (100, 100), interpolation=cv2.INTER_AREA)
            result, distance = recognizer(roi)
            if NAMES[result] in RESULT:
                RESULT[NAMES[result]] += 1
            else:
                RESULT[NAMES[result]] = 0
        cv2.waitKey(10)
    MAX = max(RESULT.values())
    FINAL = RESULT.keys()[RESULT.values().index(MAX)]
    return MAX, FINAL


def capture(label="guest"):
    check_directory(label)
    index = 0
    while True:
        get_snapshot('cap.jpg')
        gray = cv2.flip(cv2.imread('cap.jpg', cv2.IMREAD_GRAYSCALE), 1)
        image = cv2.flip(cv2.imread('cap.jpg'), 1)
        coords = detect_face(gray)
        if len(coords) > 0:
            coords *= 2
            roi = normalize_image(crop_face(gray, coords))
            roi = cv2.resize(roi, (100, 100), interpolation=cv2.INTER_AREA)
            marker(image, coords)
        cv2.imshow('Capture', image)
        if 0xFF & cv2.waitKey(10) == 32:
            roi_face = detect_face(roi)
            if len(roi_face) > 0:
                cv2.imwrite('{0}/{1}/{2}.jpg'.format(IMAGE_PATH, label, index), roi)
                index += 1
            else:
                print 'Try again, face cannot found in roi of image.'

        if index == 10:
            break


def get_snapshot(file_name='snapshot.jpg'):
    request = requests.get(URL)
    tmp = open(file_name, 'wb')
    tmp.write(request.content)
    tmp.close()


def marker(image, coords, color=(255, 0, 0), border_size=1):
    for x1, y1, x2, y2 in coords:
        cv2.rectangle(image, (x1, y1), (x2, y2), color, border_size)


def crop_face(image, coords):
    x1, y1, x2, y2 = coords[0]
    return image[y1:y2, x1:x2]


def check_directory(label):
    path = os.path.join(IMAGE_PATH, label)
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        print 'Overwriting - folder in use.'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A simple face recognizer tool write in Python with OpenCV Library.')
    parser.add_argument('-t', '--train', help='Train model file', required=False, action='store_true')
    parser.add_argument('-c', '--capture', help='Add new person in image folder', required=False, action='store',
                        nargs=1, metavar='label')
    parser.add_argument('-r', '--recognizer', help='Try recognizer a person from web cam read', required=False,
                        action='store_true')
    args = vars(parser.parse_args())

    if args['capture']:
        label = args['capture'][0]
        print 'Take snapshots in different angles. Hit space bar to get a snapshot.'
        capture(label=label)

    if args['train']:
        images, labels = read_images(sz=(100, 100))
        train_model(images, labels)

    if args['recognizer']:
        images, labels = read_images(sz=(100, 100))
        train_model(images, labels)
        rec = Facerec(100)
        det = Facedetect()

        det.start()
        rec.start()
