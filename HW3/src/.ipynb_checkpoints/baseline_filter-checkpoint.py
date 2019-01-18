import os
import cv2
import argparse

def detect(image_dir, cascade_file = "../data/lbpcascade_animeface.xml"):
    if not os.path.isfile(cascade_file):
        raise RuntimeError("%s: not found" % cascade_file)

    cascade = cv2.CascadeClassifier(cascade_file)
    files = os.listdir(image_dir)
    for filename in files:
        print(filename)
        image = cv2.imread(image_dir + filename, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = cascade.detectMultiScale(gray,
                                        # detector options
                                        scaleFactor = 1.1,
                                        minNeighbors = 5,
                                        minSize = (24, 24))

        print("Detect {} faces".format(len(faces)))
        if len(faces) >= 1:
            print(filename,"Passed!")
            os.system("cp " + image_dir + filename +
                      " ../data/tag_faces/" + filename)
        else:
            print(filename,"Fail!")
       

parser = argparse.ArgumentParser(description='Baseline Model')
parser.add_argument('--dir', type=str, help='Path to input dir')

args = parser.parse_args()
detect(args.dir)
