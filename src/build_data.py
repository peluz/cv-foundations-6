import os
from utils import *
from pathlib import Path

DIRNAME = os.path.dirname(__file__)
DATA_PATH = os.path.join(DIRNAME, "../data/lfw-deepfunneled")

people_dict = Path(os.path.join(DATA_PATH, "../people_dict.pkl"))
if not people_dict.is_file():
    extractor = feature_extractor()
    build_people_dictionary(extractor)
people_dict = load_people_dict()

write_tsv(os.path.join(DATA_PATH, "../peopleDevTrain.txt"),
          os.path.join(DATA_PATH, "../peopleDevTrain.tsv"),
          people_dict)

write_tsv(os.path.join(DATA_PATH, "../peopleDevTest.txt"),
          os.path.join(DATA_PATH, "../peopleDevTest.tsv"),
          people_dict)