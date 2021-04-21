import pickle
import os
<<<<<<< HEAD
from pathlib import Path

def get_project_root() -> str:
    return str(Path(__file__).parent.parent)
=======
from utils import get_project_root
>>>>>>> 618dbdf (first commit)

PATH = get_project_root() + '/data/trained_networks/'


def save_network(network, name):
    pickle.dump(network, open(PATH + name, 'wb'))


def delete_network(name):
    os.remove(PATH + name)


def load_network(name):
    return pickle.load(open(PATH + name, 'rb'))
