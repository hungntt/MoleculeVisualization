import numpy as np
import pandas as pd

import ase.visualize

import matplotlib.pyplot as plt
import seaborn as sns

import os

from IPython.core.display import display


def load_dir_csv(directory):
    csv_files = sorted([f for f in os.listdir(directory) if f.endswith(".csv")])
    csv_vars = [filename[:-4] for filename in csv_files]
    gDict = globals()

    for filename, var in zip(csv_files, csv_vars):
        print(f"{var:32s} = pd.read_csv({directory}/{filename})")
        gDict[var] = pd.read_csv(f"{directory}/{filename}")
        print(f"{'nb of rows ':32s} = " + str(len(gDict[var])))
        display(gDict[var].head())
