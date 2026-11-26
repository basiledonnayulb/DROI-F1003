"""Rebuild the training set and train both models.
Usage: python train.py
"""
from pathlib import Path
from src.prepare import prepare
from src.model import train
ROOT=Path(__file__).resolve().parent
if __name__=='__main__':
    prepare(ROOT/'data'/'irregular.xlsx',ROOT/'data'/'population.xlsx',
            ROOT/'data'/'source.sqlite',ROOT/'data'/'training.csv')
    train()
