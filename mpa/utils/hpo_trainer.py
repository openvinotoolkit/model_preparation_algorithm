from mpa.utils.hpo_stage import run_hpo_trainer
import pickle
import sys

if __name__ == '__main__':
    with open(sys.argv[1], "rb") as pfile:
        kwargs = pickle.load(pfile)
        run_hpo_trainer(**kwargs)
