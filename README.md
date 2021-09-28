# Code for Predictive Machine Learning of Objective Boundaries for Solving COPs

This repository contains supplementary code for the paper `Predictive Machine Learning of Objective Boundaries for Solving COPs`.

## Abstract

> Solving Constraint Optimization Problems (COPs) can be dramatically simplified by boundary estimation, that is, providing tight boundaries of cost functions. By feeding a supervised Machine Learning (ML) model with data composed of known boundaries and extracted features of COPs, it is possible to train the model to estimate boundaries of a new COP instance. In this paper, we first give an overview of the existing body of knowledge on ML for Constraint Programming (CP) which learns from problem instances. Second, we introduce a boundary estimation framework that is applied as a tool to support a CP solver. Within this framework, different ML models are discussed and evaluated regarding their suitability for boundary estimation, and countermeasures to avoid unfeasible estimations that avoid the solver to find an optimal solution are shown. Third, we present an experimental study with distinct CP solvers on seven COPs. Our results show that near-optimal boundaries can be learned for these COPs with only little overhead and that these estimated boundaries can help the solver to find near-optimal solutions early during search.

## Relevant Scripts

### `main.py`

Main script for training and evaluating the machine learning models.
It also supports the generation of scripts to run the solvers with the estimated boundaries.

```
$ python main.py --help          
Using TensorFlow backend.
usage: main.py [-h]
               [-est {network,networka,knn,forest,gp,bayridge,ridge,svm,linear,xgb,xgba}]
               [-o {1,2}] [-e ENSEMBLE]
               [-em {extreme,average,median,leaveoneout}] [-i ITER] [-v]
               [-af ADJUSTMENT] [-l {mse,mae,linex,shiftedmse,peann}]
               [-lf LOSS_FACTOR] [-f] [--grid-search]
               [--file-suffix FILE_SUFFIX] [-val]
               [-s {all,chuffed,gecode,cbc,cplex,ortools,choco,sunnycp}]
               [--validation-file VALIDATION_FILE] [-of OUTPUT_STATS]
               [-op OUTPUT_PREDICTIONS] [--smallfeat] [-scaled]
               [-use_firstsol]
               problem_name

positional arguments:
  problem_name

optional arguments:
  -h, --help            show this help message and exit
  -est {network,networka,knn,forest,gp,bayridge,ridge,svm,linear,xgb,xgba}, --estimator {network,networka,knn,forest,gp,bayridge,ridge,svm,linear,xgb,xgba}
  -o {1,2}, --outputs {1,2}
  -e ENSEMBLE, --ensemble ENSEMBLE
  -em {extreme,average,median,leaveoneout}, --ensemble-mode {extreme,average,median,leaveoneout}
  -i ITER, --iter ITER
  -v, --verbose         increase output verbosity
  -af ADJUSTMENT, --adjustment ADJUSTMENT
  -l {mse,mae,linex,shiftedmse,peann}, --loss-fn {mse,mae,linex,shiftedmse,peann}
  -lf LOSS_FACTOR, --loss-factor LOSS_FACTOR
                        Asymmetry factor (only for linex/shiftedmse/peann
  -f, --force-new       Overwrites any existing model
  --grid-search
  --file-suffix FILE_SUFFIX
  -val, --validate
  -s {all,chuffed,gecode,cbc,cplex,ortools,choco,sunnycp}, --solver {all,chuffed,gecode,cbc,cplex,ortools,choco,sunnycp}
                        If given, these solvers are used for validation of
                        estimated boundaries.
  --validation-file VALIDATION_FILE
                        File to store the validation commands, if a solver is
                        to be used.
  -of OUTPUT_STATS, --output-stats OUTPUT_STATS
                        File to store statistics about the trained estimator.
  -op OUTPUT_PREDICTIONS, --output-predictions OUTPUT_PREDICTIONS
                        File to store predictions on validation instances.
  --smallfeat
  -scaled, --scaled-prediction
                        Predict the objective in relation to the instance's
                        domain bounds and scale back to ints
  -use_firstsol
```

### `solve.py`

Running a solver + model with given boundaries.

```
$ python solve.py --help
usage: solve.py [-h] [-lb LOWERBOUND] [-ub UPPERBOUND]
                [-s {all,chuffed,gecode,cbc,cplex,ortools,choco,sunnycp}]
                [-t TIMEOUT] [-f] [--no-timeout] [-ds DATASET]
                [--output OUTPUT] [--pymzn]
                problem dzn

positional arguments:
  problem               Problem name or path to minizinc model
  dzn

optional arguments:
  -h, --help            show this help message and exit
  -lb LOWERBOUND, --lowerbound LOWERBOUND
  -ub UPPERBOUND, --upperbound UPPERBOUND
  -s {all,chuffed,gecode,cbc,cplex,ortools,choco,sunnycp}, --solver {all,chuffed,gecode,cbc,cplex,ortools,choco,sunnycp}
  -t TIMEOUT, --timeout TIMEOUT
  -f, --free            Free search (if supported by solver)
  --no-timeout
  -ds DATASET, --dataset DATASET
  --output OUTPUT
  --pymzn
```

There is also `setup_solver_commands.py` which prepares multiple calls for use in a batch processing environment.

### Other

- `features.py`: Feature extraction from MiniZinc instances for experiments.
- `evaluation.py` & `figures.py`: Generation of figures and tables from log files.
- `model.py`: Implementation of machine learning models and custom losses
- `network.py`: Neural network code
- `keras_helper.py` & `keras_losses.py`: Custom losses and functions for neural networks in keras


## Setup

Create a new Python 3.6 (others might work, too) environment and install the requirements, e.g. via
``pip install -r requirements.txt``

## Dataset

To embed objective boundaries during search a modified version of the MiniZinc models is required and available here: https://github.com/HelgeS/minizinc-benchmarks
This could be automated, but it is left as an exercise for future work.

## License

All of our code is licensed under [MIT License](LICENSE).
Third-party code remains under its original license.
