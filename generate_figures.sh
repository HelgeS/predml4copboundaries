#!/usr/bin/env bash
#python evaluation.py instances
#python evaluation.py losses
#python evaluation.py adjustment -f logs/one_output/*-stats.log
#python evaluation.py adjustment -f logs/two_outputs/*-stats.log
#python evaluation.py adjustmentg -f logs/one_output/*-stats.log
#python evaluation.py adjustmentg -f logs/two_outputs/*-stats.log
#python evaluation.py estimation -f logs/one_output/*-stats.log
#python evaluation.py estimation -f logs/two_outputs/*-stats.log
#python evaluation.py solver -f logs/bounded_random_median_1800_o1_sunnycp.csv logs/unbounded_random_1800_sunnycp.csv
#python evaluation.py solver -f logs/bounded_best_median_1800_o1_sunnycp.csv logs/unbounded_best_1800_sunnycp.csv
#python evaluation.py solver -f logs/bounded_random_median_1800_o1_sso.csv logs/unbounded_random_1800_sso.csv
python evaluation.py solver -f logs/bounded_best_median_1800_o1_sso.csv logs/unbounded_best_1800_sso.csv
#python evaluation.py solver -f logs_solver_bounded_o2.csv