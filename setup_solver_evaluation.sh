#!/usr/bin/env bash
python setup_solver_commands.py mrcpsp -s chuffed gecode ortools -t 14400 -n 30 --cluster --predictions logs/model_benchmark_o2_scaled_y/*-predict.log
python setup_solver_commands.py rcpsp -s chuffed gecode ortools -t 14400 -n 30 --cluster --predictions logs/model_benchmark_o2_scaled_y/*-predict.log
python setup_solver_commands.py 2DBinPacking -s chuffed gecode ortools -t 14400 -n 30 --cluster --predictions logs/model_benchmark_o2_scaled_y/*-predict.log
python setup_solver_commands.py cutstock -s chuffed gecode ortools -t 14400 -n 30 --cluster --predictions logs/model_benchmark_o2_scaled_y/*-predict.log
python setup_solver_commands.py jobshop -s chuffed gecode ortools -t 14400 -n 30 --cluster --predictions logs/model_benchmark_o2_scaled_y/*-predict.log
python setup_solver_commands.py vrp -s chuffed gecode ortools -t 14400 -n 30 --cluster --predictions logs/model_benchmark_o2_scaled_y/*-predict.log
python setup_solver_commands.py open_stacks -s chuffed gecode ortools -t 14400 -n 30 --cluster --predictions logs/model_benchmark_o2_scaled_y/*-predict.log
cat run_motivation_{mrcpsp,rcpsp,2DBinPacking,cutstock,jobshop,vrp,open_stacks}.sh > solver_commands_bounded_o2_median_random_30inst
