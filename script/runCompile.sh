#!/bin/sh
#SBATCH --job-name=compileCFun   # job name for easy identification in queue
#SBATCH --time=00:5:00   # walltime
#SBATCH --ntasks=1   # number of tasks (i.e. processes)
#SBATCH --cpus-per-task=2   # number of CPUs per task
#SBATCH --mem-per-cpu=2G   # memory per CPU (total 4*3 = 12 GB per process)
srun matlab -nodisplay -r "try, CompileFile(); fprintf('%d',$SLURM_ARRAY_TASK_ID); exit; catch e, disp('-----Exception:-----'), disp(getReport(e)), exit(1), end"