import gurobipy as gp
from gurobipy import GRB

N = 50; # Number of constraints
n = 5; # Number of variables

A = sprandn (N ,n ,0.1); b = ones (N ,1);
x0 = 10* randn (n ,1);