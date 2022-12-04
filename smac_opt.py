from pathlib import Path
import pandas as pd
# import pyyaml module
import yaml
from yaml.loader import SafeLoader
import numpy as np
import time
import random 
## SMAC Packages 
from ConfigSpace import ConfigurationSpace, Integer, Float, Categorical
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter, UniformFloatHyperparameter
from smac.facade.smac_ac_facade import SMAC4AC
from smac.scenario.scenario import Scenario
import os
import sys

from utilities import get_filename
from scip_solve import run_instance

def save_smac_cut_selec_param(temp_dir, instance, rand_seed,l1,l2,l3,l4):
    """
    Creates a npy file for the default cut-selector parameter values
    Args:
        temp_dir (dir): Directory where we will dump the npy file
        instance (str): The name of the instance
        rand_seed (int): The random seed of the solve
    Returns: Nothing, just creates a file
    """

    cut_selector_params = np.array([l1,l2,l3,l4])
    file_name = get_filename(temp_dir, instance, rand_seed, trans=True, root=False, sample_i=0, ext='npy')
    np.save(file_name, cut_selector_params)

    return


def simple_smac_runner(train_df, test_df):
  """
  Pass in the default run results data frame. One for the test set, one for the train set 
  """
  l1 = Float('lambda1', bounds = (0,1))
  l2 = Float('lambda2', bounds = (0,1))
  l3 = Float('lambda3', bounds = (0,1))
  l4 = Float('lambda4', bounds = (0,1))
  configspace = ConfigurationSpace()

  configspace.add_hyperparameters([l1, l2, l3, l4])
  scenario = Scenario({
      "run_obj": "quality",
      "runcount-limit": 200,  
      "cs": configspace,
      "deterministic": False
  })
  smac = SMAC4AC(scenario=scenario, tae_runner=lambda x: evaluation(x, train_df))
  bfc = smac.optimize()
  return bfc

def solve_instance(instance_name, allowed_runtime, lambda1, lambda2, lambda3, lambda4,root):

  # step 1: get mip based of instance_name
  rand_seed = 1
  time_limit = allowed_runtime

  # Change the outfile directory for this set of runs
  outfile_sub_dir_name = 'root_solve' #if root else 'full_solve'
  outfile_dir_ = os.path.join(outfile_dir, outfile_sub_dir_name)
  try:
        os.mkdir(outfile_dir_)
  except:
        pass
# step 2: set scip params using lambda's and allowed_runtime
  save_smac_cut_selec_param(temp_dir, instance_name, rand_seed,lambda1, lambda2, lambda3, lambda4)
  mps_file = get_filename(data_dir, instance_name, rand_seed, trans=True, root=False, sample_i=None, ext='mps')

  
  try:
    data = run_instance(temp_dir, mps_file, instance_name, rand_seed,0,time_limit, root, False, True)
    return data['solve_time'],data['gap']
  except:
    return -1,-1

def evaluation(params, df, n_samples = 5, gap_tolerance = 0.0002, epsilon = 1e-82):
  n_samples = min(df.shape[0], n_samples)
  lambda1 = params['lambda1']
  lambda2 = params['lambda2']
  lambda3 = params['lambda3']
  lambda4 = params['lambda4']
  instance_list = df.sample(n_samples).to_dict()
 
  scores = []
  for j in range(len(instance_list['NAME'])):

    runtime, mip_gap = solve_instance(instance_list['NAME'][j], instance_list['SOLUTION TIME'][j],lambda1, lambda2, lambda3, lambda4,root = False)
    if runtime <0 or mip_gap < 0:
        print("error on 1 instance causing 1 SMAC run to not average over {} instances".format(n_samples))
        pass
    else:
        if mip_gap < gap_tolerance: ## If Solved, Score the runtime
            imp = float((instance_list['SOLUTION TIME'][j] - runtime)) / (np.abs(instance_list['SOLUTION TIME'][j]) + 1e-8)
            scores.append(imp)
        else:  ## If not solved, score the MIP relative imrovment
            imp = float((instance_list['gap'][j] - mip_gap)) / (np.abs(instance_list['gap'][j]) + 1e-8)
            scores.append(imp)

  return np.mean(scores)

def hetero_runner(file_name):
  """
  Pass in the file that contains the MIPLIB default run results.
  Works for both 
  """
  df = pd.read_csv(file_name)
  shuffled = df.sample(frac=1)
  splits = np.array_split(shuffled, 20)
  results = []
  for i in range(20):
    test = splits[i].copy()
    train_list = [splits[j] for j in range(20) if j != i]
    train = pd.concat(train)
    results.append(smac_runner(train_df, test_df))
  return pd.concat(results)





# folder where yml files will be store
temp_dir = "/home/arnaud/Documents/mie1666/MIE-1666-Project/smac_folder/experiments"
# folder where transformed problems defult shoud be
data_dir = "/home/arnaud/Documents/mie1666/MIE-1666-Project/smac_folder/transformed_problems"
# output files will remain empty
outfile_dir = "/home/arnaud/Documents/mie1666/MIE-1666-Project/smac_folder/output_files"
try: os.mkdir(outfile_dir)
except:pass
try: os.mkdir(temp_dir)
except:pass
df = pd.read_csv("/Users/arnauddeza/Documents/first_yr_MASC/Khalil_research/Cut_Opt/df_default.csv")
msk = np.random.rand(len(df)) < 0.8
train_df = df[msk]
test_df = df[~msk]
bfc = simple_smac_runner(train_df, test_df)


print(bfc)



exit()
root = False
lambda1 = bfc['lambda1']
lambda2 = bfc['lambda2']
lambda3 = bfc['lambda3']
lambda4 = bfc['lambda4']
instance_list = df.sample(2).to_dict()
'''
scores = []
for j in range(len(instance_list['NAME'])):

    runtime, mip_gap = solve_instance(instance_list['NAME'][j], instance_list['SOLUTION TIME'][j],lambda1, lambda2, lambda3, lambda4,root = False)
    if instance_list['NAME'][j] == "23588":
        default_runtime = 4.979511
    if instance_list['NAME'][j] == "22433":
        default_runtime = 1.9013849999999999
    print("Seed 1 \t  Instance : {} \t runtime w/ optimized smac params : {} \t default runtime {}".format(instance_list['NAME'][j],runtime,default_runtime))
'''

def smac_runner(train_df, test_df):
  """
  Pass in the default run results data frame. One for the test set, one for the train set 
  """
  l1 = Float('lambda1', bounds = (0,1))
  l2 = Float('lambda2', bounds = (0,1))
  l3 = Float('lambda3', bounds = (0,1))
  l4 = Float('lambda4', bounds = (0,1))

  configspace.add_hyperparameters([l1, l2, l3, l4])
  scenario = Scenario({
      "run_obj": "quality",
      "runcount-limit": 100,  
      "cs": configspace,
      "deterministic": False
  })
  smac = SMAC4AC(scenario=scenario, tae_runner=lambda x: evaluation(x, train_df))
  bfc = smac.optimize()
  return_Df = test_df.copy()
  return_Df['lambda1'] = bfc['lambda1']
  return_Df['lambda2'] = bfc['lambda2']
  return_Df['lambda3'] = bfc['lambda3']
  return_Df['lambda4'] = bfc['lambda4']
  return return_Df