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


def simple_smac_runner(train_df, test_df,num_config_calls):
  """
  Pass in the default run results data frame. One for the test set, one for the train set 
  """
  l1 = Float('lambda1', bounds = (0,1),default =0)
  l2 = Float('lambda2', bounds = (0,1),default =1)
  l3 = Float('lambda3', bounds = (0,1),default =0.1)
  l4 = Float('lambda4', bounds = (0,1),default =0.1)
  configspace = ConfigurationSpace()

  configspace.add_hyperparameters([l1, l2, l3, l4])
  scenario = Scenario({
      "run_obj": "quality",
      "runcount-limit": num_config_calls,  
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

def solve_instance(instance_name, allowed_runtime, rand_seed,lambda1, lambda2, lambda3, lambda4,root):

  # step 1: get mip based of instance_name
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
  data = run_instance(temp_dir, mps_file, instance_name, rand_seed,0,time_limit, root, False, True)
 
  return data['solve_time'],data['gap']


def evaluation(params, df, n_samples = 5, gap_tolerance = 0.0002, epsilon = 1e-8):
  n_samples = min(df.shape[0], n_samples)
  lambda1 = params['lambda1']
  lambda2 = params['lambda2']
  lambda3 = params['lambda3']
  lambda4 = params['lambda4']
  df_sample = df.sample(n_samples)
  
  scores = []
  for j in range(n_samples):
    for rand_seed in [1,2,3]:
      runtime, mip_gap = solve_instance(df_sample['NAME'].tolist()[j], 20,rand_seed,lambda1, lambda2, lambda3, lambda4,root = False)
    
      if mip_gap < gap_tolerance: ## If Solved, Score the runtime
              imp = float((df_sample['SOLUTION TIME'].tolist()[j] - runtime)) / (np.abs(df_sample['SOLUTION TIME'].tolist()[j]) + 1e-8)
              imp = -1*imp
              scores.append(imp)
              print("\n name {} time imp {} seed {} smac time {} def time {}".format(df_sample['NAME'].tolist()[j],imp,rand_seed,df_sample['SOLUTION TIME'].tolist()[j],runtime))
      else:  ## If not solved, score the MIP relative imrovment
              imp = float((df_sample['gap'].tolist()[j] - mip_gap)) / (np.abs(df_sample['gap'].tolist()[j]) + 1e-8)
              imp = -1*imp
              scores.append(imp)
              print("\n name {} gap imp {} seed {} smac time {} def time {}".format(df_sample['NAME'].tolist()[j],imp,rand_seed,df_sample['gap'].tolist()[j],mip_gap))
  print("Done 1 SMAC evaluation")
  return np.mean(scores)



def hetero_runner(file_name,num_splits,num_config_calls):
  """
  Pass in the file that contains the MIPLIB default run results.
  Works for both 
  """
  df = pd.read_csv(file_name)
  shuffled = df.sample(frac=1)
  splits = np.array_split(shuffled, num_splits)
  results = []

  for i in range(num_splits):
    test = splits[i].copy()
    train_list = [splits[j] for j in range(num_splits) if j != i]
    train = pd.concat(train_list)
    # return best found configuration
    results.append(simple_smac_runner(train, test,num_config_calls = num_config_calls))
  return pd.concat(results)

def partitioned_runner(file_name,num_splits,num_config_calls):
  """
  Pass in the file that contains the MIPLIB default run results.
  Works for both 
  """
  df = pd.read_csv(file_name)
  shuffled = df.sample(frac=1)
  splits = np.array_split(shuffled, num_splits)
  results = []

  for i in range(num_splits):
    test = splits[i].copy()
    train_list = [splits[j] for j in range(num_splits) if j != i]
    train = pd.concat(train_list)
    # return best found configuration
    results.append(simple_smac_runner(train, test,num_config_calls = num_config_calls))
  return pd.concat(results)



# folder where yml files will be store
temp_dir = "/home/arnaud/Documents/mie1666/new_ACS/smac_runs/experiment"
# folder where transformed problems defult shoud be
data_dir = "/home/arnaud/Documents/mie1666/new_ACS/transformed_problems_default"
# output files will remain empty
outfile_dir = "/home/arnaud/Documents/mie1666/new_ACS/smac_runs/output_files"
try: os.mkdir(outfile_dir)
except:pass
try: os.mkdir(temp_dir)
except:pass

df = hetero_runner("/home/arnaud/Documents/mie1666/new_ACS/df_default.csv",num_splits = 2,num_config_calls = 200)
df.to_pickle("/home/arnaud/Documents/mie1666/new_ACS/smac_runs/smac_run.pkl")
df.to_csv("/home/arnaud/Documents/mie1666/new_ACS/smac_runs/smac_run.csv")

