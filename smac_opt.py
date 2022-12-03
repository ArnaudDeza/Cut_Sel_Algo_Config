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


temp_dir = "/home/arnaud/Documents/mie1666/MIE-1666-Project/smac_folder/experiments"
data_dir = "/home/arnaud/Documents/mie1666/MIE-1666-Project/smac_folder/transformed_problems"
outfile_dir = "/home/arnaud/Documents/mie1666/MIE-1666-Project/smac_folder/output_files"
try: os.mkdir(temp_dir)
except:pass
try: os.mkdir(outfile_dir)
except:pass

def solve_instance(instance_name, allowed_runtime, lambda1, lambda2, lambda3, lambda4,root):

  # step 1: get mip based of instance_name
  rand_seed = 1
  time_limit = allowed_runtime

  # Change the outfile directory for this set of runs
  outfile_sub_dir_name = 'root_solve' if root else 'full_solve'
  outfile_dir_ = os.path.join(outfile_dir, outfile_sub_dir_name)
  try:
        os.mkdir(outfile_dir_)
  except:
        pass
# step 2: set scip params using lambda's and allowed_runtime
  save_default_cut_selector_param_npy_file_SMAC(temp_dir, instance_name, rand_seed,lambda1, lambda2, lambda3, lambda4)
  mps_file = get_filename(data_dir, instance_name, rand_seed, trans=True, root=False, sample_i=None, ext='mps')
  
  
  data =run_slurm_job_with_random_seed(temp_dir, outfile_dir_, mps_file, instance_name, rand_seed,
                                                time_limit, root, False, True, exclusive=True)
  # step 3: solve and return mip gap and runtime          
  return data['solve_time'],data['gap']
  


def evaluation(params, df, n_samples = 5, gap_tolerance = 0.0002, epsilon = 0.00002):
  n_samples = min(df.shape[0], n_samples)
  lambda1 = params['lambda1']
  lambda2 = params['lambda2']
  lambda3 = params['lambda3']
  lambda4 = params['lambda4']
  instance_list = df.sample(n_samples).to_dict()
 
  scores = []
  for j in range(len(instance_list['NAME'])):

    runtime, mip_gap = solve_instance(instance_list['NAME'][j], instance_list['SOLUTION TIME'][j],lambda1, lambda2, lambda3, lambda4,root = False)
    if mip_gap < gap_tolerance: ## If Solved, Score the runtime
      scores.append(runtime / (instance_list['SOLUTION TIME'][j] + epsilon))
      #t_dif = runtime - instance['default_runtime']
      #score = 1 / 1 + (np.exp(-1 * t_dif))
    else: ## If not solved, score the MIP Gap
      scores.append(mip_gap / (instance_list['gap'][j] + epsilon))
      #g_dif = mip_gap - instance['default_mip_gap']
      #score = 1 / 1 + (np.exp(-1 * t_dif))
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
  print(bfc)
  exit()
  return_Df = test_df.copy()
  return_Df['lambda1'] = bfc['lambda1']
  return_Df['lambda2'] = bfc['lambda2']
  return_Df['lambda3'] = bfc['lambda3']
  return_Df['lambda4'] = bfc['lambda4']
  return return_Df

'''df = pd.read_csv("/home/arnaud/Documents/mie1666/MIE-1666-Project/data/table_page_21_2_instances.csv")


msk = np.random.rand(len(df)) < 0.8
train_df = df[msk]
test_df = df[~msk]
simple_smac_runner(df, df)'''

def convert_yml_to_csv(yml_folder):
    default_yml_paths = list(Path(yml_folder).rglob("*.yml" ))
    rows = []
    for yml in default_yml_paths:
        name = str(yml).split("experiment/")[1].split("__trans")[0]
        seed = str(yml).split("seed__")[1].split("__sample")[0]
        if "root" not in str(yml) and seed == "1":
            #print(str(yml))
            with open(str(yml)) as f:
                        data = yaml.load(f, Loader=SafeLoader)
                        #print(data)
            rows.append([name,data['gap'],data['solve_time'],data['status']])
    
    df = pd.DataFrame(rows,columns =['NAME','gap','SOLUTION TIME', 'status'])
    return df
df = convert_yml_to_csv("/home/arnaud/Documents/mie1666/MIE-1666-Project/experiment")
##simple_smac_runner(df, df)

bfc = {
  'lambda1': 0.7795422929199476,
  'lambda2': 0.8058487605433546,
  'lambda3': 0.4019380365959977,
  'lambda4': 0.12349715654529236,
}
root = False
lambda1 = bfc['lambda1']
lambda2 = bfc['lambda2']
lambda3 = bfc['lambda3']
lambda4 = bfc['lambda4']
instance_list = df.sample(2).to_dict()

scores = []
for j in range(len(instance_list['NAME'])):

    runtime, mip_gap = solve_instance(instance_list['NAME'][j], instance_list['SOLUTION TIME'][j],lambda1, lambda2, lambda3, lambda4,root = False)
    if instance_list['NAME'][j] == "23588":
        default_runtime = 4.979511
    if instance_list['NAME'][j] == "22433":
        default_runtime = 1.9013849999999999
    print("Seed 1 \t  Instance : {} \t runtime w/ optimized smac params : {} \t default runtime {}".format(instance_list['NAME'][j],runtime,default_runtime))
exit()

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