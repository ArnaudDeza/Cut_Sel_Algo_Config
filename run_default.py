'''


'''


import os
from presolve import presolve_instance
import parameters
import numpy as np
import argparse

from utilities import save_default_cut_selector_param_npy_file,get_filename
from scip_solve import run_instance
def run_pre_solve_instances(instances, instance_paths, sol_paths, rand_seeds, transformed_problem_dir, outfile_dir,
                        temp_dir, use_miplib_sols):
    """
    Function for pre-solving instances 
    Args:
        instances (list): List of instances names
        instance_paths (list): List of instance paths (indices match instances)
        sol_paths (list): List of solution file paths (indices match instances)
        rand_seeds (list): List containing the random seeds we'll use
        transformed_problem_dir (dir): Directory where we will throw our pre-solved mps instance file
        outfile_dir (dir): The directory where we throw our slurm files
        temp_dir (dir): The directory where we throw our temporary files used just for this function
        use_miplib_sols (bool): Whether we are going to use the MIPLIB sols or not.
    Returns:
        Produces the pre-solved mps instances 
    """
    outfile_dir = os.path.join(outfile_dir, 'pre_solve')
    os.mkdir(outfile_dir)

    presolved_instances_paths = []

    for i, instance in enumerate(instances):
        for rand_seed in rand_seeds:
            presolve_file_name = presolve_instance(transformed_problem_dir, instance_paths[i], sol_paths[i], instance,
                                                rand_seed, use_miplib_sols)

            presolved_instances_paths.append(presolve_file_name)
    #print(len(presolved_instances_paths))
    return 

def save_default_cut_selector_param_npy_file(temp_dir, instance, rand_seed):
    """
    Creates a npy file for the default cut-selector parameter values
    Args:
        temp_dir (dir): Directory where we will dump the npy file
        instance (str): The name of the instance
        rand_seed (int): The random seed of the solve
    Returns: Nothing, just creates a file
    """

    cut_selector_params = np.array([0.0, 1.0, 0.1, 0.1])
    file_name = get_filename(temp_dir, instance, rand_seed, trans=True, root=False, sample_i=0, ext='npy')
    np.save(file_name, cut_selector_params)

    return

def run_solve_and_get_solution_files(data_dir, temp_dir, outfile_dir, instances, rand_seeds):
    """
    The function for generating calls to solve_instance_seed_noise.py that will solve a SCIP
    instance that is only restricted by run-time. This run will be used to see if a feasible solution for the
    instance can be found.
    Args:
        data_dir (dir): Directory containing our pre-solved mps instances where we now will generate .sol files
        temp_dir (dir): Directory containing this function specific files
        outfile_dir (dir): Directory where our slurm log files will be output to
        instances (list): The list of instances we are interested in
        rand_seeds (list): The list of random seeds which we are interested in
    Returns:
        Produces the .sol files that we use as reference primal solutions for all later experiments
    """
    outfile_dir = os.path.join(outfile_dir, 'get_sol_files')
    os.mkdir(outfile_dir)
    for instance in instances:
        for rand_seed in rand_seeds:
            save_default_cut_selector_param_npy_file(temp_dir, instance, rand_seed)
            mps_file = get_filename(data_dir, instance, rand_seed, trans=True, root=False, sample_i=None, ext='mps')
            run_instance(temp_dir, mps_file,instance, rand_seed, 0, parameters.SOL_FIND_TIME_LIMIT, root = False, print_sol = True, print_stats = False)
       
    # Now move the .sol files from the instances that worked into the directory containing the pre-solved mps files
    for instance in instances:
        for rand_seed in rand_seeds:
            sol_path = get_filename(temp_dir, instance, rand_seed, trans=True, root=False, sample_i=None, ext='sol')
            new_sol_path = get_filename(data_dir, instance, rand_seed, trans=True, root=False, sample_i=None, ext='sol')
            assert os.path.isfile(sol_path)
            assert not os.path.isfile(new_sol_path)
            os.rename(sol_path, new_sol_path)



def run_scip_and_get_yml_and_log_files(data_dir, temp_dir, outfile_dir, instances, rand_seeds, root=True):
    """
    It solves a run for all instance and random seed combinations. It then creates a YML file on the statistics,
    a .stats file containing the SCIP output of the statistics, and a .log file containing the log output of the run.
    It does this for all combinations, and then filters out instances who failed on at-least one random seed,
    outputting the reason they failed.
    Args:
        data_dir (dir): Directory containing the mps files, where we will dump all final files
        temp_dir (dir): Directory where we dump all temporary files
        outfile_dir (dir): Directory where we dump all slurm .out files. These can later become .log files
        instances (list): List of instances we're interested in
        rand_seeds (list): List of random seeds we're interested in
        root (bool): Whether we restrict our solves to the root node or not
    Returns: The list of instances which solved over all random seeds successfully. Creates files of all relevant info
    """

    # Set the time limit for the run. This simply depends on if the root node or not
    time_limit = parameters.ROOT_SOLVE_TIME_LIMIT if root else parameters.FULL_SOLVE_TIME_LIMIT

    # Change the outfile directory for this set of runs
    outfile_sub_dir_name = 'root_solve' if root else 'full_solve'
    outfile_dir = os.path.join(outfile_dir, outfile_sub_dir_name)
    os.mkdir(outfile_dir)
    # Start all the individual jobs that solve and instance and a random seed
    for instance in instances:
        for rand_seed in rand_seeds:
            save_default_cut_selector_param_npy_file(temp_dir, instance, rand_seed)
            mps_file = get_filename(data_dir, instance, rand_seed, trans=True, root=False, sample_i=None, ext='mps')
            try:
              run_instance(temp_dir, mps_file, instance, rand_seed,0,time_limit, root, False, True)
            except:pass
    # Now move the files we created for the non-problematic instances
    for instance in instances:
          for rand_seed in rand_seeds:
              try:
                # First do the YAML file
                yml_file = get_filename(temp_dir, instance, rand_seed, trans=True, root=root, sample_i=0, ext='yml')
                new_yml_file = get_filename(data_dir, instance, rand_seed, trans=True, root=root, sample_i=None, ext='yml')
                assert os.path.isfile(yml_file) and not os.path.isfile(new_yml_file)
                os.rename(yml_file, new_yml_file)

                ''' # Now do the log file
                out_file = get_slurm_output_file(outfile_dir, instance, rand_seed)
                new_out_file = get_filename(data_dir, instance, rand_seed, trans=True, root=root, sample_i=None, ext='log')
                assert os.path.isfile(out_file) and not os.path.isfile(new_out_file)
                os.rename(out_file, new_out_file)'''

                # Now do the stats file
                stats_path = get_filename(temp_dir, instance, rand_seed, trans=True, root=root, sample_i=0, ext='stats')
                new_path = get_filename(data_dir, instance, rand_seed, trans=True, root=root, sample_i=None, ext='stats')
                assert os.path.isfile(stats_path) and not os.path.isfile(new_path)
                os.rename(stats_path, new_path)
              except:
                  pass
    
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('problem_dir', type=str)
    parser.add_argument('solution_dir', type=str)
    parser.add_argument('transformed_problem_dir', type=str)
    parser.add_argument('temp_dir', type=str)
    parser.add_argument('outfile_dir', type=str)
    parser.add_argument('num_rand_seeds', type=int)
    args = parser.parse_args()


    # Initialise a list of instances
    instance_names = []
    instance_file_paths = []
    sol_file_paths = []

    sol_files = os.listdir(args.solution_dir)
    for file in os.listdir(args.problem_dir):
        # Extract the instance
        assert file.endswith('.mps.gz'), 'File {} does not end with .mps.gz'.format(file)
        instance_name = file.split('.')[0]
        instance_names.append(instance_name)
        instance_file_paths.append(os.path.join(args.problem_dir, file))
        sol_file = instance_name + '.sol.gz'
        assert sol_file in sol_files, 'sol_file {} not found'.format(sol_file)
        sol_file_paths.append(os.path.join(args.solution_dir, sol_file))



    # Initialise the random seeds
    random_seeds = [random_seed for random_seed in range(1, args.num_rand_seeds + 1)]

    ''' # First we pre-solve the instances and filter those which take too long or take too much memory
    print('Pre-Solving instances', flush=True)
    run_pre_solve_instances(instance_names, instance_file_paths, sol_file_paths, random_seeds,
                                         args.transformed_problem_dir, args.outfile_dir, args.temp_dir,
                                         parameters.USE_MIPLIB_SOLUTIONS)

    if not parameters.USE_MIPLIB_SOLUTIONS:
        # We then filter those instances which cannot produce primal solutions
        print('Finding primal solutions to pre-solved instances', flush=True)
        run_solve_and_get_solution_files(args.transformed_problem_dir, args.temp_dir,
                                                           args.outfile_dir, instance_names, random_seeds)'''
        

    # We now produce YML files containing solve information for our root-node restricted solves.
    print('Producing root-node restricted solve statistics in YML files', flush=True)
    run_scip_and_get_yml_and_log_files(args.transformed_problem_dir, args.temp_dir, args.outfile_dir,
                                                          instance_names, random_seeds, root = True)
 


    
            


