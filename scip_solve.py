import os
import yaml
from utilities import  str_to_bool, read_cut_selector_param_file, get_filename, is_dir, is_file
import os
import numpy as np
import argparse
from pyscipopt import Model, quicksum, SCIP_RESULT, SCIP_PARAMSETTING, Branchrule, SCIP_PRESOLTIMING, SCIP_PROPTIMING
import parameters
from utilities import FixedAmountCutsel,RepeatSepaConshdlr
def build_scip_model(instance_path, node_lim, rand_seed, pre_solve, propagation, separators, heuristics,
                     aggressive_sep, dummy_branch_rule, time_limit=None, sol_path=None,
                     dir_cut_off=0.0, efficacy=1.0, int_support=0.1, obj_parallelism=0.1):
    """
    General function to construct a PySCIPOpt model.
    Args:
        instance_path: The path to the instance
        node_lim: The node limit
        rand_seed: The random seed
        pre_solve: Whether pre-solve should be enabled or disabled
        propagation: Whether propagators should be enabled or disabled
        separators: Whether separators should be enabled or disabled
        heuristics: Whether heuristics should be enabled or disabled
        aggressive_sep: Whether we want aggressive separators. Disabling separators overrides this.
        dummy_branch_rule: This is to cover a 'feature' of SCIP where by default strong branching is done and this can
                           give information about nodes beneath the node limit. So we add a branch-rule that can't.
        time_limit: The time_limit of the model
        sol_path: An optional path to a valid .sol file containing a primal solution to the instance
        dir_cut_off: The directed cut off weight that is applied to the custom cut-selector
        efficacy: The efficacy weight that is applied to the custom cut-selector
        int_support: The integer support weight that is applied to the custom cut-selector
        obj_parallelism: The objective parallelism weight that is applied to the custom cut-selector
    Returns:
        pyscipopt model
    """
    assert os.path.exists(instance_path)
    assert type(node_lim) == int and type(rand_seed) == int
    assert all([type(param) == bool for param in [pre_solve, propagation, separators, heuristics, aggressive_sep]])

    scip = Model()
    scip.setParam('limits/nodes', node_lim)
    scip.setParam('randomization/randomseedshift', rand_seed)
    if not pre_solve:
        # Printing the transformed MPS files keeps the fixed variables and this drastically changes the solve
        # functionality after reading in the model and re-solving. So set one round of pre-solve to remove these
        # Additionally, we want constraints to be the appropriate types and not just linear for additional separators
        scip.setParam('presolving/maxrounds', 1)
        # scip.setPresolve(SCIP_PARAMSETTING.OFF)
    if not propagation:
        scip.disablePropagation()
    if not separators:
        scip.setSeparating(SCIP_PARAMSETTING.OFF)
    elif aggressive_sep:
        # Set the number of rounds we want and the number of cuts per round that will be forced
        num_rounds = parameters.NUM_CUT_ROUNDS
        cuts_per_round = parameters.NUM_CUTS_PER_ROUND
        # Create a dummy constraint handler that forces the num_rounds amount of separation rounds
        constraint_handler = RepeatSepaConshdlr(scip, num_rounds)
        scip.includeConshdlr(constraint_handler, "RepeatSepa", "Forces a certain number of separation rounds",
                             sepapriority=-1, enfopriority=1, chckpriority=-1, sepafreq=-1, propfreq=-1,
                             eagerfreq=-1, maxprerounds=-1, delaysepa=False, delayprop=False, needscons=False,
                             presoltiming=SCIP_PRESOLTIMING.FAST, proptiming=SCIP_PROPTIMING.AFTERLPNODE)
        # Create a cut-selector with highest priority that forces cuts_per_rounds to be selected each round
        cut_selector = FixedAmountCutsel(num_cuts_per_round=cuts_per_round, dir_cutoff_dist_weight=dir_cut_off,
                                         efficacy_weight=efficacy, int_support_weight=int_support,
                                         obj_parallel_weight=obj_parallelism)
        scip.includeCutsel(cut_selector, 'FixedAmountCutSel', 'Tries to add the same number of cuts per round',
                           1000000)
        # Set the separator parameters
        scip.setParam('separating/maxstallroundsroot', num_rounds)
        scip = set_scip_separator_params(scip, num_rounds, -1, cuts_per_round, cuts_per_round, 0)
    else:
        # scip = set_scip_separator_params(scip, -1, -1, 5000, 100, 10)
        scip = set_scip_cut_selector_params(scip, dir_cut_off, efficacy, int_support, obj_parallelism)
    if not heuristics:
        scip.setHeuristics(SCIP_PARAMSETTING.OFF)
    if dummy_branch_rule:
        scip.setParam('branching/leastinf/priority', 10000000)
    if time_limit is not None:
        scip.setParam('limits/time', time_limit)

    # We do not want oribtope constraints as they're difficult to represent in the bipartite graph
    scip.setParam('misc/usesymmetry', 0)

    # read in the problem
    scip.readProblem(instance_path)

    if sol_path is not None:
        assert os.path.isfile(sol_path) and '.sol' in sol_path
        # Create the solution to add to SCIP
        sol = scip.readSolFile(sol_path)
        # Add the solution. This automatically frees the loaded solution
        scip.addSol(sol)

    return scip


def set_scip_cut_selector_params(scip, dir_cut_off, efficacy, int_support, obj_parallelism):
    """
    Sets the SCIP hybrid cut-selector parameter values in the weighted sum
    Args:
        scip: The PySCIPOpt model
        dir_cut_off: The coefficient of the directed cut-off distance
        efficacy: The coefficient of the efficacy
        int_support: The coefficient of the integer support
        obj_parallelism: The coefficient of the objective value parallelism (cosine similarity)
    Returns:
        The PySCIPOpt model with set parameters
    """
    scip.setParam("cutselection/hybrid/dircutoffdistweight", max(dir_cut_off, 0))
    scip.setParam("cutselection/hybrid/efficacyweight", max(efficacy, 0))
    scip.setParam("cutselection/hybrid/intsupportweight", max(int_support, 0))
    scip.setParam("cutselection/hybrid/objparalweight", max(obj_parallelism, 0))

    return scip


def set_scip_separator_params(scip, max_rounds_root=-1, max_rounds=-1, max_cuts_root=10000, max_cuts=10000,
                              frequency=10):
    """
    Function for setting the separator params in SCIP. It goes through all separators, enables them at all points
    in the solving process,
    Args:
        scip: The SCIP Model object
        max_rounds_root: The max number of separation rounds that can be performed at the root node
        max_rounds: The max number of separation rounds that can be performed at any non-root node
        max_cuts_root: The max number of cuts that can be added per round in the root node
        max_cuts: The max number of cuts that can be added per node at any non-root node
        frequency: The separators will be called each time the tree hits a new multiple of this depth
    Returns:
        The SCIP Model with all the appropriate parameters now set
    """

    assert type(max_cuts) == int and type(max_rounds) == int
    assert type(max_cuts_root) == int and type(max_rounds_root) == int

    # First for the aggregation heuristic separator
    scip.setParam('separating/aggregation/freq', frequency)
    scip.setParam('separating/aggregation/maxrounds', max_rounds)
    scip.setParam('separating/aggregation/maxroundsroot', max_rounds_root)
    scip.setParam('separating/aggregation/maxsepacuts', 10000)
    scip.setParam('separating/aggregation/maxsepacutsroot', 10000)

    # Now the Chvatal-Gomory w/ MIP separator
    # scip.setParam('separating/cgmip/freq', frequency)
    # scip.setParam('separating/cgmip/maxrounds', max_rounds)
    # scip.setParam('separating/cgmip/maxroundsroot', max_rounds_root)

    # The clique separator
    scip.setParam('separating/clique/freq', frequency)
    scip.setParam('separating/clique/maxsepacuts', 10000)

    # The close-cuts separator
    scip.setParam('separating/closecuts/freq', frequency)

    # The CMIR separator
    scip.setParam('separating/cmir/freq', frequency)

    # The Convex Projection separator
    scip.setParam('separating/convexproj/freq', frequency)
    scip.setParam('separating/convexproj/maxdepth', -1)

    # The disjunctive cut separator
    scip.setParam('separating/disjunctive/freq', frequency)
    scip.setParam('separating/disjunctive/maxrounds', max_rounds)
    scip.setParam('separating/disjunctive/maxroundsroot', max_rounds_root)
    scip.setParam('separating/disjunctive/maxinvcuts', 10000)
    scip.setParam('separating/disjunctive/maxinvcutsroot', 10000)
    scip.setParam('separating/disjunctive/maxdepth', -1)

    # The separator for edge-concave function
    scip.setParam('separating/eccuts/freq', frequency)
    scip.setParam('separating/eccuts/maxrounds', max_rounds)
    scip.setParam('separating/eccuts/maxroundsroot', max_rounds_root)
    scip.setParam('separating/eccuts/maxsepacuts', 10000)
    scip.setParam('separating/eccuts/maxsepacutsroot', 10000)
    scip.setParam('separating/eccuts/maxdepth', -1)

    # The flow cover cut separator
    scip.setParam('separating/flowcover/freq', frequency)

    # The gauge separator
    scip.setParam('separating/gauge/freq', frequency)

    # Gomory MIR cuts
    scip.setParam('separating/gomory/freq', frequency)
    scip.setParam('separating/gomory/maxrounds', max_rounds)
    scip.setParam('separating/gomory/maxroundsroot', max_rounds_root)
    scip.setParam('separating/gomory/maxsepacuts', 10000)
    scip.setParam('separating/gomory/maxsepacutsroot', 10000)

    # The implied bounds separator
    scip.setParam('separating/impliedbounds/freq', frequency)

    # The integer objective value separator
    scip.setParam('separating/intobj/freq', frequency)

    # The knapsack cover separator
    scip.setParam('separating/knapsackcover/freq', frequency)

    # The multi-commodity-flow network cut separator
    scip.setParam('separating/mcf/freq', frequency)
    scip.setParam('separating/mcf/maxsepacuts', 10000)
    scip.setParam('separating/mcf/maxsepacutsroot', 10000)

    # The odd cycle separator
    scip.setParam('separating/oddcycle/freq', frequency)
    scip.setParam('separating/oddcycle/maxrounds', max_rounds)
    scip.setParam('separating/oddcycle/maxroundsroot', max_rounds_root)
    scip.setParam('separating/oddcycle/maxsepacuts', 10000)
    scip.setParam('separating/oddcycle/maxsepacutsroot', 10000)

    # The rapid learning separator
    scip.setParam('separating/rapidlearning/freq', frequency)

    # The strong CG separator
    scip.setParam('separating/strongcg/freq', frequency)

    # The zero-half separator
    scip.setParam('separating/zerohalf/freq', frequency)
    scip.setParam('separating/zerohalf/maxcutcands', 100000)
    scip.setParam('separating/zerohalf/maxrounds', max_rounds)
    scip.setParam('separating/zerohalf/maxroundsroot', max_rounds_root)
    scip.setParam('separating/zerohalf/maxsepacuts', 10000)
    scip.setParam('separating/zerohalf/maxsepacutsroot', 10000)

    # The rlt separator
    scip.setParam('separating/rlt/freq', frequency)
    scip.setParam('separating/rlt/maxncuts', 10000)
    scip.setParam('separating/rlt/maxrounds', max_rounds)
    scip.setParam('separating/rlt/maxroundsroot', max_rounds_root)

    # Now the general cut and round parameters
    scip.setParam("separating/maxroundsroot", max_rounds_root)
    scip.setParam("separating/maxstallroundsroot", max_rounds_root)
    scip.setParam("separating/maxcutsroot", max_cuts_root)

    scip.setParam("separating/maxrounds", max_rounds)
    scip.setParam("separating/maxstallrounds", 1)
    scip.setParam("separating/maxcuts", max_cuts)

    return scip

def run_instance(temp_dir, instance_path, instance, rand_seed, sample_i, time_limit, root, print_sol, print_stats):
    """
    The call to solve a single instance.The function loads the correct cut-selector parameters and then solves the appropriate SCIP instance.
    Args:
        temp_dir: The directory in which all temporary files per batch will be dumped then deleted (e.g. cut-sel params)
        instance_path: The path to the MIP .mps instance
        instance: The instance base name of the MIP file
        rand_seed: The random seed which will be used to shift all SCIP randomisation
        sample_i: The sample index so we can load the sampled cut-sel param YAML file
        time_limit: The time limit, if it exists for our SCIP instance. Negative time_limit means None
        root: A boolean for whether we should restrict our solve to the root node or not
        print_sol: Whether the .sol file from the run should be printed or not
        print_stats: Whether the .stats file from the run should be printed or not
    Returns:
        Nothing. All results from this run should be output to a file in temp_dir.
        The results should contain all information about the run, (e.g. cut-sel params, solve_time, dual_bound etc)
    """

    # Load the cut-selector params
    dir_cut_off, efficacy, int_support, obj_parallelism = read_cut_selector_param_file(temp_dir, instance, rand_seed,
                                                                                       sample_i)

    # Print out the cut-sel param values to the slurm .out file
    print('DIR: {}, EFF: {}, INT: {}, OBJ: {}'.format(dir_cut_off, efficacy, int_support, obj_parallelism), flush=True)
    
    # Build the initial SCIP model for the instance
    time_limit = None if time_limit < 0 else time_limit
    node_lim = 1 if root else -1
    propagation = False if root else True
    heuristics = False if root else True
    aggressive = True if root else False
    dummy_branch = True if root else False

    # Check is a solution file exists. This solution file should be next to the instance file
    if os.path.isfile(os.path.splitext(instance_path)[0] + '.sol'):
        sol_file = os.path.splitext(instance_path)[0] + '.sol'
    else:
        sol_file = None

    # Build the actual SCIP model from the information now
    scip = build_scip_model(instance_path, node_lim, rand_seed, False, propagation, True, heuristics, aggressive,
                            dummy_branch, time_limit=time_limit, sol_path=sol_file,
                            dir_cut_off=dir_cut_off, efficacy=efficacy, int_support=int_support,
                            obj_parallelism=obj_parallelism)

    # Solve the SCIP model and extract all solve information
    solve_model_and_extract_solve_info(scip, dir_cut_off, efficacy, int_support, obj_parallelism, rand_seed, sample_i,
                                       instance, temp_dir, root=root, print_sol=print_sol, print_stats=print_stats)

    # Free the SCIP instance
    scip.freeProb()

    return


def solve_model_and_extract_solve_info(scip, dir_cut_off, efficacy, int_support, obj_parallelism, rand_seed, sample_i,
                                       instance, temp_dir, root=True, print_sol=False, print_stats=False):
    """
    Solves the given SCIP model and after solving creates a YAML file with all potentially interesting
    solve information. This information will later be read and used to update the neural_network parameters
    Args:
        scip: The PySCIPOpt model that we want to solve
        dir_cut_off: The coefficient for the directed cut-off distance
        efficacy: The coefficient for the efficacy
        int_support: The coefficient for the integer support
        obj_parallelism: The coefficient for the objective function parallelism (see also the cosine similarity)
        rand_seed: The random seed used in the scip parameter settings
        sample_i: The sample index used to locate the correct cut-sel param values used
        instance: The instance base name of our problem
        temp_dir: The temporary file directory where we place all files that are batch-specific (e.g. cut-sel params)
        root: A kwarg that informs if the solve is restricted to the root node. Used for naming the yml file
        print_sol: A kwarg that informs if the .sol file from the run should be saved to a file
        print_stats: A kwarg that informs if the .stats file from the run should be saved to a file
    Returns:
    """

    # Solve the MIP instance. All parameters should be pre-set
    scip.optimize()

    # Initialise the dictionary that will store our solve information
    data = {}

    # Get the solve_time
    data['solve_time'] = scip.getSolvingTime()
    # Get the number of cuts applied
    data['num_cuts'] = scip.getNCutsApplied()
    # Get the number of nodes in our branch and bound tree
    data['num_nodes'] = scip.getNNodes()
    # Get the best primal solution if available
    data['primal_bound'] = scip.getObjVal() if len(scip.getSols()) > 0 else 1e+20
    # Get the gap provided a primal solution exists
    data['gap'] = scip.getGap() if len(scip.getSols()) > 0 else 1e+20
    # Get the best dual bound
    data['dual_bound'] = scip.getDualbound()
    # Get the number of LP iterations
    data['num_lp_iterations'] = scip.getNLPIterations()
    # Get the status of the solve
    data['status'] = scip.getStatus()

    # Save the sol file if we've been asked to
    if len(scip.getSols()) > 0 and print_sol:
        sol = scip.getBestSol()
        sol_file = get_filename(temp_dir, instance, rand_seed, trans=True, root=False, sample_i=None, ext='sol')
       
        try:
            scip.writeSol(sol, sol_file)
        except:pass

    # Get the percentage of integer variables with fractional values. This includes implicit integer variables
    scip_vars = scip.getVars()
    non_cont_vars = [var for var in scip_vars if var.vtype() != 'CONTINUOUS']
    assert len(non_cont_vars) > 0
    if root:
        cont_valued_non_cont_vars = [var for var in non_cont_vars if not scip.isZero(scip.frac(var.getLPSol()))]
    else:
        assert len(scip.getSols()) > 0
        scip_sol = scip.getBestSol()
        cont_valued_non_cont_vars = [var for var in non_cont_vars if not scip.isZero(scip.frac(scip_sol[var]))]
    data['solution_fractionality'] = len(cont_valued_non_cont_vars) / len(non_cont_vars)

    # Add the cut-selector parameters
    data['dir_cut_off'] = dir_cut_off
    data['efficacy'] = efficacy
    data['int_support'] = int_support
    data['obj_parallelism'] = obj_parallelism

    # Get the primal dual integral. This is not really needed for root solves, but might be important to have
    # It is only accessible through the solver statistics. TODO: Write a wrapper function for this
    stat_file = get_filename(temp_dir, instance, rand_seed, trans=True, root=root, sample_i=sample_i, ext='stats')
    try:
        scip.writeStatistics(stat_file)
    except:
        pass
    with open(stat_file) as s:
        stats = s.readlines()
    # TODO: Make this safer to access.
    assert 'primal-dual' in stats[-3]
    data['primal_dual_integral'] = float(stats[-3].split(':')[1].split('     ')[1])
    if not print_stats:
        os.remove(stat_file)

    # Dump the yml file containing all of our solve info into the right place
    yml_file = get_filename(temp_dir, instance, rand_seed=rand_seed, trans=True, root=root, sample_i=sample_i,
                            ext='yml')
    with open(yml_file, 'w') as s:
        yaml.dump(data, s)

    return