import os
import numpy as np
import subprocess
import shutil
import logging
import argparse
from pyscipopt import Model, quicksum, SCIP_RESULT, SCIP_PARAMSETTING, Branchrule, SCIP_PRESOLTIMING, SCIP_PROPTIMING
from ConstraintHandler.ConstraintHandler import RepeatSepaConshdlr
from CutSelectors.FixedAmountCutsel import FixedAmountCutsel
import parameters
from pyscipopt import Model, quicksum, SCIP_RESULT, SCIP_PARAMSETTING
from pyscipopt.scip import Cutsel
import random
from pyscipopt import Model, Conshdlr, SCIP_RESULT, SCIP_PRESOLTIMING, SCIP_PROPTIMING
"""
This is a dummy constraint handler that is used to force the number of separation rounds that we want.
It checks if the number of separation rounds has been hit, and if it hasn't then it sends the solver back to solve
the same node.
To use this you must set the enforce priority to be a positive value so it is called before branching.
You must also set need constraints to be False otherwise it will not be called.
"""


class RepeatSepaConshdlr(Conshdlr):

    def __init__(self, model, max_separation_rounds):
        super().__init__()
        self.model = model
        self.max_separation_rounds = max_separation_rounds

    # fundamental callbacks
    def consenfolp(self, constraints, nusefulconss, solinfeasible):

        if self.model.getNSepaRounds() <= self.max_separation_rounds and self.model.getNNodes() == 1:
            return {'result': SCIP_RESULT.SOLVELP}
        else:
            return {'result': SCIP_RESULT.FEASIBLE}

    def conscheck(self, constraints, solution, checkintegrality, checklprows, printreason, completely):
        return {"result": SCIP_RESULT.FEASIBLE}

    def conslock(self, constraint, locktype, nlockspos, nlocksneg):
        return
class FixedAmountCutsel(Cutsel):

    def __init__(self, num_cuts_per_round=20, min_orthogonality_root=0.9,
                 min_orthogonality=0.9, dir_cutoff_dist_weight=0.0, efficacy_weight=1.0, int_support_weight=0.1,
                 obj_parallel_weight=0.1):
        super().__init__()
        self.num_cuts_per_round = num_cuts_per_round
        self.min_orthogonality_root = min_orthogonality_root
        self.min_orthogonality = min_orthogonality
        self.dir_cutoff_dist_weight = dir_cutoff_dist_weight
        self.int_support_weight = int_support_weight
        self.obj_parallel_weight = obj_parallel_weight
        self.efficacy_weight = efficacy_weight
        random.seed(42)

    def cutselselect(self, cuts, forcedcuts, root, maxnselectedcuts):
        """
        This is the main function used to select cuts. It must be named cutselselect and is called by default when
        SCIP performs cut selection if the associated cut selector has been included (assuming no cutsel with higher
        priority was called successfully before). This function aims to add self.num_cuts_per_round many cuts to
        the LP per round, prioritising the highest ranked cuts. It adds the highest ranked cuts, filtering by
        parallelism. In the case when not enough cuts are added and all the remaining cuts are too parallel,
        we simply add those with the highest score.
        @param cuts: These are the optional cuts we get to select from
        @type cuts: List of pyscipopt rows
        @param forcedcuts: These are the cuts that must be added
        @type forcedcuts: List of pyscipopt rows
        @param root: Boolean for whether we're at the root node or not
        @type root: Bool
        @param maxnselectedcuts: Maximum number of selected cuts
        @type maxnselectedcuts: int
        @return: Dictionary containing the keys 'cuts', 'nselectedcuts', result'. Warning: Cuts can only be reordered!
        @rtype: dict
        """
        # Initialise number of selected cuts and number of cuts that are still valid candidates
        n_cuts = len(cuts)
        nselectedcuts = 0

        # Get the number of cuts that we will select this round.
        num_cuts_to_select = min(maxnselectedcuts, max(self.num_cuts_per_round - len(forcedcuts), 0), n_cuts)

        # Initialises parallel thresholds. Any cut with 'good' score can be at most good_max_parallel to a previous cut,
        # while normal cuts can be at most max_parallel. (max_parallel >= good_max_parallel)
        if root:
            max_parallel = 1 - self.min_orthogonality_root
            good_max_parallel = max(0.5, max_parallel)
        else:
            max_parallel = 1 - self.min_orthogonality
            good_max_parallel = max(0.5, max_parallel)

        # Generate the scores of each cut and thereby the maximum score
        max_forced_score, forced_scores = self.scoring(forcedcuts)
        max_non_forced_score, scores = self.scoring(cuts)

        good_score = max(max_forced_score, max_non_forced_score)

        # This filters out all cuts in cuts who are parallel to a forcedcut.
        for forced_cut in forcedcuts:
            n_cuts, cuts, scores = self.filter_with_parallelism(n_cuts, nselectedcuts, forced_cut, cuts,
                                                                scores, max_parallel, good_max_parallel, good_score)

        if maxnselectedcuts > 0 and num_cuts_to_select > 0:
            while n_cuts > 0:
                # Break the loop if we have selected the required amount of cuts
                if nselectedcuts == num_cuts_to_select:
                    break
                # Re-sorts cuts and scores by putting the best cut at the beginning
                cuts, scores = self.select_best_cut(n_cuts, nselectedcuts, cuts, scores)
                nselectedcuts += 1
                n_cuts -= 1
                n_cuts, cuts, scores = self.filter_with_parallelism(n_cuts, nselectedcuts, cuts[nselectedcuts -1], cuts,
                                                                    scores, max_parallel, good_max_parallel,
                                                                    good_score)

            # So far we have done the algorithm from the default method. We will now enforce choosing the highest
            # scored cuts from those that were previously removed for being too parallel.
            # Reset the n_cuts counter
            n_cuts = len(cuts) - nselectedcuts
            for remaining_cut_i in range(nselectedcuts, num_cuts_to_select):
                cuts, scores = self.select_best_cut(n_cuts, nselectedcuts, cuts, scores)
                nselectedcuts += 1
                n_cuts -= 1

        return {'cuts': cuts, 'nselectedcuts': nselectedcuts,
                'result': SCIP_RESULT.SUCCESS}

    def scoring(self, cuts):
        """
        Scores each cut in cuts. The current rule is a weighted sum combination of the efficacy,
        directed cutoff distance, integer support, and objective function parallelism.
        @param cuts: The list of cuts we want to find scores for
        @type cuts: List of pyscipopt rows
        @return: The max score over all cuts in cuts as well as the individual scores
        @rtype: Float and List of floats
        """
        # initialise the scoring of each cut as well as the max_score
        scores = [0] * len(cuts)
        max_score = 0.0

        # We require this check as getBestSol() may return the lp solution, which is not a valid primal solution
        sol = self.model.getBestSol() if self.model.getNSols() > 0 else None

        # Separate into two cases depending on whether the directed cutoff distance contributes to the score
        if sol is not None:
            for i in range(len(cuts)):
                int_support = self.int_support_weight * \
                              self.model.getRowNumIntCols(cuts[i]) / cuts[i].getNNonz()
                obj_parallel = self.obj_parallel_weight * self.model.getRowObjParallelism(cuts[i])
                efficacy = self.model.getCutEfficacy(cuts[i])
                if cuts[i].isLocal():
                    score = self.dir_cutoff_dist_weight * efficacy
                else:
                    score = self.model.getCutLPSolCutoffDistance(cuts[i], sol)
                    score = self.dir_cutoff_dist_weight * max(score, efficacy)
                efficacy *= self.efficacy_weight
                score += obj_parallel + int_support + efficacy
                score += 1e-4 if cuts[i].isInGlobalCutpool() else 0
                score += random.uniform(0, 1e-6)
                max_score = max(max_score, score)
                scores[i] = score
        else:
            for i in range(len(cuts)):
                int_support = self.int_support_weight * \
                              self.model.getRowNumIntCols(cuts[i]) / cuts[i].getNNonz()
                obj_parallel = self.obj_parallel_weight * self.model.getRowObjParallelism(cuts[i])
                efficacy = (self.efficacy_weight + self.dir_cutoff_dist_weight) * self.model.getCutEfficacy(cuts[i])
                score = int_support + obj_parallel + efficacy
                score += 1e-4 if cuts[i].isInGlobalCutpool() else 0
                score += random.uniform(0, 1e-6)
                max_score = max(max_score, score)
                scores[i] = score

        return max_score, scores

    def filter_with_parallelism(self, n_cuts, nselectedcuts, cut, cuts, scores, max_parallel, good_max_parallel,
                                good_score):
        """
        Filters the given cut list by any cut_iter in cuts that is too parallel to cut. It does this by moving the
        parallel cut to the back of cuts, and decreasing the indices of the list that are scanned over.
        For the main portion of our selection we then never touch these cuts. In the case of us wanting to
        forcefully select an amount which is impossible under this filtering method however, we simply select the
        remaining highest scored cuts from the supposed untouched cuts.
        @param n_cuts: The number of cuts that are still viable candidates
        @type n_cuts: int
        @param nselectedcuts: The number of cuts already selected
        @type nselectedcuts: int
        @param cut: The cut which we will add, and are now using to filter the remaining cuts
        @type cut: pyscipopt row
        @param cuts: The list of cuts
        @type cuts: List of pyscipopt rows
        @param scores: The scores of each cut
        @type scores: List of floats
        @param max_parallel: The maximum allowed parallelism for non good cuts
        @type max_parallel: Float
        @param good_max_parallel: The maximum allowed parallelism for good cuts
        @type good_max_parallel: Float
        @param good_score: The benchmark of whether a cut is 'good' and should have it's allowed parallelism increased
        @type good_score: Float
        @return: The now number of viable cuts, the complete list of cuts, and the complete list of scores
        @rtype: int, list of pyscipopt rows, list of pyscipopt rows
        """
        # Go backwards through the still viable cuts.
        for i in range(nselectedcuts + n_cuts - 1, nselectedcuts - 1, -1):
            cut_parallel = self.model.getRowParallelism(cut, cuts[i])
            # The maximum allowed parallelism depends on the whether the cut is 'good'
            allowed_parallel = good_max_parallel if scores[i] >= good_score else max_parallel
            if cut_parallel > allowed_parallel:
                # Throw the cut to the end of the viable cuts and decrease the number of viable cuts
                cuts[nselectedcuts + n_cuts - 1], cuts[i] = cuts[i], cuts[nselectedcuts + n_cuts - 1]
                scores[nselectedcuts + n_cuts - 1], scores[i] = scores[i], scores[nselectedcuts + n_cuts - 1]
                n_cuts -= 1

        return n_cuts, cuts, scores

    def select_best_cut(self, n_cuts, nselectedcuts, cuts, scores):
        """
        Moves the cut with highest score which is still considered viable (not too parallel to previous cuts) to the
        front of the list. Note that 'front' here still has the requirement that all added cuts are still behind it.
        @param n_cuts: The number of still viable cuts
        @type n_cuts: int
        @param nselectedcuts: The number of cuts already selected to be added
        @type nselectedcuts: int
        @param cuts: The list of cuts themselves
        @type cuts: List of pyscipopt rows
        @param scores: The scores of each cut
        @type scores: List of floats
        @return: The re-sorted list of cuts, and the re-sorted list of scores
        @rtype: List of pyscipopt rows, list of floats
        """
        # Initialise the best index and score
        best_pos = nselectedcuts
        best_score = scores[nselectedcuts]
        for i in range(nselectedcuts + 1, nselectedcuts + n_cuts):
            if scores[i] > best_score:
                best_pos = i
                best_score = scores[i]
        # Move the cut with highest score to the front of the still viable cuts
        cuts[nselectedcuts], cuts[best_pos] = cuts[best_pos], cuts[nselectedcuts]
        scores[nselectedcuts], scores[best_pos] = scores[best_pos], scores[nselectedcuts]
        return cuts, scores

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





def read_feature_vector_files(problem_dir, instance, rand_seed, torch_output=False):
    """
    This function just grabs the pre-calculated bipartite graph features from generate_features that have been
    written to files.
    Args:
        problem_dir: The directory containing all appropriate files
        instance: The instance name
        rand_seed: The SCIP random seed shift used in the model pre-solving
        torch_output: Boolean on whether you want torch or numpy as the output format
    Returns:
        The edge_indices, coefficients, col_features, row_features of the bipartite graph representation of the instance
    """

    edge_indices = np.load(
        os.path.join(problem_dir, '{}__trans__seed__{}__edge_indices.npy'.format(instance, rand_seed)))
    coefficients = np.load(
        os.path.join(problem_dir, '{}__trans__seed__{}__coefficients.npy'.format(instance, rand_seed)))
    col_features = np.load(
        os.path.join(problem_dir, '{}__trans__seed__{}__col_features.npy'.format(instance, rand_seed)))
    row_features = np.load(
        os.path.join(problem_dir, '{}__trans__seed__{}__row_features.npy'.format(instance, rand_seed)))


    return edge_indices, coefficients, col_features, row_features


def read_cut_selector_param_file(problem_dir, instance, rand_seed, sample_i):
    """
    This function just grabs the pre-calculated cut-selector parameters that have been written to file.
    Args:
        problem_dir: The directory containing all appropriate files
        instance: The instance name
        rand_seed: The SCIP random seed shift used in the model pre-solving
        sample_i: The sample index used in the run to produce the saved file
    Returns:
        dir_cut_off, efficacy, int_support, obj_parallelism
    """

    # Get the saved file
    file_name = get_filename(problem_dir, instance, rand_seed, trans=True, root=False, sample_i=sample_i, ext='npy')
    cut_selector_params = np.load(file_name)

    dir_cut_off, efficacy, int_support, obj_parallelism = cut_selector_params.tolist()

    return dir_cut_off, efficacy, int_support, obj_parallelism


def remove_slurm_files(outfile_dir):
    """
    Removes all files from outfile_dir.
    Args:
        outfile_dir: The output directory containing all of our slurm .out files
    Returns:
        Nothing. It simply deletes the files
    """

    assert not outfile_dir == '/' and not outfile_dir == ''

    # Delete everything
    shutil.rmtree(outfile_dir)

    # Make the directory itself again
    os.mkdir(outfile_dir)

    return


def remove_temp_files(temp_dir):
    """
    Removes all files from the given directory
    Args:
        temp_dir: The directory containing all information that is batch specific
    Returns:
        Nothing, the function deletes all files in the given directory
    """

    # Get all files in the directory
    files = os.listdir(temp_dir)

    # Now cycle through the files and delete them
    for file in files:
        os.remove(os.path.join(temp_dir, file))

    return


def remove_instance_solve_data(data_dir, instance, suppress_warnings=False):
    """
    Removes all .mps, .npy, .yml, .sol, and .log files associated with the instance.
    Args:
        data_dir: The directory where we store all of our instance data
        instance: The instance name
        suppress_warnings: Whether the warnings of the files being deletes should be suppressed
    Returns:
        Nothing
    """

    assert os.path.isdir(data_dir)
    assert type(instance) == str

    # Get all files in the directory
    files = os.listdir(data_dir)

    # Get all files that being with our instance
    files = [file for file in files if file.split('__')[0] == instance]

    for file in files:
        if file.endswith('.yml') or file.endswith('.log') or file.endswith('.sol') or file.endswith('.mps')\
                or file.endswith('.npy') or file.endswith('.stats'):
            if not suppress_warnings:
                logging.warning('Deleting file {}'.format(os.path.join(data_dir, file)))
            os.remove(os.path.join(data_dir, file))

    return


def run_python_slurm_job(python_file, job_name, outfile, time_limit, arg_list, dependencies=None, num_cpus=1,
                         exclusive=False):
    """
    Function for calling a python file through slurm. This offloads the job from the current call
    and let's multiple processes run simultaneously. These processes can then share information though input output.
    Note: Spawned processes cannot directly communicate with each other
    Args:
        python_file: The python file that wil be run
        job_name: The name to give the python run in slurm
        outfile: The file in which all output from the python run will be stored
        time_limit: The time limit on the slurm job in minutes
        arg_list: The list containing all args that will be added to the python call
        dependencies: A list of slurm job ID dependencies that must first complete before this job starts
        num_cpus: The number of CPUS assigned to the single job
        exclusive: Whether the job should be the only jbo to run on a node. Doing this ignores mem and num_cpus
    Returns:
        Nothing. It simply starts a python job through the command line that will be run in slurm
    """

    if dependencies is None:
        dependencies = []
    assert os.path.isfile(python_file) and python_file.endswith('.py')
    assert not os.path.isfile(outfile) and outfile.endswith('.out'), '{}'.format(outfile)
    assert os.path.isdir(os.path.dirname(outfile)), '{}'.format(outfile)
    assert type(time_limit) == int and 0 <= time_limit <= 1e+8
    assert type(arg_list) == list
    assert dependencies is None or (type(dependencies) == list and
                                    all(type(dependency) == int for dependency in dependencies))

    # Get the current working environment.
    my_env = os.environ.copy()

    # Give the base command line call for running a single slurm job through shell.
    cmd_1 = ['sbatch',
             '--job-name={}'.format(job_name),
             '--time=0-00:{}:00'.format(time_limit)]

    if exclusive:
        # This flag makes the timing reproducible, as no memory is shared between it and other jobs.
        cmd_2 = ['--exclusive']
    else:
        # We don't run exclusive always as we want more throughput. The run is still deterministic, but time can vary
        cmd_2 = ['--cpus-per-task={}'.format(num_cpus)]
        # If you wanted to add memory limits; '--mem={}'.format(mem), where mem is in MB, e.g. 8000=8GB
    if dependencies is not None and len(dependencies) > 0:
        # Add the dependencies if they exist
        dependency_str = ''.join([str(dependency) + ':' for dependency in dependencies])[:-1]
        cmd_2 += ['--dependency=afterany:{}'.format(dependency_str)]

    cmd_3 = ['-p',
             parameters.SLURM_QUEUE,
             '--output',
             outfile,
             '--error',
             outfile,
             '{}'.format(python_file)]

    cmd = cmd_1 + cmd_2 + cmd_3

    # Add all arguments of the python file afterwards
    for arg in arg_list:
        cmd.append('{}'.format(arg))

    # Run the command in shell.
    p = subprocess.Popen(cmd, env=my_env, stdout=subprocess.PIPE)
    p.wait()

    # Now access the stdout of the subprocess for the job ID
    job_line = ''
    for line in p.stdout:
        job_line = str(line.rstrip())
        break
    assert 'Submitted batch job' in job_line, print(job_line)
    job_id = int(job_line.split(' ')[-1].split("'")[0])

    del p

    return job_id


def get_filename(parent_dir, instance, rand_seed=None, trans=False, root=False, sample_i=None, ext='yml'):
    """
    The main function for retrieving the file names for all non-temporary files. It is a shortcut to avoid constantly
    rewriting the names of the different files, such as the .yml, .sol, .log and .mps files
    Args:
        parent_dir: The parent directory where the file belongs
        instance: The instance name of the SCIP problem
        rand_seed: The random seed used in the SCIP run
        trans: Whether the filename contains the substring trans (problem has been pre-solved)
        root: If root should be included in the file name
        sample_i: The sample index used to perturb the SCIP cut-sel params
        ext: The extension of the file, e.g. yml or sol
    Returns:
        The filename e.g. 'parent_dir/toll-like__trans__seed__2__sample__2.mps'
    """

    # Initialise the base_file name. This always contains the instance name
    base_file = instance
    if trans:
        base_file += '__trans'
    if root:
        base_file += '__root'
    if rand_seed is not None:
        base_file += '__seed__{}'.format(rand_seed)
    if not (sample_i is False or sample_i is None):
        base_file += '__sample__{}'.format(sample_i)

    # Add the extension to the base file
    if ext is not None:
        base_file += '.{}'.format(ext)

    # Now join the file with its parent dir
    return os.path.join(parent_dir, base_file)


def get_slurm_output_file(outfile_dir, instance, rand_seed):
    """
    Function for getting the slurm output log for the current run.
    Args:
        outfile_dir: The directory containing all slurm .log files
        instance: The instance name
        rand_seed: The instance random seed
    Returns:
        The slurm .out file which is currently being used
    """

    assert os.path.isdir(outfile_dir)
    assert type(instance) == str
    assert type(rand_seed) == int

    # Get all slurm out files
    out_files = os.listdir(outfile_dir)

    # Get a unique substring that will only be contained for a single run
    file_substring = '__{}__seed__{}'.format(instance, rand_seed)

    unique_file = [out_file for out_file in out_files if file_substring in out_file]
    assert len(unique_file) == 1, 'Instance {} with rand_seed {} has no outfile in {}'.format(instance, rand_seed,
                                                                                              outfile_dir)
    return os.path.join(outfile_dir, unique_file[0])


def str_to_bool(word):
    """
    This is used to check if a string is trying to represent a boolean True.
    We need this because argparse doesnt by default have such a function, and using using bool('False') evaluate to True
    Args:
        word: The string we want to convert to a boolean
    Returns:
        Whether the string is representing True or not.
    """
    assert type(word) == str
    return word.lower() in ["yes", "true", "t", "1"]


def is_dir(path):
    """
    This is used to check if a string is trying to represent a directory when we parse it into argparse.
    Args:
        path: The path to a directory
    Returns:
        The string path if it is a valid directory else we raise an error
    """
    assert type(path) == str, print('{} is not a string!'.format(path))
    exists = os.path.isdir(path)
    if not exists:
        raise argparse.ArgumentTypeError('{} is not a valid directory'.format(path))
    else:
        return path


def is_file(path):
    """
    This is used to check if a string is trying to represent a file when we parse it into argparse.
    Args:
        path: The path to a file
    Returns:
        The string path if it is a valid file else we raise an error
    """
    assert type(path) == str, print('{} is not a string!'.format(path))
    exists = os.path.isfile(path)
    if not exists:
        raise argparse.ArgumentTypeError('{} is not a valid file'.format(path))
    else:
        return path
