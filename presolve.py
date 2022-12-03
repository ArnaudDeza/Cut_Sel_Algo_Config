

from scip_solve import build_scip_model
from utilities import  get_filename

PRESOLVE_TIME_LIMIT = 300



def presolve_instance(transformed_problem_dir, instance_path, sol_path, instance,rand_seed, use_miplib_sols):

    # Generate the instance post pre-solve and print out the transformed model
    sol_path = sol_path if use_miplib_sols else None
    scip = build_scip_model(instance_path, 1, rand_seed, True, True, False, False, False, True,
                            time_limit=PRESOLVE_TIME_LIMIT, sol_path=sol_path)

    if use_miplib_sols:
        # The original solution is not always feasible in the transformed space. We thus disable dual pre-solve
        # We only do this for the MIPLIB solutions, as in the other case we generate the solution ourselves.
        scip.setParam('misc/allowstrongdualreds', False)
        scip.setParam('misc/allowweakdualreds', False)
        scip.setParam('presolving/dualagg/maxrounds', 0)
        scip.setParam('presolving/dualcomp/maxrounds', 0)
        scip.setParam('presolving/dualinfer/maxrounds', 0)
        scip.setParam('presolving/dualsparsify/maxrounds', 0)

    # Put the instance through pre-solving only. We run optimize though as we additionally want our solution transformed
    scip.optimize()

    # If the instance hit the time-limit, then we can simply ignore it. Note that this includes a single root LP solve
    if scip.getStatus() == 'timelimit':
        scip.freeProb()
        print("time limit on presolve")
        quit()
    if scip.getStatus() == 'optimal':
        scip.freeProb()
        print("optimal on presolve")
        quit()

    # Get the file_name of the transformed instance that we're going to write it out to
    transformed_file_name = get_filename(transformed_problem_dir, instance, rand_seed, trans=True,root=False, sample_i=None, ext='mps')

    # Get the file_name of the transformed solution that we're going to write it out to
    transformed_sol_name = get_filename(transformed_problem_dir, instance, rand_seed, trans=True,root=False, sample_i=None, ext='sol')

    # Write the actual transformed instance file
    scip.writeProblem(filename=transformed_file_name, trans=True)

    # In the case of using MIPLIB solutions, we want to print out the pre-loaded transformed solution
    if use_miplib_sols:
        # We manually construct the solutions, as it is possible that during presolve and the root-solve that a better
        # solution that's infeasible in transformed space has been found, and writeBestTransSol would produce an error
        sols = scip.getSols()
        use_miplib_sols = scip.createSol()
        for var in scip.getVars(transformed=True):
            var_val = scip.getSolVal(sols[0], var)
            scip.setSolVal(use_miplib_sols, var, var_val)
        scip.writeTransSol(use_miplib_sols, filename=transformed_sol_name)

    scip.freeProb()

    return transformed_file_name