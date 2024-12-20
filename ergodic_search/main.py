import numpy as np
import matplotlib.pyplot as plt
import pywt
import os
import random
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpBinary, LpStatus

# Import ErgPlanner from erg_planner.py
from erg_planner import ErgPlanner

###########################################
# Helper Functions
###########################################

def create_erg_args():
    """
    Create a dummy args object similar to what ErgPlanner expects.
    Adjust these parameters as necessary for your specific use case.
    """
    class Args:
        learn_rate = [0.001]       # Learning rate for optimizer
        num_pixels = 100           # Number of pixels along one side of the map
        gpu = False                # Use GPU if available
        traj_steps = 100           # Number of steps in trajectory
        iters = 500                # Max iterations for trajectory optimization
        epsilon = 0.01             # Threshold for ergodic metric
        start_pose = [0., 0., 0.]  # Starting position (x, y, theta)
        end_pose = [0., 0., 0.]    # Ending position (x, y, theta)
        num_freqs = 0              # Number of frequencies for Fourier transforms
        erg_wt = 1.0               # Weight on ergodic metric in loss function
        transl_vel_wt = 0.1        # Weight on translational velocity
        ang_vel_wt = 0.05          # Weight on angular velocity
        bound_wt = 1000            # Weight on boundary condition
        end_pose_wt = 0.5          # Weight on end position
        debug = False              # Debug mode
        outpath = None             # Output path for saving images
        replan_type = 'full'       # Type of replanning ('full' or 'partial')
    return Args()

def run_erg_planner_on_agent(agent_map):
    """
    Instantiate the ErgPlanner with the agent's map and compute the ergodic metric.
    """
    # Normalize the agent_map to create a probability distribution
    pdf = agent_map - np.min(agent_map)
    if np.sum(pdf) == 0:
        pdf[:] = 1.0
    pdf = pdf / np.sum(pdf)
    
    # Create arguments for the planner
    args = create_erg_args()

    # Instantiate ErgPlanner with the normalized PDF
    # Initialize controls to zeros
    planner = ErgPlanner(args, pdf=pdf.flatten(), init_controls=np.zeros((args.traj_steps, 2)))
    
    # Compute trajectory and ergodic metric
    _, _, erg = planner.compute_traj(debug=False)
    
    return erg

def reconstruct_agent_maps(labels_full, approx_points, detail_points, wavelet, K, X_low_fidelity, horiz_detail, vert_detail, diag_detail):
    """
    Reconstruct each agent's map based on cluster assignments.
    """
    agent_maps = []
    num_approx = len(approx_points)
    labels_approx_sol = labels_full[:num_approx]
    labels_detail_sol = labels_full[num_approx:] - X_low_fidelity
    
    # Indices for detail coefficients
    h_h, h_w = horiz_detail.shape
    num_dpoints = h_h * h_w
    h_idx = np.arange(0, num_dpoints)
    v_idx = np.arange(num_dpoints, 2*num_dpoints)
    di_idx = np.arange(2*num_dpoints, 3*num_dpoints)
    
    for agent_id in range(K):
        if agent_id < X_low_fidelity:
            # Low-Fidelity Agent: Approximation
            approx_cluster = np.zeros_like(approx)
            cluster_indices = np.where(labels_approx_sol == agent_id)[0]
            for idx in cluster_indices:
                x_coord = int(approx_points[idx,0])
                y_coord = int(approx_points[idx,1])
                val = approx_points[idx,2]
                # Ensure indices are within bounds
                if 0 <= x_coord < approx.shape[1] and 0 <= y_coord < approx.shape[0]:
                    approx_cluster[y_coord, x_coord] = val

            # Zero out detail coefficients
            horiz_zero = np.zeros_like(horiz_detail)
            vert_zero = np.zeros_like(vert_detail)
            diag_zero = np.zeros_like(diag_detail)

            # Perform inverse wavelet transform
            recon_map = pywt.idwt2((approx_cluster, (horiz_zero, vert_zero, diag_zero)), wavelet)
            agent_maps.append(recon_map)
        else:
            # High-Fidelity Agent: Detail
            detail_cluster_idx = agent_id - X_low_fidelity
            approx_zero = np.zeros_like(approx)
            horiz_c = np.zeros_like(horiz_detail)
            vert_c = np.zeros_like(vert_detail)
            diag_c = np.zeros_like(diag_detail)
            
            cluster_indices = np.where(labels_detail_sol == detail_cluster_idx)[0]
            for idx in cluster_indices:
                val = detail_points[idx,2]
                x_coord = int(detail_points[idx,0])
                y_coord = int(detail_points[idx,1])
                
                # Assign to the appropriate detail coefficient
                if idx in h_idx:
                    if 0 <= x_coord < horiz_detail.shape[1] and 0 <= y_coord < horiz_detail.shape[0]:
                        horiz_c[y_coord, x_coord] = val
                elif idx in v_idx:
                    v_local = idx - num_dpoints
                    vy = v_local // h_w
                    vx = v_local % h_w
                    if 0 <= vx < vert_detail.shape[1] and 0 <= vy < vert_detail.shape[0]:
                        vert_c[vy, vx] = val
                elif idx in di_idx:
                    di_local = idx - 2*num_dpoints
                    diy = di_local // h_w
                    dix = di_local % h_w
                    if 0 <= dix < diag_detail.shape[1] and 0 <= diy < diag_detail.shape[0]:
                        diag_c[diy, dix] = val

            # Perform inverse wavelet transform
            recon_map = pywt.idwt2((approx_zero, (horiz_c, vert_c, diag_c)), wavelet)
            agent_maps.append(recon_map)
    
    return agent_maps

def solve_and_evaluate(mismatch_penalty, mix_penalty, balance_penalty, dists, all_labels, K, N, approx_points, detail_points, wavelet, X_low_fidelity, horiz_detail, vert_detail, diag_detail):
    """
    Solve the ILP with given penalties and evaluate the ergodic metric.
    """
    # Initialize ILP model
    model = LpProblem("Joint_Clustering_Tuning_with_Erg", LpMinimize)
    z = LpVariable.dicts("z",(range(N), range(K)), cat=LpBinary)
    
    # Define objective terms
    variance_term = lpSum([dists[i,c]*z[i][c] for i in range(N) for c in range(K)])
    mismatch_term = lpSum([mismatch_penalty*z[i,c]
                           for i in range(N) for c in range(K)
                           if (all_labels[i] == 0 and c >= X_low_fidelity) or
                              (all_labels[i] == 1 and c < X_low_fidelity)])
    
    # Add to model
    model += variance_term + mismatch_term
    
    # Constraint: Each point assigned exactly once
    for i in range(N):
        model += lpSum([z[i][c] for c in range(K)]) == 1
    
    # Variables to track presence of approximation and detail points in clusters
    has_approx = LpVariable.dicts("has_approx", range(K), cat=LpBinary)
    has_detail = LpVariable.dicts("has_detail", range(K), cat=LpBinary)
    M = 10000  # Big M
    
    for c in range(K):
        # Approximation points in cluster c
        model += lpSum([z[i][c] for i in range(N) if all_labels[i] == 0]) >= has_approx[c]
        model += lpSum([z[i][c] for i in range(N) if all_labels[i] == 0]) <= M * has_approx[c]
        # Detail points in cluster c
        model += lpSum([z[i][c] for i in range(N) if all_labels[i] == 1]) >= has_detail[c]
        model += lpSum([z[i][c] for i in range(N) if all_labels[i] == 1]) <= M * has_detail[c]
    
    # Mixing constraints
    w = LpVariable.dicts("w", range(K), cat=LpBinary)
    for c in range(K):
        model += w[c] <= has_approx[c]
        model += w[c] <= has_detail[c]
        model += w[c] >= has_approx[c] + has_detail[c] - 1
    
    # Mixing penalty
    mix_term = lpSum([mix_penalty*w[c] for c in range(K)])
    
    # Balancing constraints
    n_star = N / float(K)
    cluster_size_plus = {c: LpVariable(f"cluster_size_plus_{c}", lowBound=0) for c in range(K)}
    cluster_size_neg = {c: LpVariable(f"cluster_size_neg_{c}", lowBound=0) for c in range(K)}
    
    for c in range(K):
        model += (lpSum([z[i][c] for i in range(N)]) - n_star) == cluster_size_plus[c] - cluster_size_neg[c]
    
    # Balancing penalty
    balance_term = lpSum([balance_penalty*(cluster_size_plus[c] + cluster_size_neg[c]) for c in range(K)])
    
    # Complete Objective
    model.setObjective(variance_term + mismatch_term + mix_term + balance_term)
    
    # Solve ILP
    print(f"Solving ILP with mismatch={mismatch_penalty}, mix={mix_penalty}, balance={balance_penalty}...")
    model.solve()
    
    if LpStatus[model.status] != 'Optimal':
        print(f"ILP not optimal for params mismatch={mismatch_penalty}, mix={mix_penalty}, balance={balance_penalty}")
        return None, np.inf
    
    # Extract solution
    labels_full = np.zeros(N, dtype=int)
    for i in range(N):
        for c in range(K):
            if z[i][c].varValue > 0.5:
                labels_full[i] = c
                break
    
    # Reconstruct agent maps
    agent_maps = reconstruct_agent_maps(labels_full, approx_points, detail_points, wavelet, K, X_low_fidelity, horiz_detail, vert_detail, diag_detail)
    
    # Compute ergodic metric for each agent's map
    erg_list = []
    for agent_map in agent_maps:
        erg_score = run_erg_planner_on_agent(agent_map)
        erg_list.append(erg_score)
    
    # Average ergodicity
    avg_erg = np.mean(erg_list)
    print(f"Parameters mismatch={mismatch_penalty}, mix={mix_penalty}, balance={balance_penalty} -> Avg Ergodicity: {avg_erg}")
    
    return labels_full, avg_erg

def reconstruct_and_save(agent_maps, labels_full, approx_points, detail_points, wavelet, K, X_low_fidelity, output_dir):
    """
    Reconstruct and save agent maps as images.
    """
    num_approx = len(approx_points)
    labels_approx_sol = labels_full[:num_approx]
    labels_detail_sol = labels_full[num_approx:] - X_low_fidelity
    
    # Indices for detail coefficients
    h_h, h_w = horiz_detail.shape
    num_dpoints = h_h * h_w
    h_idx = np.arange(0, num_dpoints)
    v_idx = np.arange(num_dpoints, 2*num_dpoints)
    di_idx = np.arange(2*num_dpoints, 3*num_dpoints)
    
    # Save reconstructed maps
    for agent_id, agent_map in enumerate(agent_maps):
        if agent_id < X_low_fidelity:
            # Low-Fidelity Agent
            plt.figure()
            plt.imshow(agent_map, origin='lower', extent=[-5,5,-5,5])
            plt.title(f"Reconstructed Map: Low-Fidelity Agent {agent_id}")
            plt.colorbar()
            plt.savefig(os.path.join(output_dir, f"reconstructed_low_fidelity_agent_{agent_id}.png"), dpi=150)
            plt.close()
        else:
            # High-Fidelity Agent
            plt.figure()
            plt.imshow(agent_map, origin='lower', extent=[-5,5,-5,5])
            plt.title(f"Reconstructed Map: High-Fidelity Agent {agent_id}")
            plt.colorbar()
            plt.savefig(os.path.join(output_dir, f"reconstructed_high_fidelity_agent_{agent_id}.png"), dpi=150)
            plt.close()

###########################################
# Main Parameter Tuning Loop
###########################################

def main():
    # Initialize output directory
    output_dir = "clustering_results_tuning_with_erg"
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Generate Synthetic Map (already done above)
    # Steps 2 and 3: Wavelet Decomposition and Data Preparation (already done above)
    # No need to repeat these steps here
    
    # Parameter ranges
    mismatch_values = [1, 10, 20]      # Example values
    mix_values = [1, 5, 10]            # Example values
    balance_values = [0.01, 0.1, 1.0]  # Example values

    print("hello 1!")
    
    best_params = None
    best_ergodicity = np.inf

    print("hello!")
    
    # Iterate over all combinations of parameters
    for m_val in mismatch_values:
        for mx_val in mix_values:
            for b_val in balance_values:
                labels_full, avg_erg = solve_and_evaluate(
                    mismatch_penalty=m_val,
                    mix_penalty=mx_val,
                    balance_penalty=b_val,
                    dists=dists,
                    all_labels=all_labels,
                    K=K,
                    N=N,
                    approx_points=approx_points,
                    detail_points=detail_points,
                    wavelet=wavelet,
                    X_low_fidelity=X_low_fidelity,
                    horiz_detail=horiz_detail,
                    vert_detail=vert_detail,
                    diag_detail=diag_detail
                )
                
                if labels_full is None:
                    # ILP was not optimal for this parameter set
                    continue
                
                # Check if this is the best ergodicity so far
                if avg_erg < best_ergodicity:
                    best_ergodicity = avg_erg
                    best_params = (m_val, mx_val, b_val)
                    
                    # Reconstruct agent maps
                    agent_maps = reconstruct_agent_maps(labels_full, approx_points, detail_points, wavelet, K, X_low_fidelity, horiz_detail, vert_detail, diag_detail)
                    
                    # Save reconstructed maps for the best parameters
                    reconstruct_and_save(agent_maps, labels_full, approx_points, detail_points, wavelet, K, X_low_fidelity, output_dir)
    
    # After all iterations, report the best parameters
    if best_params is not None:
        print("\nBest parameters found:")
        print(f"Mismatch penalty: {best_params[0]}")
        print(f"Mix penalty: {best_params[1]}")
        print(f"Balance penalty: {best_params[2]}")
        print(f"With average ergodicity measure: {best_ergodicity}")
    else:
        print("No optimal solution found for any parameter set.")

if __name__ == "__main__":
    print("HELLLOOOOO\n\n\n\n\n\n\n\n\n")
    main()
