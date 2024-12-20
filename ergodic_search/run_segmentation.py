import numpy as np
import matplotlib.pyplot as plt
import pywt
import os
import argparse
import sys
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpBinary, LpStatus
from erg_planner import ErgPlanner

def create_erg_args(num_freqs=100):
    """
    Create an Args object with specified parameters for ErgPlanner.
    """
    class Args:
        def __init__(self, num_freqs):
            self.learn_rate = [0.001]
            self.num_pixels = 100
            self.gpu = False
            self.traj_steps = 100
            self.iters = 500
            self.epsilon = 0.01
            self.start_pose = [0., 0., 0.]
            self.end_pose = [0., 0., 0.]
            self.erg_wt = 1.0
            self.transl_vel_wt = 0.1
            self.ang_vel_wt = 0.05
            self.bound_wt = 1000
            self.end_pose_wt = 0.5
            self.debug = False
            self.outpath = None
            self.replan_type = 'full'
            self.num_freqs = num_freqs
    return Args(num_freqs)

def run_erg_planner_on_agent(agent_map, num_freqs=100):
    """
    Instantiate the ErgPlanner with the agent's map and compute the ergodic metric and trajectory.
    """
    pdf = agent_map - np.min(agent_map)
    if np.sum(pdf) == 0:
        pdf[:] = 1.0
    pdf = pdf / np.sum(pdf)
    args = create_erg_args(num_freqs=num_freqs)
    planner = ErgPlanner(args, pdf=pdf.flatten(), init_controls=np.zeros((args.traj_steps, 2)))
    controls, traj, erg = planner.compute_traj(debug=False)
    erg_score = erg.detach().item()
    traj = traj.detach().cpu().numpy()  # Assuming traj is a tensor of shape (traj_steps, 3)
    return erg_score, traj

def reconstruct_agent_maps(labels_full, approx_points, detail_points, wavelet, K, X_low_fidelity, horiz_detail, vert_detail, diag_detail, approx_coeffs):
    """
    Reconstruct each agent's map based on cluster assignments.
    """
    agent_maps = []
    num_approx = len(approx_points)
    labels_approx_sol = labels_full[:num_approx]
    labels_detail_sol = labels_full[num_approx:] - X_low_fidelity
    h_h, h_w = horiz_detail.shape
    num_dpoints = h_h * h_w
    h_idx = np.arange(0, num_dpoints)
    v_idx = np.arange(num_dpoints, 2*num_dpoints)
    di_idx = np.arange(2*num_dpoints, 3*num_dpoints)
    
    for agent_id in range(K):
        if agent_id < X_low_fidelity:
            approx_cluster = np.zeros_like(approx_coeffs)
            cluster_indices = np.where(labels_approx_sol == agent_id)[0]
            for idx in cluster_indices:
                x_coord = int(approx_points[idx,0])
                y_coord = int(approx_points[idx,1])
                val = approx_points[idx,2]
                if 0 <= x_coord < approx_coeffs.shape[1] and 0 <= y_coord < approx_coeffs.shape[0]:
                    approx_cluster[y_coord, x_coord] = val
            horiz_zero = np.zeros_like(horiz_detail)
            vert_zero = np.zeros_like(vert_detail)
            diag_zero = np.zeros_like(diag_detail)
            recon_map = pywt.idwt2((approx_cluster, (horiz_zero, vert_zero, diag_zero)), wavelet)
            agent_maps.append(recon_map)
        else:
            detail_cluster_idx = agent_id - X_low_fidelity
            approx_zero = np.zeros_like(approx_coeffs)
            horiz_c = np.zeros_like(horiz_detail)
            vert_c = np.zeros_like(vert_detail)
            diag_c = np.zeros_like(diag_detail)
            cluster_indices = np.where(labels_detail_sol == detail_cluster_idx)[0]
            for idx in cluster_indices:
                val = detail_points[idx,2]
                x_coord = int(detail_points[idx,0])
                y_coord = int(detail_points[idx,1])
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
            recon_map = pywt.idwt2((approx_zero, (horiz_c, vert_c, diag_c)), wavelet)
            agent_maps.append(recon_map)
    return agent_maps

def solve_and_evaluate(mismatch_penalty, mix_penalty, balance_penalty, dists, all_labels, K, N, approx_points, detail_points, wavelet, X_low_fidelity, horiz_detail, vert_detail, diag_detail, approx_coeffs, info_map, num_freqs=100):
    """
    Solve the ILP with given penalties and evaluate the joint ergodic metric.
    """
    model = LpProblem("Joint_Clustering_Tuning_with_Erg", LpMinimize)
    z = LpVariable.dicts("z",(range(N), range(K)), cat=LpBinary)
    variance_term = lpSum([dists[i,c]*z[i][c] for i in range(N) for c in range(K)])
    mismatch_term = lpSum([mismatch_penalty*z[i][c]
                           for i in range(N) for c in range(K)
                           if (all_labels[i] == 0 and c >= X_low_fidelity) or
                              (all_labels[i] == 1 and c < X_low_fidelity)])
    model += variance_term + mismatch_term
    for i in range(N):
        model += lpSum([z[i][c] for c in range(K)]) == 1
    has_approx = LpVariable.dicts("has_approx", range(K), cat=LpBinary)
    has_detail = LpVariable.dicts("has_detail", range(K), cat=LpBinary)
    M = 10000
    for c in range(K):
        model += lpSum([z[i][c] for i in range(N) if all_labels[i] == 0]) >= has_approx[c]
        model += lpSum([z[i][c] for i in range(N) if all_labels[i] == 0]) <= M * has_approx[c]
        model += lpSum([z[i][c] for i in range(N) if all_labels[i] == 1]) >= has_detail[c]
        model += lpSum([z[i][c] for i in range(N) == 1]) <= M * has_detail[c]
    w = LpVariable.dicts("w", range(K), cat=LpBinary)
    for c in range(K):
        model += w[c] <= has_approx[c]
        model += w[c] <= has_detail[c]
        model += w[c] >= has_approx[c] + has_detail[c] - 1
    mix_term = lpSum([mix_penalty*w[c] for c in range(K)])
    n_star = N / float(K)
    cluster_size_plus = {c: LpVariable(f"cluster_size_plus_{c}", lowBound=0) for c in range(K)}
    cluster_size_neg = {c: LpVariable(f"cluster_size_neg_{c}", lowBound=0) for c in range(K)}
    for c in range(K):
        model += (lpSum([z[i][c] for i in range(N)]) - n_star) == cluster_size_plus[c] - cluster_size_neg[c]
    balance_term = lpSum([balance_penalty*(cluster_size_plus[c] + cluster_size_neg[c]) for c in range(K)])
    model.setObjective(variance_term + mismatch_term + mix_term + balance_term)
    model.solve()
    if LpStatus[model.status] != 'Optimal':
        print(f"ILP did not find an optimal solution. Status: {LpStatus[model.status]}")
        return None, np.inf, []
    labels_full = np.zeros(N, dtype=int)
    for i in range(N):
        for c in range(K):
            if z[i][c].varValue > 0.5:
                labels_full[i] = c
                break
    agent_maps = reconstruct_agent_maps(labels_full, approx_points, detail_points, wavelet, K, X_low_fidelity, horiz_detail, vert_detail, diag_detail, approx_coeffs)
    # Collect all trajectories
    all_trajectories = []
    for agent_map in agent_maps:
        erg_score, traj = run_erg_planner_on_agent(agent_map, num_freqs=num_freqs)
        all_trajectories.append(traj)
    
    # Compute joint coverage statistics
    joint_coverage = np.zeros_like(info_map)
    for traj in all_trajectories:
        for pos in traj:
            # pos is assumed to have at least two elements: (x, y, ...)
            if len(pos) < 2:
                continue  # Skip if position data is incomplete
            x, y = pos[0], pos[1]
            ix = int((x + 5) / 10 * info_map.shape[1])
            iy = int((y + 5) / 10 * info_map.shape[0])
            if 0 <= ix < info_map.shape[1] and 0 <= iy < info_map.shape[0]:
                joint_coverage[iy, ix] += 1
    # Normalize joint coverage
    if len(all_trajectories) > 0 and len(all_trajectories[0]) > 0:
        joint_coverage /= (len(all_trajectories) * len(all_trajectories[0]))
    else:
        print("Warning: Trajectories are empty. Joint coverage remains zero.")
    
    # Compute ergodic metric using Fourier coefficients
    m = 10  # Number of Fourier modes
    erg_metric = 0
    # Compute Fourier coefficients
    C_k = np.fft.fft2(joint_coverage)
    C_k = C_k[:m, :m]
    xi_k = np.fft.fft2(info_map)
    xi_k = xi_k[:m, :m]
    # Compute weighted squared differences
    for k in range(m):
        for l in range(m):
            lambda_k = 1 / (k**2 + l**2 + 1e-6)  # Added small term to avoid division by zero
            erg_metric += lambda_k * np.abs(C_k[k, l] - xi_k[k, l])**2
    return labels_full, erg_metric, all_trajectories

def reconstruct_and_save(agent_maps, trajectories, output_dir, K, X_low_fidelity):
    """
    Reconstruct and save agent maps and plot trajectories on them.
    """
    for agent_id, (agent_map, traj) in enumerate(zip(agent_maps, trajectories)):
        plt.figure()
        plt.imshow(agent_map, origin='lower', extent=[-5,5,-5,5], cmap='viridis')
        plt.colorbar()
        if traj.size > 0 and traj.shape[1] >= 2:
            plt.plot(traj[:,0], traj[:,1], marker='o', markersize=2, label='Trajectory', color='red')
            plt.legend()
        if agent_id < X_low_fidelity:
            plt.title(f"Low-Fidelity Agent {agent_id} Map with Trajectory")
            filename = f"low_fidelity_agent_{agent_id}_trajectory.png"
        else:
            plt.title(f"High-Fidelity Agent {agent_id} Map with Trajectory")
            filename = f"high_fidelity_agent_{agent_id}_trajectory.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=150)
        plt.close()

def plot_all_trajectories_on_original(original_map, trajectories, output_dir):
    """
    Plot all agents' trajectories on the original information map.
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(original_map, origin='lower', extent=[-5,5,-5,5], cmap='viridis', alpha=0.7)
    plt.colorbar()
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow']
    for idx, traj in enumerate(trajectories):
        if traj.size > 0 and traj.shape[1] >= 2:
            plt.plot(traj[:,0], traj[:,1], marker='o', markersize=2, color=colors[idx % len(colors)], label=f'Agent {idx}')
    plt.title("All Agents' Trajectories on Original Information Map")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "all_agents_trajectories_on_original_map.png"), dpi=150)
    plt.close()

def parse_arguments():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Ergodic Search with Agent Trajectories and Fourier Analysis")
    parser.add_argument('--mismatch_penalty', type=int, required=True, help='Mismatch penalty parameter (integer)')
    parser.add_argument('--mix_penalty', type=int, required=True, help='Mix penalty parameter (integer)')
    parser.add_argument('--balance_penalty', type=float, required=True, help='Balance penalty parameter (float)')
    parser.add_argument('--num_freqs', type=int, required=True, help='Number of Fourier frequencies (integer)')
    parser.add_argument('--output_dir', type=str, default="clustering_results_tuning_with_erg", help='Directory to save output images')
    return parser.parse_args()

def main():
    """
    Main function to run the ergodic search with specified parameters and plot trajectories.
    """
    args = parse_arguments()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Generate Synthetic Map
    grid_size = 100
    Xg, Yg = np.meshgrid(np.linspace(-5,5,grid_size), np.linspace(-5,5,grid_size))
    gaussians = [
        {"amp": 1.0, "x0": 0.0, "y0": 0.0, "sigma_x": 0.3, "sigma_y": 0.3},
        {"amp": 0.5, "x0": 2.0, "y0": 2.0, "sigma_x": 1.5, "sigma_y": 1.5},
        {"amp": 0.8, "x0": -2.0, "y0": -2.0, "sigma_x": 0.8, "sigma_y": 0.8},
        {"amp": 0.2, "x0": -3.0, "y0": 2.5, "sigma_x": 2.0, "sigma_y": 2.0},
        {"amp": 0.7, "x0": 3.0, "y0": -3.0, "sigma_x": 1.0, "sigma_y": 0.5}
    ]
    info_map = np.zeros((grid_size, grid_size))
    for g in gaussians:
        info_map += g["amp"] * np.exp(-(((Xg - g["x0"])**2)/(2*g["sigma_x"]**2) 
                                        + ((Yg - g["y0"])**2)/(2*g["sigma_y"]**2)))
    plt.figure()
    plt.imshow(info_map, origin='lower', extent=[-5,5,-5,5], cmap='viridis')
    plt.title("Original Information Map")
    plt.colorbar()
    plt.savefig(os.path.join(output_dir, "original_info_map.png"), dpi=150)
    plt.close()
    
    # Step 2: Wavelet Decomposition
    wavelet = 'db2'
    coeffs = pywt.dwt2(info_map, wavelet)
    approx, (horiz_detail, vert_detail, diag_detail) = coeffs
    plt.figure()
    plt.imshow(approx, origin='lower', cmap='viridis')
    plt.title("Approximation Coefficients")
    plt.colorbar()
    plt.savefig(os.path.join(output_dir, "approximation_coeffs.png"), dpi=150)
    plt.close()
    
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.imshow(horiz_detail, origin='lower', cmap='viridis')
    plt.title("Horizontal Detail")
    plt.colorbar()
    plt.subplot(1,3,2)
    plt.imshow(vert_detail, origin='lower', cmap='viridis')
    plt.title("Vertical Detail")
    plt.colorbar()
    plt.subplot(1,3,3)
    plt.imshow(diag_detail, origin='lower', cmap='viridis')
    plt.title("Diagonal Detail")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "detail_coeffs.png"), dpi=150)
    plt.close()
    
    # Step 3: Prepare Data
    X_low_fidelity = 2
    Y_high_fidelity = 2
    K = X_low_fidelity + Y_high_fidelity
    
    a_h, a_w = approx.shape
    aX, aY = np.meshgrid(np.arange(a_w), np.arange(a_h))
    approx_points = np.stack([aX.flatten(), aY.flatten(), approx.flatten()], axis=1)
    
    h_h, h_w = horiz_detail.shape
    dX, dY = np.meshgrid(np.arange(h_w), np.arange(h_h))
    
    h_points = np.stack([dX.flatten(), dY.flatten(), horiz_detail.flatten()], axis=1)
    v_points = np.stack([dX.flatten(), dY.flatten(), vert_detail.flatten()], axis=1)
    di_points = np.stack([dX.flatten(), dY.flatten(), diag_detail.flatten()], axis=1)
    
    detail_points = np.concatenate([h_points, v_points, di_points], axis=0)
    
    approx_labels = np.zeros(len(approx_points), dtype=int)
    detail_labels = np.ones(len(detail_points), dtype=int)
    
    all_points = np.concatenate([approx_points, detail_points], axis=0)
    all_labels = np.concatenate([approx_labels, detail_labels])
    
    N = len(all_points)
    
    # Step 4: Candidate Clusters
    np.random.seed(42)
    candidate_indices = np.random.choice(N, K, replace=False)
    centroids = all_points[candidate_indices, :2]
    
    point_coords = all_points[:, :2]
    dists = np.zeros((N,K))
    for i in range(N):
        for c in range(K):
            dists[i,c] = np.sum((point_coords[i] - centroids[c])**2)
    
    # Step 5: Run with Specified Parameters
    print("Running ergodic search with specified parameters...")
    print(f"Mismatch penalty: {args.mismatch_penalty}")
    print(f"Mix penalty: {args.mix_penalty}")
    print(f"Balance penalty: {args.balance_penalty}")
    print(f"Number of Fourier frequencies: {args.num_freqs}")
    
    labels_full, erg_metric, trajectories = solve_and_evaluate(
        mismatch_penalty=args.mismatch_penalty,
        mix_penalty=args.mix_penalty,
        balance_penalty=args.balance_penalty,
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
        diag_detail=diag_detail,
        approx_coeffs=approx,
        info_map=info_map,
        num_freqs=args.num_freqs
    )
    
    if labels_full is not None:
        print("\nGenerating and saving plots...")
        agent_maps = reconstruct_agent_maps(labels_full, approx_points, detail_points, wavelet, K, X_low_fidelity, horiz_detail, vert_detail, diag_detail, approx)
        reconstruct_and_save(agent_maps, trajectories, output_dir, K, X_low_fidelity)
        plot_all_trajectories_on_original(info_map, trajectories, output_dir)
        print(f"All plots have been saved to the directory: {output_dir}")
        print(f"\nErgodic Metric: {erg_metric}")
    else:
        print("No valid labels found. Skipping plot generation and ergodic metric calculation.")

if __name__ == "__main__":
    main()
