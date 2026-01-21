import numpy as np
import scipy.optimize as opt
import csv
import os

# --- 1. PROBLEM PARAMETERS ---
INITIAL_EXTENT = 1.0
N_POINTS_X = 31  
N_POINTS_Y = 31  
MIN_DIST_SQ = 1.0**2

# --- 2. HELPER FUNCTIONS ---
def pack_variables(z1_internal, z2_internal, extent):
    return np.concatenate(([extent], z1_internal, z2_internal))

def unpack_variables(X, N_x, N_y, aspect_ratio):
    x_extent = X[0] * np.sqrt(aspect_ratio)
    y_extent = X[0] / np.sqrt(aspect_ratio)
    num_internal_x = N_x - 1
    z1_internal = X[1 : 1 + num_internal_x]
    z2_internal = X[1 + num_internal_x :]
    z1_full = np.concatenate((z1_internal, [0.0]))
    z2_full = np.concatenate((z2_internal, [0.0]))
    x_coords = np.linspace(0, x_extent, N_x)
    y_coords = np.linspace(0, y_extent, N_y)
    dx = (x_extent) / (N_x - 1)
    dy = (y_extent) / (N_y - 1)
    return z1_full, z2_full, x_coords, y_coords, dx, dy, x_extent, y_extent

# --- 3. OBJECTIVE FUNCTION ---
def objective_function(X, N_x, N_y, aspect_ratio):
    z1, z2, _, _, dx, dy, _, _ = unpack_variables(X, N_x, N_y, aspect_ratio)
    dz1 = np.diff(z1)
    len_1 = np.sum(np.sqrt(dx**2 + dz1**2))
    dz2 = np.diff(z2)
    len_2 = np.sum(np.sqrt(dy**2 + dz2**2))
    return len_1 + len_2

# --- 4. CONSTRAINT FUNCTION ---
def constraint_function(X, N_x, N_y, aspect_ratio, min_dist_sq):
    z1, z2, x_coords, y_coords, _, _, _, _ = unpack_variables(X, N_x, N_y, aspect_ratio)
    z1_col = z1.reshape(-1, 1)
    z2_row = z2.reshape(1, -1)
    x_col = x_coords.reshape(-1, 1)
    y_row = y_coords.reshape(1, -1)
    dist_sq_matrix = x_col**2 + y_row**2 + (z1_col - z2_row)**2
    return (dist_sq_matrix - min_dist_sq).flatten()

# --- 5. INITIAL GUESS ---
def get_initial_guess(N_x, N_y, aspect_ratio, initial_extent):
    initial_x_extent = initial_extent * np.sqrt(aspect_ratio)
    initial_y_extent = initial_extent / np.sqrt(aspect_ratio)
    x_coords = np.linspace(0, initial_x_extent, N_x)
    y_coords = np.linspace(0, initial_y_extent, N_y)
    z1_guess_full = 0.5 * np.cos(np.pi * x_coords / initial_x_extent / 2)
    z2_guess_full = -0.5 * np.cos(np.pi * y_coords / initial_y_extent / 2)
    z1_guess_internal = z1_guess_full[0:-1]
    z2_guess_internal = z2_guess_full[0:-1]
    return pack_variables(z1_guess_internal, z2_guess_internal, initial_extent)

# --- 6. MAIN EXECUTION ---
if __name__ == "__main__":
    # Create a folder for outputs if it doesn't exist
    if not os.path.exists("outputs"):
        os.makedirs("outputs")

    for ASPECT_RATIO in np.arange(1.0, 2.01, 0.01):
        print(f"\nAspect ratio set to {ASPECT_RATIO:.2f}")
        X0 = get_initial_guess(N_POINTS_X, N_POINTS_Y, ASPECT_RATIO, INITIAL_EXTENT)
        cons_fun_with_args = lambda X: constraint_function(X, N_POINTS_X, N_POINTS_Y, ASPECT_RATIO, MIN_DIST_SQ)
        constraints = [{'type': 'ineq', 'fun': cons_fun_with_args}]
        
        # Note: Updated bounds logic to match the length of X0 (1 extent + internal Zs)
        bounds = [(0.1, 20.0)] + [(-10.0, 10.0)] * (len(X0) - 1)

        obj_fun_with_args = lambda X: objective_function(X, N_POINTS_X, N_POINTS_Y, ASPECT_RATIO)

        print("Running optimizer (SLSQP)...")
        result = opt.minimize(
            obj_fun_with_args, X0, method='SLSQP', bounds=bounds,
            constraints=constraints, options={'disp': False, 'maxiter': 200}
        )

        if result.success:
            (z1_opt, z2_opt, x_coords_opt, y_coords_opt, dx_opt, dy_opt, x_ext_opt, y_ext_opt) = unpack_variables(result.x, N_POINTS_X, N_POINTS_Y, ASPECT_RATIO)
            
            filename = os.path.join("outputs", f"output_{ASPECT_RATIO:.2f}.csv")
            with open(filename, "w", newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(x_coords_opt)
                writer.writerow(z1_opt)
                writer.writerow(y_coords_opt)
                writer.writerow(z2_opt)
            print(f"Success! Saved to {filename}")
        else:
            print(f"Optimization FAILED for {ASPECT_RATIO:.2f}: {result.message}")