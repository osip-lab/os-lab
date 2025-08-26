import numpy as np
import matplotlib.pyplot as plt
# n_H = 1.3
# n_L = 1.2
# m = 20
# n_sub = 1.76
# n_M = 1
# phi_M = 0.1
# d_L = 1
#
# snell_argument_entrance = n_M * np.sin(phi_M)
#
# cos_phi_H = np.sqrt(1 - snell_argument_entrance ** 2 / n_H ** 2)
# cos_phi_L = np.sqrt(1 - snell_argument_entrance ** 2 / n_L ** 2)
#
# n_H_s = n_H * cos_phi_H
# n_L_s = n_L * cos_phi_L
#
# n_H_p = n_H / cos_phi_H
# n_L_p = n_L / cos_phi_L
#
# Y_perpendicular = (n_H / n_L) ** 2 * m * n_H**2 / n_sub
#
# Y_H_s = (n_H_s / n_L_s) ** (2 * m) * n_H_s**2 / n_sub
# Y_H_p = (n_H_p / n_L_p) ** (2 * m) * n_H_p**2 / n_sub
#
# Y_L_s = (n_L_s / n_H_s) ** (2 * m) * n_L_s**2 / n_sub
# Y_L_p = (n_L_p / n_H_p) ** (2 * m) * n_L_p**2 / n_sub
#
# R_numerator = n_M - Y_perpendicular


# %%
def cos_theta_layer(theta_vacuum, n):
    return np.sqrt(1 - (np.sin(theta_vacuum) / n) ** 2)

def k_z_layer(theta_vacuum, n, k_0):
    return n * k_0 * cos_theta_layer(theta_vacuum, n)

def phase_layer(theta_vacuum, n, k_0, d):
    k_z_layer_value = k_z_layer(theta_vacuum, n, k_0)
    return k_z_layer_value * d

def eta_layer(n, theta_vacuum, polarization: str):
    cos_theta_layer_value = cos_theta_layer(theta_vacuum, n)
    if polarization == 's':
        return n * cos_theta_layer_value
    elif polarization == 'p':
        return n / cos_theta_layer_value
    else:
        raise ValueError("Polarization must be either 's' or 'p'.")


def layer_characteristic_matrix(n, theta_vacuum, d, k_0, polarization: str):
    # Compute phase and eta arrays (may be scalar or array)
    phase = phase_layer(theta_vacuum, n, k_0, d)
    eta = eta_layer(n, theta_vacuum, polarization)

    # Ensure broadcastable shapes
    phase, eta = np.broadcast_arrays(phase, eta)

    # Construct output matrix of shape (..., 2, 2)
    cos_phase = np.cos(phase)
    sin_term = 1j * np.sin(phase) / eta
    product_term = 1j * np.sin(phase) * eta

    top_row = np.stack([cos_phase, sin_term], axis=-1)
    bottom_row = np.stack([product_term, cos_phase], axis=-1)
    matrix = np.stack([top_row, bottom_row], axis=-2)

    return matrix

def admittance_of_a_matrix(matrix, theta_vacuum, n_substrate, polarization: str):
    eta_substrate = eta_layer(n_substrate, theta_vacuum, polarization)
    output_field = np.array([1, eta_substrate])  # shape: (2,)

    # Expand to shape (..., 2, 1) for broadcasting
    output_field = output_field.reshape((1,) * (matrix.ndim - 2) + (2, 1))

    # Multiply each 2x2 matrix with the output_field column vector
    input_field = matrix @ output_field  # shape: (..., 2, 1)

    admittance = input_field[..., 1, 0] / input_field[..., 0, 0]
    return admittance

def reflectance_of_addmitance(addmitance):
    r = (1 - addmitance) / (1 + addmitance)
    R = np.abs(r) ** 2
    return R

# %%

def matrix_power_batch(matrices, power):
    """
    Raise a batch of 2x2 matrices to a given power.
    Input:
        matrices: (..., 2, 2) array
        power: int
    Output:
        (..., 2, 2) array
    """
    shape = matrices.shape[:-2]
    result = np.empty_like(matrices)
    for index in np.ndindex(shape):
        result[index] = np.linalg.matrix_power(matrices[index], power)
    return result

lambda_0 = 1064e-9  # Wavelength in meters
k_0 = 2 * np.pi / lambda_0  # Wave number in vacuum
theta_vacuum = np.deg2rad(5)
n_substrate = 1.76

n_1 = 1.5
n_2 = 1.2
polarization = 's'
m_stacks = 10

d_1_relative_perturbation = np.linspace(-0.05, 0.05, 100)
d_2_relative_perturbation = np.linspace(-0.05, 0.05, 100)

d_1 = np.pi / (2 * k_z_layer(theta_vacuum, n_1, k_0)) * (1 + d_1_relative_perturbation)
d_2 = np.pi / (2 * k_z_layer(theta_vacuum, n_2, k_0)) * (1 + d_2_relative_perturbation)

D_1, D_2 = np.meshgrid(d_1, d_2, indexing='ij')

matrix_1 = layer_characteristic_matrix(n_1, theta_vacuum, D_1, k_0, polarization)
matrix_2 = layer_characteristic_matrix(n_2, theta_vacuum, D_2, k_0, polarization)

product = matrix_1 @ matrix_2  # shape: (100, 100, 2, 2)
product_powered = matrix_power_batch(product, m_stacks)  # elementwise matrix^m_stacks
matrix = product_powered @ matrix_1  # shape: (100, 100, 2, 2)

admittance = admittance_of_a_matrix(matrix, theta_vacuum, n_substrate, polarization)

R = reflectance_of_addmitance(admittance)

# Plot the results and put the min of 1-R in the title:

plt.figure(figsize=(10, 6))
plt.imshow(R, extent=(d_2_relative_perturbation[0], d_2_relative_perturbation[-1],
                       d_1_relative_perturbation[0], d_1_relative_perturbation[-1]),
           origin='lower', aspect='auto', cmap='viridis')
plt.colorbar(label='Reflectance (R)')
plt.xlabel('d_2 Relative Perturbation')
plt.ylabel('d_1 Relative Perturbation')
plt.title(f'Max Reflectance: 1 - {np.min(1-R):.4f}')
plt.grid(False)
plt.show()







