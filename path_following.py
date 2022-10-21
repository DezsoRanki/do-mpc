import numpy as np

# Add do_mpc to path. This is not necessary if it was installed via pip.
import sys
sys.path.append('../../')

# Import do_mpc package:
import do_mpc
from casadi import *

model_type = 'continuous' # either 'discrete' or 'continuous'
model = do_mpc.model.Model(model_type)

# states
theta = model.set_variable(var_type='_x', var_name='theta', shape=(1,1))
x_path = model.set_variable(var_type='_x', var_name='x_path', shape=(1,1))
y_path = model.set_variable(var_type='_x', var_name='y_path', shape=(1,1))
phi_path = model.set_variable(var_type='_x', var_name='phi_path', shape=(1,1))
x = model.set_variable(var_type='_x', var_name='x', shape=(1,1))
y = model.set_variable(var_type='_x', var_name='y', shape=(1,1))
phi = model.set_variable(var_type='_x', var_name='phi', shape=(1,1))
v = model.set_variable(var_type='_x', var_name='v', shape=(1,1))
v_ref = model.set_variable(var_type='_x', var_name='v_ref', shape=(1,1))
delta_s = model.set_variable(var_type='_x', var_name='delta_s', shape=(1,1))
delta_s_ref = model.set_variable(var_type='_x', var_name='delta_s_ref', shape=(1,1))

# inputs
_theta = model.set_variable(var_type='_u', var_name='_theta')
a_ref = model.set_variable(var_type='_u', var_name='a_ref')
w_s_ref = model.set_variable(var_type='_u', var_name='w_s_ref')

# parameters
k0 = -0.44
k1 = 928
k2 = -2.39
lr = 1.51
lf = 1.13
T_v = 1.2
T_delta = 0.15
_theta_ref = 10

delta_a = (delta_s - k0) / (k1 + k2 * v)
beta = np.arctan((lr / (lf + lr)) * np.tan(delta_a))

# dynamics
model.set_rhs('theta', _theta)
model.set_rhs('x_path', _theta)
model.set_rhs('y_path', 2 * theta * _theta)
model.set_rhs('phi_path', (1 / (1 + 4 * theta * theta)) * 2 * theta * _theta)
# model.set_rhs('theta', _theta)
# model.set_rhs('x_path', (1 + 2 * theta + 3 * theta * theta) * _theta) # x=theta+theta^2+theta^3
# model.set_rhs('y_path', (1 - 2 * theta + 3 * theta * theta - 5 * theta * theta * theta * theta) * _theta)
# model.set_rhs('phi_path', np.arctan(((1 - 2 * theta + 6 * theta * theta - 5 * theta * theta * theta * theta) * _theta) / ((1 + 2 * theta + 3 * theta * theta) * _theta)))
model.set_rhs('x', v * np.cos(phi + beta))
model.set_rhs('y', v * np.sin(phi + beta))
model.set_rhs('phi', (v / lr) * np.sin(beta))
model.set_rhs('v', 1 / T_v * (v_ref - v))
model.set_rhs('v_ref', a_ref)
model.set_rhs('delta_s', 1 / T_delta * (delta_s_ref - delta_s))
model.set_rhs('delta_s_ref', w_s_ref)

h1 = x + lf * np.cos(phi)
h2 = y + lf * np.sin(phi)
h3 = phi

model.setup()

mpc = do_mpc.controller.MPC(model)
setup_mpc = {
    'n_horizon': 30,
    't_step': 0.1,
    'n_robust': 1,
    'store_full_solution': True,
}
mpc.set_param(**setup_mpc)

# object function
p_x = 2.5e4
p_y = 2.5e4
q_x = 2.5e4
q_y = 2.5e4
q_phi = 2700
p_phi = 2700
r_a = 700
r_w = 0.2
r_theta = 1000
p_a = 10e5
q_a = 10e5

mterm = p_x * ((x + lf * np.cos(phi) - x_path) ** 2) + p_y * ((y + lf * np.sin(phi) - y_path) ** 2) + p_phi * ((phi - phi_path) ** 2) + \
        p_a * ((v * ((v / lr) * np.sin(np.arctan((lr / (lf + lr)) * np.tan((delta_s - k0) / (k1 + k2 * v)))))) ** 2)
lterm = q_x * ((x + lf * np.cos(phi) - x_path) ** 2) + q_y * ((y + lf * np.sin(phi) - y_path) ** 2) + q_phi * ((phi - phi_path) ** 2) + \
        q_a * ((v * ((v / lr) * np.sin(np.arctan((lr / (lf + lr)) * np.tan((delta_s - k0) / (k1 + k2 * v)))))) ** 2) + \
        r_a * (a_ref ** 2) + r_w * (w_s_ref ** 2) + r_theta * ((_theta - _theta_ref) ** 2)

mpc.set_objective(mterm=mterm, lterm=lterm)

# constraints
mpc.bounds['lower','_x', 'v_ref'] = 0
mpc.bounds['lower','_x', 'delta_s_ref'] = -460
mpc.bounds['lower','_x', 'theta'] = 0

mpc.bounds['upper','_x', 'v_ref'] = 60
mpc.bounds['upper','_x', 'delta_s_ref'] = 460
mpc.bounds['upper','_x', 'theta'] = 60

mpc.bounds['lower','_u', 'a_ref'] = -3
mpc.bounds['lower','_u', 'w_s_ref'] = -250
mpc.bounds['lower','_u', '_theta'] = 0

mpc.bounds['upper','_u', 'a_ref'] = 2
mpc.bounds['upper','_u', 'w_s_ref'] = 250
mpc.bounds['upper','_u', '_theta'] = 60

mpc.setup()

simulator = do_mpc.simulator.Simulator(model)
simulator.set_param(t_step = 0.1)

simulator.setup()

x0 = np.array([1, 1, 1, np.arctan(2), 1, 1, np.arctan(2), 1, 1, 1, 1]).reshape(-1,1)

simulator.x0 = x0
mpc.x0 = x0

mpc.set_initial_guess()

import matplotlib.pyplot as plt
import matplotlib as mpl
# Customizing Matplotlib:
mpl.rcParams['font.size'] = 18
mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['axes.grid'] = True

mpc_graphics = do_mpc.graphics.Graphics(mpc.data)
sim_graphics = do_mpc.graphics.Graphics(simulator.data)

simulator.reset_history()
simulator.x0 = x0
mpc.reset_history()

for i in range(15):
    u0 = mpc.make_step(x0)
    x0 = simulator.make_step(u0)
    print(i)

path_error = (x0[1][0] - x0[4][0]) ** 2 + (x0[2][0] - x0[5][0]) ** 2 + (x0[2][0] - x0[6][0]) ** 2
print(x0)
print(path_error)

fig, ax = plt.subplots(1, sharex=True, figsize=(16,9))
ax.plot(sim_graphics.data['_x', 'x_path'],sim_graphics.data['_x', 'y_path'])
ax.plot(sim_graphics.data['_x', 'x'],sim_graphics.data['_x', 'y'])
plt.show()