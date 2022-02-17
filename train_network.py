import os
import time
import numpy as np
import netCDF4 as nc
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from scipy import interpolate
import nfft

torch.manual_seed(1033)
np.random.seed(1033)

DATA_DIR = 'ML_convection_data_cheyenne_Jan_2020'
N_FILES = 100
X_SIZ, Y_SIZ, Z_SIZ = 36, 90, 48
N_LEVELS = 30
INPUT_VARS = ['TABS', 'q']
OUTPUT_VARS = ['DQP']
STDEV_OUT = np.sqrt(14.93)

INTERP_Z_INPUT, INTERP_Z_OUTPUT = False, False
new_zs = np.linspace(40, 13407.63, 30)
INTERP_PRESSURE_INPUT, INTERP_PRESSURE_OUTPUT = True, True
new_pressures = np.linspace(1000, 130, 30)

NUFFT_INPUTS = True
N_INPUT_FCS = 30
NUFFT_OUTPUTS = False
N_OUTPUT_FCS = 30

INPUT_PC_CONFIG = 'both'  # one of ['var1', 'var2', 'both', 'none']
N_INPUT_PCS = [20, 30]
OUTPUT_PC_CONFIG = 'var1'  # one of ['var1', 'none']
N_OUTPUT_PCS = 10

CRITERION = nn.MSELoss()

suffix = 'nc220stdinputstdoutput-zinterp.lin.40.13407_63.30'


# read in the data
nc_files = os.listdir(DATA_DIR)[:N_FILES]
ncs = [nc.Dataset('{}/{}'.format(DATA_DIR, nc_file)) for nc_file in nc_files]

# construct input/output pairs
inputs_raw = np.empty((N_FILES, X_SIZ, Y_SIZ, N_LEVELS, len(INPUT_VARS)))
outputs_raw = np.empty((N_FILES, X_SIZ, Y_SIZ, N_LEVELS, len(OUTPUT_VARS)))
inputs_full = np.empty((N_FILES, X_SIZ, Y_SIZ, N_LEVELS, len(INPUT_VARS)))
outputs_full = np.empty((N_FILES, X_SIZ, Y_SIZ, N_LEVELS, len(OUTPUT_VARS)))
if INTERP_PRESSURE_INPUT or INTERP_PRESSURE_OUTPUT:
    pressures_full = np.empty((N_FILES, X_SIZ, Y_SIZ, N_LEVELS))
equator_dists = np.empty((N_FILES, X_SIZ * Y_SIZ))
zs = ncs[0]['z'][:].data[:N_LEVELS]
for nc_i, netcdf in enumerate(ncs):
    equator_dists[nc_i] = np.tile(np.abs(netcdf['y'][:].data-8634000), X_SIZ)
    if INTERP_PRESSURE_INPUT or INTERP_PRESSURE_OUTPUT:
        pressures_full[nc_i, :, :] = (netcdf['p'][:].reshape(Z_SIZ, 1, 1) + netcdf['PP'][:]/100)[:N_LEVELS,:,:].T.reshape(X_SIZ, Y_SIZ, N_LEVELS)
    for input_i, input_var in enumerate(INPUT_VARS):
        if input_var == 'q':
            raw_input = (netcdf['Q'][:] + netcdf['QN'][:])[:N_LEVELS,:,:].T.reshape(X_SIZ, Y_SIZ, N_LEVELS)
        else:
            raw_input = netcdf[input_var][:][:N_LEVELS,:,:].T.reshape(X_SIZ, Y_SIZ, N_LEVELS)
        inputs_raw[nc_i, :, :, :, input_i] = raw_input
    if INTERP_Z_INPUT:
        interp_f = interpolate.interp1d(zs, raw_input, kind='linear', fill_value='extrapolate')
        inputs_full[nc_i, :, :, :, input_i] = interp_f(new_zs)
    elif INTERP_PRESSURE_INPUT:
        for x in range(X_SIZ):
            for y in range(Y_SIZ):
                interp_f = interpolate.interp1d(pressures_full[nc_i, x, y], raw_input[x, y], kind='linear', fill_value='extrapolate')
                inputs_full[nc_i, x, y, :, input_i] = interp_f(new_pressures)
    for output_i, output_var in enumerate(OUTPUT_VARS):
        raw_output = netcdf[output_var][:][:N_LEVELS,:,:].T.reshape(X_SIZ, Y_SIZ, N_LEVELS)
        outputs_raw[nc_i, :, :, :, output_i] = raw_output
        if INTERP_Z_OUTPUT:
            interp_f = interpolate.interp1d(zs, raw_output, kind='linear', fill_value='extrapolate')
            outputs_full[nc_i, :, :, :, output_i] = interp_f(new_zs)
        elif INTERP_PRESSURE_OUTPUT:
            for x in range(X_SIZ):
                for y in range(Y_SIZ):
                    interp_f = interpolate.interp1d(pressures_full[nc_i, x, y], raw_output[x, y], kind='linear', fill_value='extrapolate')
                    outputs_full[nc_i, x, y, :, output_i] = interp_f(new_pressures)
    netcdf.close()
inputs_r = np.transpose(inputs_raw, (0, 1, 2, 4, 3)).reshape(-1, len(INPUT_VARS), N_LEVELS)
outputs_r = np.transpose(outputs_raw, (0, 1, 2, 4, 3)).reshape(-1, len(OUTPUT_VARS), N_LEVELS)
inputs = np.transpose(inputs_full, (0, 1, 2, 4, 3)).reshape(-1, len(INPUT_VARS), N_LEVELS)
outputs = np.transpose(outputs_full, (0, 1, 2, 4, 3)).reshape(-1, len(OUTPUT_VARS), N_LEVELS)
pressures = pressures_full.reshape(-1, N_LEVELS)

# for now, concatenate variables together
inputs = inputs.reshape(-1, len(INPUT_VARS) * N_LEVELS)
outputs = outputs.reshape(-1, len(OUTPUT_VARS) * N_LEVELS)

# add distance to equator as an input
equator_dists = equator_dists.reshape(-1, 1)
inputs = np.hstack((inputs, equator_dists))

# convert to tensors
inputs = torch.as_tensor(inputs)
outputs = torch.as_tensor(outputs)

if NUFFT_INPUTS:
    # we must scale the sample points to within [-0.5, 0.5]
    zs = np.interp(zs, (min(zs), max(zs)), (-0.5, +0.5))
    def input_NUFFT(inputs, zs, fcs):
        # apply NUFFT separately to each variable, taking only one of the symmetric halves
        # do not apply NUFFT to equator distance input, if used
        ft = [np.hstack((nfft.nfft_adjoint(zs, inp[:N_LEVELS], fcs)[:fcs//2+1], nfft.nfft_adjoint(zs, inp[N_LEVELS:2*N_LEVELS], fcs)[:fcs//2+1])) for inp in inputs]
        # encode each complex component as i, j representing the real and imaginary components
        ft = torch.view_as_real(torch.as_tensor(np.array(ft))).numpy().reshape(-1, len(INPUT_VARS) * (fcs//2+1) * 2)
        # stack equator distance back on
        return np.hstack((ft, inputs[:, -1].reshape(-1, 1)))
    inputs = input_NUFFT(inputs, zs, N_INPUT_FCS)
if NUFFT_OUTPUTS:
    # we must scale the sample points to within [-0.5, 0.5]
    zs = np.interp(zs, (min(zs), max(zs)), (-0.5, +0.5))
    def output_NUFFT(outputs, zs, fcs):
        ft = [nfft.nfft_adjoint(zs, out, fcs)[:fcs//2+1] for out in outputs]
        ft = torch.view_as_real(torch.as_tensor(np.array(ft))).numpy().reshape(-1, len(OUTPUT_VARS) * (fcs//2+1) * 2)
        return ft
    outputs = output_NUFFT(outputs, zs, N_OUTPUT_FCS)

# split into train, validation, and test sets
train_prop, val_prop, test_prop = 0.90, 0.05, 0.05
train_idxs, val_idxs, test_idxs = [], [], []
for idx in range(inputs.shape[0]):
    randnum = np.random.rand()
    if randnum < train_prop:
        train_idxs.append(idx)
    elif randnum < train_prop + val_prop:
        val_idxs.append(idx)
    else:
        test_idxs.append(idx)
train_idxs, val_idxs, test_idxs = np.array(train_idxs), np.array(val_idxs), np.array(test_idxs)
train_inputs, train_outputs = inputs[train_idxs], outputs[train_idxs]
val_inputs, val_outputs = inputs[val_idxs], outputs[val_idxs]
test_inputs, test_outputs = inputs[test_idxs], outputs[test_idxs]

n_train, n_val, n_test = train_idxs.size, val_idxs.size, test_idxs.size

# standardize the inputs
input_scaler = StandardScaler()
train_inputs = input_scaler.fit_transform(train_inputs)
val_inputs = input_scaler.transform(val_inputs)
test_inputs = input_scaler.transform(test_inputs)
# standardize the outputs
output_scaler = StandardScaler()
train_outputs = output_scaler.fit_transform(train_outputs) * STDEV_OUT
val_outputs = output_scaler.transform(val_outputs) * STDEV_OUT
test_outputs = output_scaler.transform(test_outputs) * STDEV_OUT

# perform PCA on inputs
if INPUT_PC_CONFIG == 'both':
    input_pcas = [PCA(n_components=n_pcs) for n_pcs in N_INPUT_PCS]
    train_inputs = np.hstack([input_pca.fit_transform(train_input) for input_pca, train_input in zip(input_pcas, np.split(train_inputs[:,:-1], len(INPUT_VARS), axis=1))] +
                             [train_inputs[:,-1:]])
    val_inputs = np.hstack([input_pca.transform(val_input) for input_pca, val_input in zip(input_pcas, np.split(val_inputs[:,:-1], len(INPUT_VARS), axis=1))] +
                           [val_inputs[:,-1:]])
    test_inputs = np.hstack([input_pca.transform(test_input) for input_pca, test_input in zip(input_pcas, np.split(test_inputs[:,:-1], len(INPUT_VARS), axis=1))] +
                            [test_inputs[:,-1:]])
elif INPUT_PC_CONFIG == 'var1':
    input_pca = PCA(n_components=N_INPUT_PCS)
    train_inputs = np.hstack([input_pca.fit_transform(train_inputs[:,:N_LEVELS])] + [train_inputs[:,N_LEVELS:]])
    val_inputs = np.hstack([input_pca.transform(val_inputs[:,:N_LEVELS])] + [val_inputs[:,N_LEVELS:]])
    test_inputs = np.hstack([input_pca.transform(test_inputs[:,:N_LEVELS])] + [test_inputs[:,N_LEVELS:]])
elif INPUT_PC_CONFIG == 'var2':
    input_pca = PCA(n_components=N_INPUT_PCS)
    train_inputs = np.hstack([train_inputs[:,:N_LEVELS]] + [input_pca.fit_transform(train_inputs[:,N_LEVELS:2*N_LEVELS])] + [train_inputs[:,2*N_LEVELS:]])
    val_inputs = np.hstack([val_inputs[:,:N_LEVELS]] + [input_pca.transform(val_inputs[:,N_LEVELS:2*N_LEVELS])] + [val_inputs[:,2*N_LEVELS:]])
    test_inputs = np.hstack([test_inputs[:,:N_LEVELS]] + [input_pca.transform(test_inputs[:,N_LEVELS:2*N_LEVELS])] + [test_inputs[:,2*N_LEVELS:]])
# perform PCA on outputs
if OUTPUT_PC_CONFIG == 'var1':
    output_pca = PCA(n_components=N_OUTPUT_PCS)
    train_outputs = output_pca.fit_transform(train_outputs)
    val_outputs = output_pca.transform(val_outputs)
    test_outputs = output_pca.transform(test_outputs)

# convert all to tensors
train_inputs = torch.as_tensor(train_inputs).float()
val_inputs = torch.as_tensor(val_inputs).float()
test_inputs = torch.as_tensor(test_inputs).float()
train_outputs = torch.as_tensor(train_outputs).float()
val_outputs = torch.as_tensor(val_outputs).float()
test_outputs = torch.as_tensor(test_outputs).float()


# model definition
class ClimateNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(ClimateNet, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.linears = nn.Sequential(
            nn.Linear(self.input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.output_size),
        )

    def forward(self, x):
        return self.linears(x)


def eval_on(eval_inputs, eval_outputs, climateNet):
    with torch.no_grad():
        eval_preds = climateNet(eval_inputs)
        eval_loss = CRITERION(eval_preds, eval_outputs)
        eval_r2 = r2_score(eval_preds.flatten(), eval_outputs.flatten(), multioutput='variance_weighted')
        if OUTPUT_PC_CONFIG == 'var1':
            eval_preds = output_pca.inverse_transform(eval_preds)
            eval_outputs = output_pca.inverse_transform(eval_outputs)
        unscaled_preds = output_scaler.inverse_transform(eval_preds/STDEV_OUT)
        unscaled_targs = output_scaler.inverse_transform(eval_outputs/STDEV_OUT)
        unscaled_r2 = r2_score(unscaled_preds.flatten(), unscaled_targs.flatten(), multioutput='variance_weighted')
        return eval_preds, unscaled_preds, eval_loss.item(), eval_r2, unscaled_r2


def train_epoch(climateNet, climateNet_optimizer, loader, scheduler):
    climateNet_optimizer.zero_grad()
    train_loss = 0
    for step, (batch_x, batch_y) in enumerate(loader):
        prediction = climateNet(batch_x)
        loss = CRITERION(prediction, batch_y)
        climateNet_optimizer.zero_grad()
        loss.backward()
        climateNet_optimizer.step()
        climateNet_optimizer.zero_grad()
        train_loss += loss.data.numpy()
        scheduler.step()
    return train_loss.item()


def train_model(climateNet, suffix, batch_size=1024, learning_rate=1e-7, epochs=7, min_lr=2e-4, max_lr=2e-3, step_size_up=4000, save_every=1):
    start_time = time.time()
    train_losses, val_losses = [], []

    torch_dataset = torch.utils.data.TensorDataset(train_inputs, train_outputs)
    loader = torch.utils.data.DataLoader(dataset=torch_dataset, batch_size=batch_size, shuffle=True)

    climateNet_optimizer = torch.optim.Adam(climateNet.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CyclicLR(climateNet_optimizer, base_lr=min_lr, max_lr=max_lr, step_size_up=step_size_up, cycle_momentum=False)

    for epoch in range(epochs+epochs-2):
        train_loss = train_epoch(climateNet, climateNet_optimizer, loader, scheduler)
        if epoch % save_every == 0:
            torch.save(climateNet.state_dict(), 'saved_models/climateNet-{}-epoch{}.pt'.format(suffix, epoch+1))
        val_preds, unscaled_preds, val_loss, val_r2, unscaled_r2 = eval_on(val_inputs, val_outputs, climateNet)

        if INTERP_Z_OUTPUT or INTERP_PRESSURE_OUTPUT:
            # re-interpolate output back to original coordinates
            reinterp_predictions = []
            for i in range(len(unscaled_preds)):
                interp_f = interpolate.interp1d(new_zs if INTERP_Z_OUTPUT else new_pressures, unscaled_preds[i], kind='linear', fill_value='extrapolate')
                reinterp_predictions.append(interp_f(zs) if INTERP_Z_OUTPUT else interp_f(pressures[val_idxs[i]]))
            reinterp_predictions = np.array(reinterp_predictions)
            reinterp_r2 = r2_score(reinterp_predictions.flatten(), outputs_r[val_idxs].flatten(), multioutput='variance_weighted')

        print('Current Validation Loss: {:e}'.format(val_loss))
        # print('R^2: {:.3f}'.format(val_r2))
        print('R^2: {:.3f}'.format(unscaled_r2))
        if INTERP_Z_OUTPUT or INTERP_PRESSURE_OUTPUT:
            print('Re-interpolated R^2: {:.3f}'.format(reinterp_r2))
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print('Finished epoch {} ({}) Train Loss: {:e}'.format(epoch+1, '{:02d}m {:02d}s'.format(*divmod(time.time()-start_time, 60)), train_loss))

        if epoch == epochs - 1:
            scheduler = torch.optim.lr_scheduler.CyclicLR(climateNet_optimizer, base_lr=min_lr/10, max_lr=max_lr/10, step_size_up=step_size_up, cycle_momentum=False)

    return train_losses, val_losses


# define and train the model
climateNet = ClimateNet(input_size=train_inputs.shape[1], output_size=train_outputs.shape[1])
train_losses, val_losses = train_model(climateNet, suffix, epochs=7, save_every=1)

