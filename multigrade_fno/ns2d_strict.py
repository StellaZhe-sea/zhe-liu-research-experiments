"""
@author: Zongyi Li
This file is the Fourier Neural Operator for 2D problem such as the Navier-Stokes equation discussed in Section 5.3 in the [paper](https://arxiv.org/pdf/2010.08895.pdf),
which uses a recurrent structure to propagates in time.
"""


import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
from utilities3 import *

import operator
from functools import reduce
from functools import partial

from timeit import default_timer
import scipy.io

torch.manual_seed(0)
np.random.seed(0)

#Complex multiplication
def compl_mul2d(a, b):
    # PyTorch 2.x compatibility: retain the released source's explicit
    # real/imaginary parameter axis, but execute the same complex product
    # with the native complex einsum. This changes no trainable parameters.
    a_complex = torch.view_as_complex(a.contiguous())
    b_complex = torch.view_as_complex(b.contiguous())
    result = torch.einsum("bctq,dctq->bdtq", a_complex, b_complex)
    return torch.view_as_real(result)

################################################################
# fourier layer
################################################################

class SpectralConv2d_fast(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d_fast, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, 2))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, 2))

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        # PyTorch 2.x equivalent of torch.rfft(..., normalized=True,
        # onesided=True). Preserve the explicit real/imaginary axis used by
        # the paper-era parameters and multiplication code.
        x_ft = torch.view_as_real(torch.fft.rfft2(x, norm='ortho'))

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, 2, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(
            torch.view_as_complex(out_ft.contiguous()),
            s=(x.size(-2), x.size(-1)),
            norm='ortho'
        )
        return x

class SimpleBlock2d(nn.Module):
    def __init__(self, modes1, modes2, width):
        super(SimpleBlock2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x=64, y=64, c=12)
        output: the solution of the next timestep
        output shape: (batchsize, x=64, y=64, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.fc0 = nn.Linear(12, self.width)
        # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)

        self.conv0 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        self.bn0 = torch.nn.BatchNorm2d(self.width)
        self.bn1 = torch.nn.BatchNorm2d(self.width)
        self.bn2 = torch.nn.BatchNorm2d(self.width)
        self.bn3 = torch.nn.BatchNorm2d(self.width)


        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        batchsize = x.shape[0]
        size_x, size_y = x.shape[1], x.shape[2]

        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        x1 = self.conv0(x)
        x2 = self.w0(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = self.bn0(x1 + x2)
        x = F.relu(x)
        x1 = self.conv1(x)
        x2 = self.w1(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = self.bn1(x1 + x2)
        x = F.relu(x)
        x1 = self.conv2(x)
        x2 = self.w2(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = self.bn2(x1 + x2)
        x = F.relu(x)
        x1 = self.conv3(x)
        x2 = self.w3(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = self.bn3(x1 + x2)


        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

class Net2d(nn.Module):
    def __init__(self, modes, width):
        super(Net2d, self).__init__()

        """
        A wrapper function
        """

        self.conv1 = SimpleBlock2d(modes, modes, width)


    def forward(self, x):
        x = self.conv1(x)
        return x


    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))

        return c


################################################################
# configs
################################################################
TRAIN_PATH = os.getenv(
    'PAPER_FNO_NS_TRAIN',
    r'D:\Workspace_codex_Fno\fourier-neural-operator-main\data\NavierStokes_V1e-5_N1200_T20.mat'
)
TEST_PATH = os.getenv('PAPER_FNO_NS_TEST', TRAIN_PATH)

ntrain = int(os.getenv('PAPER_FNO_NTRAIN', '1000'))
ntest = int(os.getenv('PAPER_FNO_NTEST', '200'))

# The paper-reported parameter count corresponds to modes=8. The released
# 2021 GitHub source sets modes=12. Keep 8 as the strict default and allow a
# controlled modes-only ablation through the environment.
modes = int(os.getenv('PAPER_FNO_MODES', '8'))
width = int(os.getenv('PAPER_FNO_WIDTH', '20'))

batch_size = 20
batch_size2 = batch_size

epochs = int(os.getenv('PAPER_FNO_EPOCHS', '500'))
learning_rate = 0.0025
scheduler_step = 100
scheduler_gamma = 0.5

print(
    f'configuration: modes={modes}, width={width}, epochs={epochs}, '
    f'learning_rate={learning_rate}, '
    f'scheduler_step={scheduler_step}, scheduler_gamma={scheduler_gamma}'
)

path = 'ns_fourier_2d_rnn_V10000_T20_N'+str(ntrain)+'_ep' + str(epochs) + '_m' + str(modes) + '_w' + str(width)
path_model = 'model/'+path
path_train_err = 'results/'+path+'train.txt'
path_test_err = 'results/'+path+'test.txt'
path_image = 'image/'+path

runtime = np.zeros(2, )
t1 = default_timer()

sub = 1
S = 64
T_in = 10
T = int(os.getenv('PAPER_FNO_T', '10'))
step = 1

output_dir = Path(os.getenv(
    'PAPER_FNO_OUTPUT_DIR',
    f'results/ns2d_official_v3_T{T_in + T}_N{ntrain}_seed0'
))
output_dir.mkdir(parents=True, exist_ok=True)

################################################################
# load data
################################################################

def sample_count(file_path):
    reader = MatReader(file_path, to_torch=False)
    if reader.old_mat:
        return reader.data['u'].shape[0]
    count = reader.data['u'].shape[-1]
    reader.data.close()
    return count


def read_subset(file_path, start, stop):
    reader = MatReader(file_path)
    if reader.old_mat:
        u = reader.read_field('u')[start:stop, ::sub, ::sub, :T_in + T]
    else:
        # MATLAB v7.3 stores this dataset as (time, y, x, sample).
        # This slice is exactly the subset selected after read_field() in the
        # official script, without first loading the entire multi-GB file.
        u = reader.data['u'][:T_in + T, ::sub, ::sub, start:stop]
        u = np.transpose(u, axes=(3, 2, 1, 0))
        u = torch.from_numpy(np.ascontiguousarray(u, dtype=np.float32))
        reader.data.close()
    return u[..., :T_in], u[..., T_in:T_in + T]


train_total = sample_count(TRAIN_PATH)
test_total = sample_count(TEST_PATH)
if Path(TRAIN_PATH).resolve() == Path(TEST_PATH).resolve():
    assert ntrain + ntest <= train_total

train_a, train_u = read_subset(TRAIN_PATH, 0, ntrain)
test_a, test_u = read_subset(TEST_PATH, test_total - ntest, test_total)

print(f'train_target_shape: {tuple(train_u.shape)}')
print(f'test_target_shape: {tuple(test_u.shape)}')
assert (S == train_u.shape[-2])
assert (T == train_u.shape[-1])

train_a = train_a.reshape(ntrain,S,S,T_in)
test_a = test_a.reshape(ntest,S,S,T_in)

# pad the location (x,y)
gridx = torch.tensor(np.linspace(0, 1, S), dtype=torch.float)
gridx = gridx.reshape(1, S, 1, 1).repeat([1, 1, S, 1])
gridy = torch.tensor(np.linspace(0, 1, S), dtype=torch.float)
gridy = gridy.reshape(1, 1, S, 1).repeat([1, S, 1, 1])

train_a = torch.cat((train_a, gridx.repeat([ntrain,1,1,1]), gridy.repeat([ntrain,1,1,1])), dim=-1)
test_a = torch.cat((test_a, gridx.repeat([ntest,1,1,1]), gridy.repeat([ntest,1,1,1])), dim=-1)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False)

t2 = default_timer()

print(f'preprocessing_seconds: {t2 - t1:.4f}')
device = torch.device('cuda')

################################################################
# training and evaluation
################################################################

model = Net2d(modes, width).cuda()
# model = torch.load('model/ns_fourier_V100_N1000_ep100_m8_w20')

print(f'model_parameter_count: {model.count_params()}')
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)


myloss = LpLoss(size_average=False)
gridx = gridx.to(device)
gridy = gridy.to(device)

metrics = []
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2_step = 0
    train_l2_full = 0
    for xx, yy in train_loader:
        loss = 0
        xx = xx.to(device)
        yy = yy.to(device)

        for t in range(0, T, step):
            y = yy[..., t:t + step]
            im = model(xx)
            loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

            if t == 0:
                pred = im
            else:
                pred = torch.cat((pred, im), -1)

            xx = torch.cat((xx[..., step:-2], im,
                            gridx.repeat([batch_size, 1, 1, 1]), gridy.repeat([batch_size, 1, 1, 1])), dim=-1)

        train_l2_step += loss.item()
        l2_full = myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1))
        train_l2_full += l2_full.item()

        optimizer.zero_grad()
        loss.backward()
        # l2_full.backward()
        optimizer.step()

    test_l2_step = 0
    test_l2_full = 0
    with torch.no_grad():
        for xx, yy in test_loader:
            loss = 0
            xx = xx.to(device)
            yy = yy.to(device)

            for t in range(0, T, step):
                y = yy[..., t:t + step]
                im = model(xx)
                loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

                if t == 0:
                    pred = im
                else:
                    pred = torch.cat((pred, im), -1)

                xx = torch.cat((xx[..., step:-2], im,
                                gridx.repeat([batch_size, 1, 1, 1]), gridy.repeat([batch_size, 1, 1, 1])), dim=-1)


            test_l2_step += loss.item()
            test_l2_full += myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1)).item()

    t2 = default_timer()
    scheduler.step()
    print(
        f'epoch: {ep + 1}, epoch time: {t2 - t1:.4f}, '
        f'train_relative_l2: {train_l2_full / ntrain:.8e}, '
        f'test_relative_l2: {test_l2_full / ntest:.8e}, '
        f'learning_rate: {optimizer.param_groups[0]["lr"]:.8e}'
    )
    metrics.append([
        ep + 1,
        t2 - t1,
        train_l2_step / ntrain / (T / step),
        train_l2_full / ntrain,
        test_l2_step / ntest / (T / step),
        test_l2_full / ntest,
        optimizer.param_groups[0]['lr'],
    ])
# torch.save(model, path_model)

np.savetxt(
    output_dir / 'metrics.csv',
    np.asarray(metrics, dtype=np.float64),
    delimiter=',',
    header=(
        'epoch,epoch_seconds,train_step_relative_l2,train_full_relative_l2,'
        'test_step_relative_l2,test_full_relative_l2,learning_rate'
    ),
    comments=''
)

config = {
    'official_commit': '1c3c77fa79c16fb5c6ba29f78e050c4ebf785afa',
    'official_source': 'FNO-torch.1.6/fourier_2d_time.py',
    'configuration_variant': (
        'github_2021_released_source' if modes == 12
        else 'paper_reported_parameter_count'
    ),
    'seed': 0,
    'train_path': str(Path(TRAIN_PATH).resolve()),
    'test_path': str(Path(TEST_PATH).resolve()),
    'ntrain': ntrain,
    'ntest': ntest,
    'spatial_resolution': S,
    'input_timesteps': T_in,
    'predicted_timesteps': T,
    'total_timesteps': T_in + T,
    'batch_size': batch_size,
    'epochs': epochs,
    'learning_rate': learning_rate,
    'step_size': scheduler_step,
    'gamma': scheduler_gamma,
    'modes': modes,
    'width': width,
    'weight_decay': 1e-4,
    'paper_reported_parameter_count': 414517,
    'released_source_modes': 12,
    'source_discrepancy': 'paper parameter count requires modes=8; released 2021 source sets modes=12',
    'ablation': 'modes-only comparison; all other settings are held fixed',
    'test_mode': 'train mode, exactly as official source (no model.eval call)',
    'fft_compatibility': 'torch.rfft/irfft replaced by torch.fft; explicit real/imaginary parameter axis retained',
    'complex_multiplication_compatibility': 'four real einsums replaced by one mathematically equivalent native-complex einsum',
}
(output_dir / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')

torch.save(
    {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'config': config,
    },
    output_dir / 'model_checkpoint.pt'
)

# Preserve the official test behavior: the model remains in train mode, so
# BatchNorm uses each test batch's statistics.
predictions = []
with torch.no_grad():
    for xx, _ in test_loader:
        xx = xx.to(device)
        rollout = []
        for _ in range(0, T, step):
            im = model(xx)
            rollout.append(im)
            xx = torch.cat((xx[..., step:-2], im,
                            gridx.repeat([batch_size, 1, 1, 1]),
                            gridy.repeat([batch_size, 1, 1, 1])), dim=-1)
        predictions.append(torch.cat(rollout, dim=-1).cpu())

torch.save(
    {
        'x_test': test_a.detach().cpu(),
        'y_test': test_u.detach().cpu(),
        'prediction': torch.cat(predictions, dim=0),
    },
    output_dir / 'test_predictions.pt'
)


# pred = torch.zeros(test_u.shape)
# index = 0
# test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=1, shuffle=False)
# with torch.no_grad():
#     for x, y in test_loader:
#         test_l2 = 0;
#         x, y = x.cuda(), y.cuda()
#
#         out = model(x)
#         out = y_normalizer.decode(out)
#         pred[index] = out
#
#         test_l2 += myloss(out.view(1, -1), y.view(1, -1)).item()
#         print(index, test_l2)
#         index = index + 1

# scipy.io.savemat('pred/'+path+'.mat', mdict={'pred': pred.cpu().numpy()})
