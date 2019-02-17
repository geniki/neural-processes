import os
import argparse
import random

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

parser = argparse.ArgumentParser(description='Neural Processes (NP) for MNIST image completion')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training')
parser.add_argument('--epochs', type=int, default=300, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--r_dim', type=int, default=128, metavar='N',
                    help='dimension of r, the hidden representation of the context points')
parser.add_argument('--z_dim', type=int, default=128, metavar='N',
                    help='dimension of z, the global latent variable')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
random.seed(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)


def get_context_idx(N):
    # generate the indeces of the N context points in a flattened image
    idx = random.sample(range(0, 784), N)
    idx = torch.tensor(idx, device=device)
    return idx


def generate_grid(h, w):
    rows = torch.linspace(0, 1, h, device=device)
    cols = torch.linspace(0, 1, w, device=device)
    grid = torch.stack([cols.repeat(h, 1).t().contiguous().view(-1), rows.repeat(w)], dim=1)
    grid = grid.unsqueeze(0)
    return grid


def idx_to_y(idx, data):
    # get the [0;1] pixel intensity at each index
    y = torch.index_select(data, dim=1, index=idx)
    return y


def idx_to_x(idx, batch_size):
    # From flat idx to 2d coordinates of the 28x28 grid. E.g. 35 -> (1, 7)
    # Equivalent to np.unravel_index()
    x = torch.index_select(x_grid, dim=1, index=idx)
    x = x.expand(batch_size, -1, -1)
    return x


class NP(nn.Module):
    def __init__(self, args):
        super(NP, self).__init__()
        self.r_dim = args.r_dim
        self.z_dim = args.z_dim

        self.h_1 = nn.Linear(3, 400)
        self.h_2 = nn.Linear(400, 400)
        self.h_3 = nn.Linear(400, self.r_dim)

        self.r_to_z_mean = nn.Linear(self.r_dim, self.z_dim)
        self.r_to_z_logvar = nn.Linear(self.r_dim, self.z_dim)

        self.g_1 = nn.Linear(self.z_dim + 2, 400)
        self.g_2 = nn.Linear(400, 400)
        self.g_3 = nn.Linear(400, 400)
        self.g_4 = nn.Linear(400, 400)
        self.g_5 = nn.Linear(400, 1)

    def h(self, x_y):
        x_y = F.relu(self.h_1(x_y))
        x_y = F.relu(self.h_2(x_y))
        x_y = F.relu(self.h_3(x_y))
        return x_y

    def aggregate(self, r):
        return torch.mean(r, dim=1)

    def reparameterise(self, z):
        mu, logvar = z
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z_sample = eps.mul(std).add_(mu)
        z_sample = z_sample.unsqueeze(1).expand(-1, 784, -1)
        return z_sample

    def g(self, z_sample, x_target):
        z_x = torch.cat([z_sample, x_target], dim=2)
        input = F.relu(self.g_1(z_x))
        input = F.relu(self.g_2(input))
        input = F.relu(self.g_3(input))
        input = F.relu(self.g_4(input))
        y_hat = torch.sigmoid(self.g_5(input))
        return y_hat

    def xy_to_z_params(self, x, y):
        x_y = torch.cat([x, y], dim=2)
        r_i = self.h(x_y)
        r = self.aggregate(r_i)

        mu = self.r_to_z_mean(r)
        logvar = self.r_to_z_logvar(r)

        return mu, logvar

    def forward(self, x_context, y_context, x_all=None, y_all=None):
        z_context = self.xy_to_z_params(x_context, y_context)  # (mu, logvar) of z
        if self.training:  # loss function will try to keep z_context close to z_all
            z_all = self.xy_to_z_params(x_all, y_all)
        else:  # at test time we don't have the image so we use only the context
            z_all = z_context

        z_sample = self.reparameterise(z_all)

        # reconstruct the whole image including the provided context points
        x_target = x_grid.expand(y_context.shape[0], -1, -1)
        y_hat = self.g(z_sample, x_target)

        return y_hat, z_all, z_context


def kl_div_gaussians(mu_q, logvar_q, mu_p, logvar_p):
    var_p = torch.exp(logvar_p)
    kl_div = (torch.exp(logvar_q) + (mu_q - mu_p) ** 2) / var_p \
             - 1.0 \
             + logvar_p - logvar_q
    kl_div = 0.5 * kl_div.sum()
    return kl_div


def np_loss(y_hat, y, z_all, z_context):
    BCE = F.binary_cross_entropy(y_hat, y, reduction="sum")
    KLD = kl_div_gaussians(z_all[0], z_all[1], z_context[0], z_context[1])
    return BCE + KLD


model = NP(args).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
x_grid = generate_grid(28, 28)
os.makedirs("results/", exist_ok=True)


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (y_all, _) in enumerate(train_loader):
        batch_size = y_all.shape[0]
        y_all = y_all.to(device).view(batch_size, -1, 1)

        N = random.randint(1, 784)  # number of context points
        context_idx = get_context_idx(N)
        x_context = idx_to_x(context_idx, batch_size)
        y_context = idx_to_y(context_idx, y_all)
        x_all = x_grid.expand(batch_size, -1, -1)

        optimizer.zero_grad()
        y_hat, z_all, z_context = model(x_context, y_context, x_all, y_all)

        loss = np_loss(y_hat, y_all, z_all, z_context)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(y_all), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                       loss.item() / len(y_all)))
    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (y_all, _) in enumerate(test_loader):
            y_all = y_all.to(device).view(y_all.shape[0], -1, 1)
            batch_size = y_all.shape[0]

            N = 300
            context_idx = get_context_idx(N)
            x_context = idx_to_x(context_idx, batch_size)
            y_context = idx_to_y(context_idx, y_all)

            y_hat, z_all, z_context = model(x_context, y_context)
            test_loss += np_loss(y_hat, y_all, z_all, z_context).item()

            if i == 0:  # save PNG of reconstructed examples
                plot_Ns = [10, 100, 300, 784]
                num_examples = min(batch_size, 16)
                for N in plot_Ns:
                    recons = []
                    context_idx = get_context_idx(N)
                    x_context = idx_to_x(context_idx, batch_size)
                    y_context = idx_to_y(context_idx, y_all)
                    for d in range(5):
                        y_hat, _, _ = model(x_context, y_context)
                        recons.append(y_hat[:num_examples])
                    recons = torch.cat(recons).view(-1, 1, 28, 28).expand(-1, 3, -1, -1)
                    background = torch.tensor([0., 0., 1.], device=device)
                    background = background.view(1, -1, 1).expand(num_examples, 3, 784).contiguous()
                    context_pixels = y_all[:num_examples].view(num_examples, 1, -1)[:, :, context_idx]
                    context_pixels = context_pixels.expand(num_examples, 3, -1)
                    background[:, :, context_idx] = context_pixels
                    comparison = torch.cat([background.view(-1, 3, 28, 28),
                                            recons])
                    save_image(comparison.cpu(),
                               'results/ep_' + str(epoch) +
                               '_cps_' + str(N) + '.png', nrow=num_examples)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


for epoch in range(1, args.epochs + 1):
    train(epoch)
    test(epoch)
