import argparse

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from datasets.bmnist import bmnist


class Encoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()
        
        #hard-coded input dim, for general purpose include in func args
        self.linear = nn.Linear(784, hidden_dim) 
        
        self.mu = nn.Linear(hidden_dim, z_dim)
        self.sigma = nn.Linear(hidden_dim, z_dim)

    def forward(self, input):
        """
        Perform forward pass of encoder.

        Returns mean and std with shape [batch_size, z_dim]. Make sure
        that any constraints are enforced.
        """
        inter = torch.tanh(self.linear(input)) 
        
        mean = torch.tanh(self.mu(inter))
        std = torch.sigmoid(self.sigma(inter))

        return mean, std


class Decoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()
        
        self.linear1 = nn.Linear(z_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 784)

    def forward(self, input):
        """
        Perform forward pass of encoder.

        Returns mean with shape [batch_size, 784].
        """
        inter = torch.tanh(self.linear1(input))
        mean = torch.sigmoid(self.linear2(inter))

        return mean


class VAE(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20, device="cpu"):
        super().__init__()

        self.z_dim = z_dim
        self.encoder = Encoder(hidden_dim, z_dim)
        self.decoder = Decoder(hidden_dim, z_dim)
        self.device = device

    def forward(self, input):
        """
        Given input, perform an encoding and decoding step and return the
        negative average elbo for the given batch.
        """
        mean, std = self.encoder(input)
        eps = self.sampleNormal()
        mean_g = self.decoder(std*eps + mean)
        
        gen1 = torch.sum(std,dim=1)
        gen2 = torch.diag(mean@mean.T)
        gen3 = torch.log(torch.prod(std,dim=1))
        Lgen = 0.5*(gen1 + gen2 - gen3 - self.z_dim ) #dim: B
#         rec1 = torch.tensor(np.random.binomial(size=(input.size(0),784),n=1, p=np.array(mean_g.detach())))
#         Lrec = -torch.sum(torch.log(rec1.type(torch.FloatTensor)),dim=1)
        Pgx = torch.zeros(mean_g.shape).to(self.device)
        Pgx[input==1] = mean_g[input==1]
        Pgx[input==0] = 1-mean_g[input==0]
        Lrec = -torch.sum(torch.log(Pgx),dim=1)      
        average_negative_elbo = torch.sum(Lgen + Lrec)/input.size(0)
        return average_negative_elbo

    def sample(self, n_samples):
        """
        Sample n_samples from the model. Return both the sampled images
        (from bernoulli) and the means for these bernoullis (as these are
        used to plot the data manifold).
        """
        #could do without for-loop
        sampled_ims = []
        im_means = []
        with torch.no_grad():
            for n in range(n_samples):
                eps = self.sampleNormal()
                mean_g = self.decoder(eps)
                sampled_im = torch.tensor(np.random.binomial(size=784,n=1, p=np.array(mean_g.cpu())))
                sampled_ims.append(sampled_im)
                im_means.append(mean_g)
        
        return sampled_ims, im_means
    
    def sampleNormal(self):
        #we were not allowed to use torch.distributions, so I didn't
        sample = np.random.multivariate_normal(np.zeros(self.z_dim),np.identity(self.z_dim))
        sample = torch.tensor(sample).type(torch.FloatTensor).to(self.device)
        return sample

def epoch_iter(model, data, optimizer, device):
    """
    Perform a single epoch for either the training or validation.
    use model.training to determine if in 'training mode' or not.

    Returns the average elbo for the complete epoch.
    """   
    average_epoch_elbo = 0
    for step, batch_input in enumerate(data):
        batch_input = batch_input.view(batch_input.size(0),-1).to(device)
        average_batch_elbo = model(batch_input)
        if model.training:
            optimizer.zero_grad()
            average_batch_elbo.backward()
            optimizer.step()
        average_epoch_elbo += average_batch_elbo
    average_epoch_elbo /= step
    return average_epoch_elbo


def run_epoch(model, data, optimizer, device):
    """
    Run a train and validation epoch and return average elbo for each.
    """
    traindata, valdata = data

    model.train()
    train_elbo = epoch_iter(model, traindata, optimizer, device)

    model.eval()
    val_elbo = epoch_iter(model, valdata, optimizer, device)

    return train_elbo, val_elbo


def save_elbo_plot(train_curve, val_curve, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(train_curve, label='train elbo')
    plt.plot(val_curve, label='validation elbo')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('ELBO')
    plt.tight_layout()
    plt.savefig(filename)


def main():
    
    device = torch.device(ARGS.device)
    data = bmnist()[:2]  # ignore test split
    model = VAE(z_dim=ARGS.zdim, device=device).to(device)
    torch.save(model.state_dict(), ARGS.model_file+"0.pt")
    if ARGS.use=="train":
        #which epochs to save; first and last already included
        save_inter_epochs = np.array([5,10,20]) #could make ARG
        results = open(ARGS.out, "w+")
        results.write(f"#learning_rate : {ARGS.lr}\n#z_dim : {ARGS.zdim}\n#epochs : {ARGS.epochs}\n")
        results.write("#epoch train_elbo val_elbo\n")
        
        optimizer = torch.optim.Adam(model.parameters(),lr=ARGS.lr)
        train_curve, val_curve = [], []
        for epoch in range(ARGS.epochs):
            if np.sum(epoch==save_inter_epochs):
                torch.save(model.state_dict(), ARGS.model_file+f"{epoch}.pt")
            elbos = run_epoch(model, data, optimizer, device)
            train_elbo, val_elbo = elbos
            train_curve.append(train_elbo)
            val_curve.append(val_elbo)
            print(f"[Epoch {epoch}] train elbo: {train_elbo} val_elbo: {val_elbo}")
            results.write(f"{epoch} {train_elbo} {val_elbo}\n")
        torch.save(model.state_dict(), ARGS.model_file+f"{ARGS.epochs}.pt")
        
        results.close()
    else:
        model.load_state_dict(torch.load(ARGS.load, map_location=lambda storage, loc: storage))
        model.eval()
        N = 40 #could make ARG
        B = 8 #could make ARG
        imgs, means = model.sample(int(N/2.))
        plt.figure(figsize=(B*1.5,(int(N/B)+1)*1.5))
        for i in range(int(N/2.)):
            plt.subplot(int(N/B)+1,B,i*2+1)
            plt.imshow(means[i].view(28,28),cmap= "gray")
            plt.axis('off')
            plt.subplot(int(N/B)+1,B,i*2+2)
            plt.imshow(imgs[i].view(28,28),cmap= "binary") 
            plt.axis('off')
        plt.show()
            

    # --------------------------------------------------------------------
    #  Add functionality to plot plot the learned data manifold after
    #  if required (i.e., if zdim == 2). You can use the make_grid
    #  functionality that is already imported.
    # --------------------------------------------------------------------

#     save_elbo_plot(train_curve, val_curve, 'elbo.pdf')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=40, type=int,
                        help='max number of epochs')
    parser.add_argument('--zdim', default=20, type=int,
                        help='dimensionality of latent space')
    parser.add_argument('--device', default="cuda:0", type=str,
                        help='device')
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='learning rate')
    parser.add_argument('--out', default="results.dat", type=str,
                        help='output file')
    parser.add_argument('--model_file', default="model", type=str,
                        help='output file without extension')
    parser.add_argument('--use', default="train", type=str,
                        help='model use')
    parser.add_argument('--load', default="model40.pt", type=str,
                        help='model to load')

    ARGS = parser.parse_args()

    main()
