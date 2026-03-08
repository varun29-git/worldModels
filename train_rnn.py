import torch
import math
from model import MDN_RNN


def mdn_loss(pi, mu, sigma, z_next):

    z_next = z_next.unsqueeze(2)

    log_prob = -0.5 * (
        ((z_next - mu) / sigma) ** 2 +
        2 * torch.log(sigma) +
        math.log(2 * math.pi)
    )

    log_prob = log_prob.sum(dim=-1)

    log_prob = log_prob + torch.log(pi + 1e-8)

    log_prob = torch.logsumexp(log_prob, dim=-1)

    loss = -log_prob.mean()

    return loss


def train(model, vae, dataloader, optimizer, device):

    model.train()

    total_loss = 0

    for obs, a in dataloader:

        obs = obs.to(device)
        a = a.to(device)
        
        # pass sequence of observations through VAE to get latents z
        batch_size, seq_len, c, h, w = obs.size()
        obs_flat = obs.view(batch_size * seq_len, c, h, w)
        
        with torch.no_grad():
            mu, logvar = vae.encode(obs_flat)
            z_flat = vae.reparameterize(mu, logvar)
            
        z = z_flat.view(batch_size, seq_len, -1)

        z_input = z[:, :-1]
        z_next = z[:, 1:]
        a_input = a[:, :-1]

        optimizer.zero_grad()

        pi, mu, sigma, _ = model(z_input, a_input)

        loss = mdn_loss(pi, mu, sigma, z_next)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    z_dim = 32
    action_dim = 3
    hidden_dim = 256
    num_gaussians = 5

    # ------------------ Added Dataloader ------------------
    import pickle
    import torchvision.transforms as transforms
    from torch.utils.data import Dataset, DataLoader
    from model.vae import VAE
    
    class SequenceDataset(Dataset):
        def __init__(self, dataset_path, seq_len=300):
            with open(dataset_path, "rb") as f:
                self.data = pickle.load(f)
            self.seq_len = seq_len
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((64,64)),
                transforms.ToTensor()
            ])
            
        def __len__(self):
            return len(self.data)
            
        def __getitem__(self, idx):
            episode = self.data[idx]
            obs = episode["observations"][:self.seq_len]
            actions = episode["actions"][:self.seq_len]
            
            # pad if sequence is too short
            if len(obs) < self.seq_len:
                # In CarRacing we usually have 1000 steps, but collect_data limits episodes if done early.
                pad_len = self.seq_len - len(obs)
                obs.extend([obs[-1]] * pad_len) 
                actions.extend([actions[-1]] * pad_len)
                
            obs_tensor = torch.stack([self.transform(o) for o in obs])
            import numpy as np
            action_tensor = torch.tensor(np.array(actions), dtype=torch.float32)
            return obs_tensor, action_tensor

    print("Loading dataset...")
    dataset = SequenceDataset("carRacing_dataset.pkl", seq_len=100)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    print("Loading VAE...")
    vae = VAE(input_shape=(3,64,64), latent_dim=z_dim).to(device)
    vae.load_state_dict(torch.load("vae.pt", map_location=device, weights_only=True))
    vae.eval()
    # -----------------------------------------------------

    model = MDN_RNN(z_dim, action_dim, hidden_dim, num_gaussians).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)

    num_epochs = 100

    for epoch in range(num_epochs):

        loss = train(model, vae, dataloader, optimizer, device)

        print(f"Epoch {epoch} | Loss: {loss:.4f}")
        scheduler.step(loss)

    torch.save(model.state_dict(), "rnn.pt")