# train_conditional_gan.py
import os, cv2, numpy as np, argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
import torch.nn.functional as F

# ----------------
# Utils
# ----------------
def load_feature_extractor(path, device):
    model = models.resnet18(pretrained=False)
    model.fc = nn.Identity()
    model.load_state_dict(torch.load(path, map_location=device))
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model

def tensor_to_bgr(img_tensor):
    # img_tensor: 1x3xHxW in [-1,1]
    img = img_tensor.detach().cpu().numpy()[0]
    img = (np.transpose(img, (1,2,0)) + 1.0) * 127.5
    img = np.clip(img, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

# ----------------
# Simple architectures
# ----------------
class Generator(nn.Module):
    def __init__(self, z_dim, feat_dim, ngf=64, out_channels=3, img_size=128):
        super().__init__()
        self.z_dim = z_dim
        self.feat_dim = feat_dim
        self.proj = nn.Linear(feat_dim, 256)
        input_dim = z_dim + 256
        self.net = nn.Sequential(
            nn.Linear(input_dim, ngf*8*4*4),
            nn.BatchNorm1d(ngf*8*4*4),
            nn.ReLU(True),
            View((-1, ngf*8, 4, 4)),
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1), # 8x8
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1), # 16x16
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1),   # 32x32
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, out_channels, 4, 2, 1), # 64x64
            nn.Tanh()
        )
    def forward(self, z, feat):
        feat_proj = torch.relu(self.proj(feat))
        x = torch.cat([z, feat_proj], dim=1)
        return self.net(x)

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, ndf=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, ndf, 4, 2, 1), # 64->32
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf*2, 4, 2, 1), #32->16
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1), #16->8
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*4, 1, 4, 1, 0),
        )
    def forward(self, x):
        out = self.net(x)
        return out.view(x.size(0), -1)

class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape
    def forward(self, x):
        return x.view(*self.shape)

# ----------------
# Data loader
# ----------------
def get_dataloaders(data_dir, img_size=128, batch_size=32, num_workers=4):
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])
    tf_gen = transforms.Compose([
        transforms.Resize((img_size,img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])
    train_ds = datasets.ImageFolder(os.path.join(data_dir,'train'), transform=tf)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return train_loader, train_ds.classes

# ----------------
# Training loop
# ----------------
def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_loader, classes = get_dataloaders(args.data_dir, img_size=args.img_size, batch_size=args.batch_size, num_workers=args.num_workers)
    feat_model = load_feature_extractor(args.feature_extractor, device)
    feat_dim = 512  # resnet18 penultimate dimension

    G = Generator(z_dim=args.z_dim, feat_dim=feat_dim, img_size=args.img_size).to(device)
    D = Discriminator(in_channels=3).to(device)
    optG = optim.Adam(G.parameters(), lr=args.lr, betas=(0.5,0.999))
    optD = optim.Adam(D.parameters(), lr=args.lr, betas=(0.5,0.999))

    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()

    real_label = 1.0; fake_label = 0.0

    # Precompute class mean features if requested
    if args.use_class_mean:
        print("Computing class mean features...")
        class_feats = {i: [] for i in range(len(classes))}
        with torch.no_grad():
            for imgs, labels in tqdm(train_loader, desc="Feat collect"):
                imgs = imgs.to(device)
                feats = feat_model(imgs)  # B x feat_dim
                for i,l in enumerate(labels):
                    class_feats[l.item()].append(feats[i].cpu().numpy())
        class_mean = {}
        for k,v in class_feats.items():
            if len(v)>0:
                class_mean[k] = torch.tensor(np.stack(v).mean(axis=0), dtype=torch.float32, device=device)
            else:
                class_mean[k] = torch.zeros(feat_dim, device=device)
    else:
        class_mean = None

    for epoch in range(args.epochs):
        pbar = tqdm(train_loader, desc=f"Epoch{epoch}")
        for imgs, labels in pbar:
            batch_size = imgs.size(0)
            imgs_real = imgs.to(device)  # normalized for feature extractor
            labels = labels.to(device)

            # --------------------
            # Train D: real vs fake
            # --------------------
            D.zero_grad()
            out_real = D(denorm_to_tanh(imgs_real))  # map to [-1,1] space expected by generator
            real_targets = torch.full((batch_size, out_real.size(1)), real_label, device=device)
            lossD_real = bce(out_real, real_targets)

            # sample features: either class mean or features from real images
            with torch.no_grad():
                if args.use_class_mean:
                    feats_target = torch.stack([class_mean[int(l.item())] for l in labels], dim=0)
                else:
                    feats_target = feat_model(imgs_real)

            z = torch.randn(batch_size, args.z_dim, device=device)
            fake = G(z, feats_target)
            out_fake = D(fake.detach())
            fake_targets = torch.full((batch_size, out_fake.size(1)), fake_label, device=device)
            lossD_fake = bce(out_fake, fake_targets)

            lossD = (lossD_real + lossD_fake) * 0.5
            lossD.backward()
            optD.step()

            # --------------------
            # Train G
            # --------------------
            G.zero_grad()
            out_fake_forG = D(fake)
            gen_targets = torch.full((batch_size, out_fake_forG.size(1)), real_label, device=device)
            lossG_gan = bce(out_fake_forG, gen_targets)

            # feature matching loss
            with torch.no_grad():
                # compute features of real images (if not using class mean)
                if args.use_class_mean:
                    feats_target = torch.stack([class_mean[int(l.item())] for l in labels], dim=0)
                else:
                    feats_target = feat_model(imgs_real)

            # compute features of fake images (need to re-normalize fake back to classifier range)
            fake_for_feat = fake.clone()
            # fake in [-1,1]; convert to [0,1] then normalize
            fake_for_feat = ((fake_for_feat + 1.0) / 2.0).clamp(0,1)
            # normalize to ImageNet stats
            mean = torch.tensor([0.485,0.456,0.406], device=device).view(1,3,1,1)
            std  = torch.tensor([0.229,0.224,0.225], device=device).view(1,3,1,1)
            fake_norm = (fake_for_feat - mean) / std
            feats_fake = feat_model(fake_norm)
            loss_feat = mse(feats_fake, feats_target)

            lossG = lossG_gan + args.lambda_feat * loss_feat
            lossG.backward()
            optG.step()

            pbar.set_postfix({"lossD": lossD.item(), "lossG": lossG.item(), "loss_feat": loss_feat.item()})

        # Save samples at epoch end
        os.makedirs(args.out_dir, exist_ok=True)
        with torch.no_grad():
            # pick a few labels and generate
            for cls in range(min(5, len(classes))):
                if args.use_class_mean:
                    feat = class_mean[cls].unsqueeze(0)
                else:
                    # sample a single real image from train_loader of that class (cheap: reuse earlier batch)
                    feat = feat_model(imgs_real[:1])
                z = torch.randn(1, args.z_dim, device=device)
                fake_sample = G(z, feat)
                img = tensor_to_bgr( to_display(fake_sample) )
                cv2.imwrite(os.path.join(args.out_dir, f"epoch{epoch}_class{cls}.png"), img)

        # Save model
        torch.save(G.state_dict(), os.path.join(args.out_dir, f"G_epoch{epoch}.pt"))
        torch.save(D.state_dict(), os.path.join(args.out_dir, f"D_epoch{epoch}.pt"))

# Utility helpers
def denorm_to_tanh(x):
    # Input x normalized as ImageNet ([0,1]-based normalized). Convert to [-1,1] range for discriminator input convention.
    # x is normalized by mean/std; we must reverse normalization to [0,1] then map to [-1,1]
    mean = torch.tensor([0.485,0.456,0.406], device=x.device).view(1,3,1,1)
    std  = torch.tensor([0.229,0.224,0.225], device=x.device).view(1,3,1,1)
    x = x * std + mean
    x = x.clamp(0,1)
    x = x * 2.0 - 1.0
    return x

def to_display(x):
    # ensure x is in [-1,1] and size Bx3xHxW
    return x

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--feature_extractor', required=True, help='path to feature_extractor.pt from classifier step')
    parser.add_argument('--out_dir', default='gan_out')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--img_size', type=int, default=128)
    parser.add_argument('--z_dim', type=int, default=100)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--lambda_feat', type=float, default=10.0)
    parser.add_argument('--use_class_mean', action='store_true', help='use precomputed class mean features instead of per-sample features')
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()
    train(args)
