# ----------------------------- Imports -----------------------------
import os, glob, numpy as np
import mne
import torch, torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

# ----------------------------- PhysioNet Loader -----------------------------
def extract_physionet_subject(subject_folder, tmin=0.0, tmax=3.2, l_freq=0.5, h_freq=50.0, sfreq_target=160.0):
    edf_files = sorted(glob.glob(os.path.join(subject_folder, '*.edf')))
    all_X, all_y, ch_names = [], [], None
    nt = int((tmax - tmin) * sfreq_target)
    for ef in edf_files:
        raw = mne.io.read_raw_edf(ef, preload=True, verbose=False)
        raw.filter(l_freq, h_freq, verbose=False)
        if abs(raw.info['sfreq'] - sfreq_target) > 1e-3:
            raw.resample(sfreq_target, npad='auto', verbose=False)
        picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
        try:
            events, event_id = mne.events_from_annotations(raw, verbose=False)
        except Exception:
            events = []
        for ev in events:
            samp = int(ev[0])
            start = int(samp + tmin * raw.info['sfreq'])
            stop  = int(samp + tmax * raw.info['sfreq'])
            if start < 0 or stop > raw.n_times:
                continue
            data = raw.get_data(picks=picks, start=start, stop=stop)
            if data.shape[1] != nt:
                if data.shape[1] > nt:
                    data = data[:, :nt]
                else:
                    pad = np.zeros((len(picks), nt - data.shape[1]))
                    data = np.hstack([data, pad])
            code = ev[2]
            lbl = 0 if (int(code) % 2 == 0) else 1
            all_X.append(data)
            all_y.append(lbl)
        if ch_names is None:
            ch_names = [raw.ch_names[i] for i in picks]
    if len(all_X) == 0:
        return np.zeros((0, len(picks) if picks is not None else 0, nt)), np.zeros((0,), dtype=int), ch_names
    return np.stack(all_X, axis=0).astype(np.float32), np.array(all_y, dtype=int), ch_names

# ----------------------------- EEG-ARNN Model -----------------------------
class TFEMBlock(nn.Module):
    def __init__(self, nch, F=16, k_t=15, pool=False, pool_k=4, drop=0.25):
        super().__init__()
        pad_t = (k_t - 1) // 2
        self.conv = nn.Conv2d(1, F, kernel_size=(1, k_t), padding=(0, pad_t))
        self.bn = nn.BatchNorm2d(F)
        self.pw = nn.Conv2d(F, 1, kernel_size=1)
        self.pool = nn.AvgPool2d((1, pool_k)) if pool else None
        self.elu = nn.ELU()
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        b, nch, t = x.shape
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.bn(x)
        x = self.elu(x)
        x = self.pw(x)
        if self.pool:
            x = self.pool(x)
        x = self.drop(x)
        return x.squeeze(1)

class CARM(nn.Module):
    def __init__(self, Wref, tdim, drop=0.25):
        super().__init__()
        self.Wref = Wref
        self.Theta = nn.Parameter(torch.randn(tdim, tdim) * 0.01)
        self.elu = nn.ELU()
        self.drop = nn.Dropout(drop)
    def forward(self,x):
        h = torch.einsum('ij,bjf->bif', self.Wref, x)
        out = torch.einsum('bif,fg->big', h, self.Theta)
        out = self.elu(out)
        out = self.drop(out)
        return out

class EEG_ARNN(nn.Module):
    def __init__(self, nch, T0, ncls=2, F=16, pool_k=4, rho=0.001):
        super().__init__()
        self.nch=nch; self.T0=T0; self.ncls=ncls; self.rho=rho
        W0 = torch.ones(nch,nch) - torch.eye(nch)
        Wt = W0 + torch.eye(nch)
        D = Wt.sum(dim=1)
        Dinv = torch.diag(1.0 / torch.sqrt(D + 1e-12))
        self.W = nn.Parameter(Dinv @ Wt @ Dinv)
        # TFEM + CARM blocks
        self.tf1 = TFEMBlock(nch, F=F, k_t=15, pool=False)
        self.c1  = CARM(self.W, tdim=T0)
        self.tf2 = TFEMBlock(nch, F=F, k_t=15, pool=True, pool_k=pool_k)
        T2 = T0 // pool_k
        self.c2  = CARM(self.W, tdim=T2)
        self.tf3 = TFEMBlock(nch, F=F, k_t=15, pool=True, pool_k=pool_k)
        T3 = T2 // pool_k
        self.c3  = CARM(self.W, tdim=T3)
        # Fusion + classifier
        self.fuse = nn.Conv2d(1,16,kernel_size=(nch,1))
        self.bn = nn.BatchNorm2d(16)
        self.drop = nn.Dropout(0.25)
        self.elu = nn.ELU()
        self.fc = nn.Linear(16*T3, ncls)
    def forward(self,x):
        x = self.tf1(x); x = self.c1(x)
        x = self.tf2(x); x = self.c2(x)
        x = self.tf3(x); x = self.c3(x)
        x = x.unsqueeze(1)
        x = self.fuse(x)
        x = self.bn(x)
        x = self.elu(x)
        x = self.drop(x)
        x = x.squeeze(2)
        b,oc,t = x.shape
        x = x.view(b, oc*t)
        return self.fc(x)

# ----------------------------- Train / Eval Helpers -----------------------------
def train_epoch(model,loader,opt,device,crit,rho):
    model.train(); total=0; n=0
    for xb,yb in loader:
        xb=xb.to(device); yb=yb.to(device)
        opt.zero_grad()
        out=model(xb); loss=crit(out,yb)
        loss.backward(); opt.step()
        with torch.no_grad():
            if model.W.grad is not None:
                model.W.data = (1.0 - rho) * model.W.data - rho * model.W.grad.data
                model.W.grad.zero_()
        total += loss.item() * xb.size(0); n += xb.size(0)
    return total / max(1,n)

def eval_model(model,loader,device):
    model.eval(); preds=[]; ys=[]
    with torch.no_grad():
        for xb,yb in loader:
            xb=xb.to(device); yb=yb.to(device)
            out=model(xb)
            preds.append(out.argmax(dim=1).cpu().numpy())
            ys.append(yb.cpu().numpy())
    return np.concatenate(preds), np.concatenate(ys)

# ----------------------------- AS Top-k Selection -----------------------------
def select_topk_channels_AS(adj_matrix, k):
    node_scores = torch.sum(torch.abs(adj_matrix), dim=1) + torch.diag(adj_matrix)
    topk_indices = torch.topk(node_scores, k=k).indices.tolist()
    return topk_indices

# ----------------------------- Main Pipeline -----------------------------
DATA_ROOT = 'bciciv2a/files'
SUBJECT = 'S001'
SUBJ_FOLDER = os.path.join(DATA_ROOT, SUBJECT)
assert os.path.exists(SUBJ_FOLDER), f"Folder not found: {SUBJ_FOLDER}"

X, y, chs = extract_physionet_subject(SUBJ_FOLDER)
print("Extracted X:", X.shape, "y:", y.shape, "channels:", len(chs))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
EPOCHS, BATCH, LR = 5, 16, 1e-3
k_top = 20  # top-k channels

trained_models = []

# ----------------------------- 10-fold CV training -----------------------------
# ----------------------------- 10-fold CV training + evaluation -----------------------------
full_accs, topk_accs = [], []

for fold,(train_idx,val_idx) in enumerate(skf.split(X,y),1):
    print(f"\n--- Fold {fold} ---")
    Xtr, ytr = X[train_idx], y[train_idx]
    Xval, yval = X[val_idx], y[val_idx]

    tr_loader = DataLoader(TensorDataset(torch.tensor(Xtr,dtype=torch.float32),torch.tensor(ytr)),
                           batch_size=BATCH, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(Xval,dtype=torch.float32),torch.tensor(yval)),
                            batch_size=BATCH, shuffle=False)

    # ---------------- Full-channel EEG-ARNN ----------------
    model = EEG_ARNN(nch=X.shape[1], T0=X.shape[2], ncls=len(np.unique(y))).to(device)
    params = [p for n,p in model.named_parameters() if n!='W' and p.requires_grad]
    opt = torch.optim.Adam(params, lr=LR)
    crit = nn.CrossEntropyLoss()

    for ep in range(1,EPOCHS+1):
        loss = train_epoch(model, tr_loader, opt, device, crit, rho=model.rho)
        if ep%max(1,EPOCHS//5)==0 or ep==1:
            print(f"Epoch {ep}/{EPOCHS} -> train_loss={loss:.4f}")

    preds, ytrue = eval_model(model, val_loader, device)
    acc = accuracy_score(ytrue,preds)
    full_accs.append(acc)
    print(f"Fold {fold} full-channel accuracy: {acc*100:.2f}%")

    # ---------------- Select top-k channels ----------------
    W_star = model.W.detach().cpu()
    topk_channels = select_topk_channels_AS(W_star, k_top)
    print(f"Fold {fold} top-{k_top} channels:", topk_channels)

    # ---------------- Reduced dataset ----------------
    Xtr_topk = Xtr[:, topk_channels, :]
    Xval_topk = Xval[:, topk_channels, :]

    tr_loader_topk = DataLoader(TensorDataset(torch.tensor(Xtr_topk,dtype=torch.float32),torch.tensor(ytr)),
                                batch_size=BATCH, shuffle=True)
    val_loader_topk = DataLoader(TensorDataset(torch.tensor(Xval_topk,dtype=torch.float32),torch.tensor(yval)),
                                 batch_size=BATCH, shuffle=False)

    # ---------------- Top-k EEG-ARNN ----------------
    topk_model = EEG_ARNN(nch=k_top, T0=Xtr_topk.shape[2], ncls=len(np.unique(y))).to(device)
    params = [p for n,p in topk_model.named_parameters() if n!='W' and p.requires_grad]
    opt = torch.optim.Adam(params, lr=LR)
    crit = nn.CrossEntropyLoss()

    for ep in range(1,EPOCHS+1):
        loss = train_epoch(topk_model, tr_loader_topk, opt, device, crit, rho=topk_model.rho)
        if ep%max(1,EPOCHS//5)==0 or ep==1:
            print(f"Top-k Epoch {ep}/{EPOCHS} -> train_loss={loss:.4f}")

    preds_topk, ytrue_topk = eval_model(topk_model, val_loader_topk, device)
    acc_topk = accuracy_score(ytrue_topk,preds_topk)
    topk_accs.append(acc_topk)
    print(f"Fold {fold} top-{k_top} EEG-ARNN accuracy: {acc_topk*100:.2f}%")

# ----------------------------- Summary -----------------------------
print("\n=== Cross-Validation Summary ===")
for f,(fa,ta) in enumerate(zip(full_accs, topk_accs),1):
    print(f"Fold {f}: Full-channel={fa*100:.2f}%, Top-{k_top}={ta*100:.2f}%")
print(f"Average full-channel accuracy: {np.mean(full_accs)*100:.2f}%")
print(f"Average top-{k_top} accuracy: {np.mean(topk_accs)*100:.2f}%")
