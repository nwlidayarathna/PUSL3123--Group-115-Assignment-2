import argparse, json, random, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parent
CLASSES = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
SEED = 42

def seed():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

def transform(augment=False):
    ops = [transforms.Grayscale(), transforms.Resize((48,48))]
    if augment: ops += [transforms.RandomHorizontalFlip(), transforms.RandomRotation(10)]
    return transforms.Compose(ops + [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

class EmotionCNN(nn.Module):
    def __init__(self, variant='baseline'):
        super().__init__()
        width = 32 if variant != 'wider' else 48
        layers = []
        channels = 1
        for out in (width, width*2, width*4):
            layers.append(nn.Conv2d(channels, out, 3, padding=1))
            if variant == 'batchnorm': layers.append(nn.BatchNorm2d(out))
            layers.extend([nn.ReLU(), nn.MaxPool2d(2)])
            channels = out
        layers += [nn.Flatten()]
        if variant == 'dropout': layers.append(nn.Dropout(0.4))
        layers.append(nn.Linear(channels*6*6, 7))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

def data_sets():
    train_dir, test_dir = ROOT/'data'/'train', ROOT/'data'/'test'
    if not train_dir.is_dir() or not test_dir.is_dir():
        raise SystemExit('Missing data/train or data/test. Place the DLE FER2013 class folders there; see README.md.')
    base = datasets.ImageFolder(train_dir, transform=transform())
    test = datasets.ImageFolder(test_dir, transform=transform())
    if len(base.classes) != 7 or base.class_to_idx != test.class_to_idx:
        raise ValueError('Train and test must have the same seven class-folder names.')
    # Stratified split to preserve rare classes in the validation set.
    rng = np.random.default_rng(SEED); train_ids=[]; val_ids=[]
    targets = np.array(base.targets)
    for label in range(7):
        ids = rng.permutation(np.flatnonzero(targets == label)); n = max(1, int(len(ids)*0.2))
        if len(ids) < 2: raise ValueError('Each training class must contain at least two images.')
        val_ids.extend(ids[:n].tolist()); train_ids.extend(ids[n:].tolist())
    return base, test, train_ids, val_ids

def metrics(model, loader, device, loss_fn):
    model.eval(); matrix=torch.zeros((7,7),dtype=torch.int64); loss=0; count=0
    with torch.no_grad():
        for x,y in loader:
            x,y=x.to(device),y.to(device); out=model(x); loss+=loss_fn(out,y).item()*len(y); count+=len(y)
            for a,b in zip(y.cpu(),out.argmax(1).cpu()): matrix[a,b]+=1
    m=matrix.float(); tp=m.diag(); precision=tp/m.sum(0).clamp(min=1); recall=tp/m.sum(1).clamp(min=1)
    return {'loss':loss/count, 'accuracy':(tp.sum()/m.sum()).item(), 'macro_precision':precision.mean().item(), 'macro_recall':recall.mean().item(), 'per_class_precision':precision.tolist(), 'per_class_recall':recall.tolist(), 'confusion_matrix':matrix.tolist()}

def run(epochs, batch, resume=False):
    base,test,train_ids,val_ids=data_sets(); device=torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print('Device:',device,'Classes:',base.classes,'Split:',len(train_ids),len(val_ids),len(test),flush=True)
    output=ROOT/'results'; output.mkdir(exist_ok=True); candidates=[]
    for variant in ['baseline','dropout','augmentation','batchnorm','wider']:
        result_file = output/f'{variant}.json'
        checkpoint_file = output/f'{variant}.pt'
        if resume and result_file.exists() and checkpoint_file.exists():
            record = json.loads(result_file.read_text())
            if len(record.get('history', [])) != epochs:
                raise ValueError(f'{variant}: saved epoch count differs; use the original --epochs or run without --resume.')
            candidates.append(record)
            print(f'Reusing completed {variant} experiment', flush=True)
            continue
        seed(); train_data=datasets.ImageFolder(ROOT/'data'/'train',transform=transform(variant=='augmentation'))
        train=DataLoader(Subset(train_data,train_ids),batch_size=batch,shuffle=True,generator=torch.Generator().manual_seed(SEED))
        val=DataLoader(Subset(base,val_ids),batch_size=batch)
        model=EmotionCNN(variant).to(device); opt=torch.optim.Adam(model.parameters(),lr=0.001); criterion=nn.CrossEntropyLoss(); best=-1; history=[]
        for epoch in range(epochs):
            model.train(); total=0; count=0
            for x,y in train:
                x,y=x.to(device),y.to(device); opt.zero_grad(); out=model(x); loss=criterion(out,y); loss.backward(); opt.step(); total+=loss.item()*len(y); count+=len(y)
            v=metrics(model,val,device,criterion); history.append({'epoch':epoch+1,'training_loss':total/count,'validation_loss':v['loss'],'validation_accuracy':v['accuracy']})
            print(f"{variant} epoch {epoch+1}/{epochs} train loss {total/count:.4f} val accuracy {v['accuracy']:.4f}",flush=True)
            if v['accuracy']>best:
                best=v['accuracy']; torch.save({'state':model.state_dict(),'variant':variant,'classes':base.classes,'epoch':epoch+1},output/f'{variant}.pt'); best_metrics=v
        record={'variant':variant,'parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'validation':best_metrics,'history':history}
        (output/f'{variant}.json').write_text(json.dumps(record,indent=2)); candidates.append(record)
    chosen=max(candidates,key=lambda x:x['validation']['accuracy']); checkpoint=torch.load(output/f"{chosen['variant']}.pt",map_location=device,weights_only=True)
    model=EmotionCNN(checkpoint['variant']).to(device); model.load_state_dict(checkpoint['state']); final=metrics(model,DataLoader(test,batch_size=batch),device,nn.CrossEntropyLoss())
    (output/'final_test.json').write_text(json.dumps({'selected_variant':chosen['variant'],'classes':base.classes,'test':final},indent=2))
    print('Selected by validation:',chosen['variant'],'test:',final,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--epochs',type=int,default=10); p.add_argument('--batch-size',type=int,default=64); p.add_argument('--resume',action='store_true',help='Reuse completed experiments with matching epoch count'); a=p.parse_args()
    if a.epochs<1 or a.batch_size<1: p.error('epochs and batch size must be positive')
    run(a.epochs,a.batch_size,a.resume)
