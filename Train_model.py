import torch
import glob
import gc
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
from config import config as cfg, forward
from Data_handler import Tiny

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create data loader
tiny = Tiny(cfg.dataset_dir, mode='model_train', transform=cfg.AUGMENT)
data_loader = DataLoader(tiny, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=4)

# Create the model and load most recent ckpt model if available
res_model = models.resnet50(pretrained=True)
num_features = res_model.fc.in_features
res_model.fc = nn.Linear(num_features, cfg.NUM_FEATURES)
res_model = res_model.to(device)
latest_model = sorted(glob.glob('%s/*.ckpt' % cfg.out_model_dir))[-1]
try:
    res_model.load_state_dict(torch.load(latest_model))
except:
    print('didnt properly load, or model doesn\'t exist')
    exit(0)


# Loss and optimizer
# opt = torch.optim.SGD(MODEL.parameters(), lr = 0.0001, momentum=0.9)
opt = torch.optim.Adam(res_model.parameters(), lr=cfg.LEARNING_RATE)

TOTAL_STEP = len(data_loader)
CURR_LR = cfg.LEARNING_RATE

for epoch in range(cfg.NUM_EPOCHS):
    for i, (query, pos, neg) in enumerate(data_loader):
        gc.collect()  # probably not needed, but security
        print(i, end='\r')
        query = forward(query, res_model, device)
        pos = forward(pos)
        neg = forward(neg)

        # compute loss https://pytorch.org/docs/0.3.1/nn.html?highlight=tripletmarginloss
        triplet_loss = nn.TripletMarginLoss(margin=cfg.MARGIN, p=2)
        loss = triplet_loss(query, pos, neg)

        opt.zero_grad()
        loss.backward()
        opt.step()
        # scheduler.step(loss.item())
        if (i+1) % 1 == 0:
            print("Epoch [{}/{}], Step [{}/{}] Loss: {:.8e}".format(epoch + 1,
                                                                    cfg.NUM_EPOCHS,
                                                                    i + 1,
                                                                    TOTAL_STEP,
                                                                    loss.item()))

        # probably unessisary but I was having memory overflow during training
        # I fixed the issue by not augmenting the images, and i'm just keeping
        # these deletes just incase
        del triplet_loss
        del loss
        del query
        del pos
        del neg
        gc.collect()

    # Decay learning rate
    if (epoch+1) % 3 == 0:
        CURR_LR /= 1.5
        for param_group in opt.param_groups:
            param_group['lr'] = CURR_LR
        print('UPDATED LR TO: ', CURR_LR)
    torch.save(res_model.state_dict(), '%s/MRS%s.ckpt' % (cfg.out_model_dir,
                                                          epoch + cfg.START_EPOCH))
