import sys
sys.path.append("../src/")

from tqdm.notebook import tqdm
import src.config as config
import numpy as np
import torch.nn.functional as F
import torch
import neptune

def train_one_epoch(model,optimizer , data_train, loss_mod1, loss_mod2, epochs):
    model.train()
    mean_loss = []
    mean_diag_train = []
    #for x1, x2 in tqdm(data_train):
    for batch in data_train:
        x1, x2 = batch.values()
        optimizer.zero_grad()
        logits, _, __ = model(x1.to(config.DEVICE), x2.to(config.DEVICE))
        labels = torch.tensor(np.arange(logits.shape[0]))
        
        total_loss = ((loss_mod1(logits, labels.to(config.DEVICE)) + loss_mod2(logits.T, labels.to(config.DEVICE)))/2 
                      #+0.0001*torch.norm(model.encoder_modality1.fc_list[0].weight, p=1)
                      +0.0001*torch.norm(model.encoder_modality2.fc_list[0].weight, p=1)
                     )
                    
        diag = torch.mean(F.softmax(logits).diagonal())
        
        mean_diag_train.append(diag.item())
        mean_loss.append(total_loss.item())
        
        neptune.log_metric('train/loss batch', total_loss.item())
        neptune.log_metric('train/metric batch', diag)
        #wandb.log({"train_batch_loss": total_loss.item(), "train_batch_mean_diag": diag})
        
        
        total_loss.backward()
        optimizer.step()
    neptune.log_metric('train/loss epoch', np.mean(mean_loss))
    neptune.log_metric('train/metric epoch', np.mean(mean_diag_train))
    #wandb.log({'train_epochs_loss': np.mean(mean_loss), 'train_epochs_mean_diag': np.mean(mean_diag_train), 'epochs': epochs})
    

def test_one_epoch(model, data_test, loss_mod1, loss_mod2):
    model.eval()
    mean_diag_test = []
    mean_loss_test = []
    #for x1, x2 in tqdm(data_test):
    with torch.no_grad():
        for batch in data_test:
            x1, x2 = batch.values()
            logits, _, __ = model(x1.to(config.DEVICE), x2.to(config.DEVICE))
            labels = torch.tensor(np.arange(logits.shape[0]))
            
            diag = torch.mean(F.softmax(logits).diagonal()).item()
            mean_diag_test.append(diag)
            
            total_loss = (loss_mod1(logits, labels.to(config.DEVICE)) + loss_mod2(logits.T, labels.to(config.DEVICE)))/2
            mean_loss_test.append(total_loss.item())
            
            neptune.log_metric('test/loss batch', total_loss.item())
            neptune.log_metric('test/metric batch', diag)
        
        #wandb.log({"test_batch_loss": total_loss.item(), "test_batch_mean_diag": diag})
    
    neptune.log_metric('test/loss epoch',  np.mean(mean_loss_test).item())
    neptune.log_metric('test/metric epoch', np.mean(mean_diag_test))

    
def get_prediction_adj(model,data_test_metric, adj, epochs, postprocessing=False):
    model.eval()
    all_emb_mod1 = []
    all_emb_mod2 = []
    indexes = []
    i = 0
    #for x1, x2 in tqdm(data_test_metric):
    with torch.no_grad():
        for batch in data_test_metric:
            x1, x2 = batch.values()
            _,features_mod1, features_mod2 = model(x1.to(config.DEVICE), x2.to(config.DEVICE))
            all_emb_mod1.append(features_mod1.detach().cpu())
            all_emb_mod2.append(features_mod2.detach().cpu())
        
    all_emb_mod1 = torch.cat(all_emb_mod1)
    all_emb_mod2 = torch.cat(all_emb_mod2)
    
    logits=model.logit_scale.exp().cpu()*all_emb_mod1@all_emb_mod2.T
    
    out1_2 = F.softmax(logits)
    out2_1 = F.softmax(logits.T)
    
    mask1_2 = adj == 1
    mask2_1 = adj.transpose() == 1
    
    #wandb.log({"test_metric_mod1_2": out1_2.detach().cpu().numpy()[mask1_2].mean()*1000
    #           , "test_metric_mod2_1": out2_1.detach().cpu().numpy()[mask2_1].mean()*1000, 'epochs': epochs})
    
    #print(out1_2.detach().cpu().numpy()[mask1_2].mean()*1000)
    #print(out2_1.detach().cpu().numpy()[mask2_1].mean()*1000)
    if(postprocessing==False):
        metric1_2 = out1_2.detach().cpu().numpy()[mask1_2].mean()*1000
        metric2_1 = out2_1.detach().cpu().numpy()[mask2_1].mean()*1000
        
        neptune.log_metric('test/adt2gex_lb_metric', metric1_2)
        neptune.log_metric('test/gex2adt_lb_metric', metric2_1)
        neptune.log_metric('test/mean_lb_metric', (metric2_1 + metric1_2)/2)
        return (metric1_2 + metric2_1)/2
    else:
        #TODO postprocessing
        return None
    
    
    