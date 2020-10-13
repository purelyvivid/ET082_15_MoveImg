
import torch
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from vgg import vggface
from torchvision.models import vgg19
from loss import *

vgg19 = vgg19(pretrained=True).cuda()
vggface = vggface('vgg_face_dag.pth').cuda()

criterion_EG = LossEG(vggface , vgg19)
criterion_D = LossD()

def save_n_draw_result(E, G, source_landmark , target_frame , context,
                       save_out_img=True, show_plot=True, 
                       save_out_img_path='./', save_out_img_fname='test.jpg',
                       i_fe = 0, i_fl = 1,
                      ):
    """
    i_fe = 0 # index of face_expression
    i_fl = 3 # index of face_look  # batch_size has to be larger and equal to than 4    
    """
    E.eval()
    G.eval()

    source_landmark , target_frame , context = \
    source_landmark.cuda() , target_frame.cuda() , context.cuda()
    if source_landmark.shape[0]==1:
        print("batch_size is 1 !!")
        i_fl = 0

    with torch.no_grad():
        emb = E(context) # emb: [B, 512]
        x_fe = target_frame[i_fe:i_fe+1].cuda()
        x_fl = target_frame[i_fl:i_fl+1].cuda()
        y_fe = source_landmark[i_fe:i_fe+1].cuda()
        e_fe = emb[i_fe:i_fe+1].cuda()# [1, 512]
        e_fl = emb[i_fl:i_fl+1].cuda()
        x_hat_fe = G(y_fe , e_fe)
        x_hat_fl = G(y_fe , e_fl)
        # save out
        if save_out_img:
            tot = torch.cat([x_fe, y_fe, x_fl, x_hat_fe.cuda(), x_hat_fl.cuda()], 0)
            save_image(tot, save_out_img_path+save_out_img_fname,nrow=5)
        # plot
        if show_plot:
            tot = torch.cat([x_fe, y_fe, x_fl, x_hat_fe.cuda(), x_hat_fl.cuda()], -1)[0]
            img_arr = tot.detach().cpu().numpy().transpose(1,2,0)
            plt.axis('off')
            plt.imshow(img_arr)
            plt.show()
        #return img_arr 
        
def save_n_draw_result_few_shot(E, G, ds, ds_test,
             test_as_face_expression=True, 
             selected_video_idx=9):
    
    i , source_landmark , target_frame , context = ds.__getitem__(selected_video_idx)
    i , source_landmark_test , target_frame_test , context_test = ds_test.__getitem__(0)
    source_landmark = torch.cat( [source_landmark[None,...] , source_landmark_test[None,...] ]  )
    target_frame = torch.cat( [target_frame[None,...] , target_frame_test[None,...] ]  )
    context = torch.cat( [context[None,...] , context[None,...] ]  )
    if test_as_face_expression:
        save_n_draw_result(E, G, source_landmark , target_frame , context,
                           save_out_img_path='./', save_out_img_fname='test.jpg',
                           #save_out_img=False,
                           i_fe = 1, i_fl = 0,
                           )
    else:
        save_n_draw_result(E, G, source_landmark , target_frame , context,
                           save_out_img_path='./', save_out_img_fname='test.jpg',
                           #save_out_img=False,
                           i_fe = 0, i_fl = 1,
                           )  
        
def train_step_few_shot( E, G, D, optim_EG, optim_D, i , source_landmark , target_frame , context, 
               update_D_again=False, print_loss=False):
    E.train()
    G.train()
    D.train()
        
    emb = E(context) # [B, 512]
    target_frame_ = G(source_landmark , emb)
    
    r_x = D( torch.cat([target_frame,source_landmark],1) , i)
    r_x_hat = D.forward_few_shot( torch.cat([target_frame_,source_landmark],1) , emb)
    
    optim_D.zero_grad()
    optim_EG.zero_grad()
    
    loss_EG = criterion_EG(target_frame_ , target_frame , r_x_hat)
    loss_D = criterion_D(r_x , r_x_hat)
    loss = loss_EG + loss_D
    loss.backward()
    
    optim_EG.step()
    optim_D.step()
    
    if print_loss: print("loss:{:.4f}, loss_D:{:.4f}, loss_EG:{:.4f}".format(loss.item(), loss_D.item(), loss_EG.item()))
    
    if update_D_again:
        target_frame_ = target_frame_.detach()
        r_x = D( torch.cat([target_frame,source_landmark],1) , i)
        r_x_hat = D( torch.cat([target_frame_,source_landmark],1) , i)

        optim_D.zero_grad()
        loss_D = criterion_D(r_x, r_x_hat)
        loss_D.backward()
        optim_D.step()
    
        if print_loss: print("loss_D:{:.4f}, ".format(loss_D.item() ))
            
        
    
    return target_frame_,emb,i

def train_step( E, G, D, optim_EG, optim_D, i , source_landmark , target_frame , context, 
               update_D_again=False, set_W=False, print_loss=False):
    E.train()
    G.train()
    D.train()
        
    emb = E(context) #(B, 512)
    target_frame_ = G(source_landmark , emb)
    
    r_x = D( torch.cat([target_frame,source_landmark],1) , i)
    r_x_hat = D( torch.cat([target_frame_,source_landmark],1) , i)
    
    optim_D.zero_grad()
    optim_EG.zero_grad()
    
    loss_EG = criterion_EG(target_frame_ , target_frame , r_x_hat)
    loss_D = criterion_D(r_x , r_x_hat)
    loss = loss_EG + loss_D
    loss.backward()
    
    optim_EG.step()
    optim_D.step()
    
    if print_loss: print("loss:{:.4f}, loss_D:{:.4f}, loss_EG:{:.4f}".format(loss.item(), loss_D.item(), loss_EG.item()))
    
    if update_D_again:
        target_frame_ = target_frame_.detach()
        r_x = D( torch.cat([target_frame,source_landmark],1) , i)
        r_x_hat = D( torch.cat([target_frame_,source_landmark],1) , i)

        optim_D.zero_grad()
        loss_D = criterion_D(r_x, r_x_hat)
        loss_D.backward()
        optim_D.step()
    
        if print_loss: print("loss_D:{:.4f}, ".format(loss_D.item() ))
            
    if set_W:
        D.set_W(i, emb)
        
    
    return target_frame_,emb,i

def train_step_EG( E, G, optim_EG, i , source_landmark , target_frame , context):
    emb = E(context)
    target_frame_ = G(source_landmark , emb)

    optim_EG.zero_grad()
    loss = criterion_EG.loss_cnt_(target_frame_ , target_frame)
    loss.backward()
    optim_EG.step()
    
    print(loss.item())
    
    return target_frame_

    
def load_model(E,G,D,load_model_folder="./", postfix=""):
    E.load_state_dict(torch.load(f"{load_model_folder}E_{postfix}.torchmodel"))
    G.load_state_dict(torch.load(f"{load_model_folder}G_{postfix}.torchmodel"))
    D.load_state_dict(torch.load(f"{load_model_folder}D_{postfix}.torchmodel"))
    return E,G,D


def save_model(E, G, D, save_model_folder="./", postfix=""):
    torch.save(E.state_dict(), f"{save_model_folder}E_{postfix}.torchmodel")
    torch.save(G.state_dict(), f"{save_model_folder}G_{postfix}.torchmodel")
    torch.save(D.state_dict(), f"{save_model_folder}D_{postfix}.torchmodel")

