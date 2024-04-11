import torch
from model import DetModel
# from cam import get_image_from_cam
from torchvision import transforms
import numpy as np
import time


world_target_global = np.array([0.7188, 0.6944])
world_target_local = np.array([0.5625,0.8958])
world_target_local_down = np.array([0.5844,0.4861])

_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 缩放
    transforms.Normalize([0.5]*6, [0.5]*6),  # 标准化均值为0标准差为1
])
class Train(object):
    def __init__(self, model):
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer,
                                                             max_lr=1e-3,
                                                             total_steps=2000,
                                                             div_factor=25,
                                                             final_div_factor=100,
                                                             three_phase=True)
        self.device = torch.device('cpu')
        self.model.to(self.device)

    def sample(self, state):
        state = _transform(torch.unsqueeze(torch.FloatTensor(state), dim=0)).to(self.device)
        self.model.eval()
        obs = self.model(state)
        obs = obs.detach().cpu().numpy()[0]
        self.model.train()
        return obs

    def load_checkpoint(self, path):
        paraters = torch.load(path, map_location=torch.device('cpu'))
        if 'model' in paraters:
            self.model.load_state_dict(paraters['model'])
        else:
            self.model.load_state_dict(paraters)


    def get_action_from_obs(self,obs_target, arm_down):

        if abs(obs_target[6]-obs_target[4])<0.7: # global
            obs_pos = np.array([obs_target[4]/2+obs_target[6]/2,obs_target[5]/2+obs_target[7]/2])
            error_2 = obs_pos - world_target_global 
        else:
            error_2 = np.zeros(2)

        if abs(obs_target[2]-obs_target[0])<0.7: # local: check if bbox not too big
            obs_pos = np.array([obs_target[0]/2+obs_target[2]/2,obs_target[1]/2+obs_target[3]/2])
            if arm_down==0:
                error_1 = obs_pos - world_target_local
            else:
                error_1 = obs_pos - world_target_local_down
        else:
            error_1 = error_2

        if arm_down==1: # switch from global to local
            error = 1.0*error_1 + 0.0*error_2
        else:
            error = [0.8*error_1[0]+0.2*error_2[0], 0.2*error_1[1]+0.8*error_2[1]]

        dist = (error[0] ** 2 + error[1] ** 2) ** 0.5
        # print('\r Dist: %.4f   Error:%.4f  %.4f'%(dist,error[0],error[1]),end='')

        if dist<=0.30:
            arm_down = 1 # put arm down
        
        if dist>0.30 and arm_down==0: # move bigger
            if error[0]>0.07: # d
                # self.step([-10, -10, -10, -10, 0, 0, 0])
                act_id = 0
            elif error[0]<-0.07: # a
                # self.step([10, 10, 10, 10, 0, 0, 0])
                act_id = 1
            else:
                if error[1] < 0: # w
                    # self.step([-20,-20,20,20,0,0,0])
                    act_id = 2
                else: # s
                    # self.step([20,20,-20,-20,0,0,0])
                    act_id = 3

        elif 0.30>=dist>0.075 or (dist>0.30 and arm_down==1) or (dist<=0.075 and (error[0]>0.035 or error[0]<-0.035)): # arm down

            arm_down = 1

            if error[0]>0.035:
                # self.step([-80, -80, -80, -80, 0, 0, 0])
                act_id = 4
            elif error[0]<-0.035:
                # self.step([60, 60, 60, 60, 0, 0, 0])
                act_id = 5
            else:
                if error[1] < 0:
                    # self.step([-100,-100,100,100,0,0,0])
                    act_id = 6
                else:
                    # self.step([100,100,-100,-100,0,0,0])
                    act_id = 7

        else:
            arm_down = 1
            act_id = 8
        
        return act_id, arm_down, dist, error

def load_agent(path):
    agent = Train(model=DetModel(4,'swin_tiny_patch4_window7_224'))
    agent.load_checkpoint(path)
    print('Load model from: {}'.format(path))
    return agent


if __name__=='__main__':
    
    agent = load_agent()
    print('done')