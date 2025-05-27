import torch
import torch.utils.data as data
import os
import random
import numpy as np

import sys
sys.path.append('/home/zx/Desktop/bzq_exp') # add your project path

### rotate and flip
class Augment_RGB_torch:
    def __init__(self):
        pass
    def transform0(self, torch_tensor):
        return torch_tensor   
    def transform1(self, torch_tensor):
        torch_tensor = torch.rot90(torch_tensor, k=1, dims=[-1,-2])
        return torch_tensor
    def transform2(self, torch_tensor):
        torch_tensor = torch.rot90(torch_tensor, k=2, dims=[-1,-2])
        return torch_tensor
    def transform3(self, torch_tensor):
        torch_tensor = torch.rot90(torch_tensor, k=3, dims=[-1,-2])
        return torch_tensor
    def transform4(self, torch_tensor):
        torch_tensor = torch_tensor.flip(-2)
        return torch_tensor
    def transform5(self, torch_tensor):
        torch_tensor = (torch.rot90(torch_tensor, k=1, dims=[-1,-2])).flip(-2)
        return torch_tensor
    def transform6(self, torch_tensor):
        torch_tensor = (torch.rot90(torch_tensor, k=2, dims=[-1,-2])).flip(-2)
        return torch_tensor
    def transform7(self, torch_tensor):
        torch_tensor = (torch.rot90(torch_tensor, k=3, dims=[-1,-2])).flip(-2)
        return torch_tensor

augment   = Augment_RGB_torch()
transforms_aug = [method for method in dir(augment) if callable(getattr(augment, method)) if not method.startswith('_')] 

class Dataset_Holo_256(data.Dataset):
    """256patch"""

    def __init__(self, opt):
        super(Dataset_Holo_256, self).__init__()
        self.data_dir = opt['dir_paths']

        self.mode = opt['mode']

        self.data_all_dir = os.listdir(self.data_dir)
        self.numpy_list = []
        for i in self.data_all_dir:
            if i.endswith('.npy'):
                self.numpy_list.append(i) 
        
        self.ob_real_list = []
        for i in self.numpy_list:
            if 'ob_real' in i:
                self.ob_real_list.append(i) 

        random.seed(2021)
        random.shuffle(self.ob_real_list)
        self.train_ob_real_list = self.ob_real_list[0:int(len(self.ob_real_list)*0.9)]  
        self.val_ob_real_list = self.ob_real_list[int(len(self.ob_real_list)*0.9):]

        self.gt_dic = {}
        for i in self.ob_real_list:
            self.gt_dic[i] = (i.replace('ob_real', 'GT_real'), i.replace('ob_real', 'GT_imag'))

        self.ob_imag_dict = {}
        for i in self.ob_real_list:
            self.ob_imag_dict[i] = i.replace('ob_real', 'ob_imag') 

        self.ft_dic = {}
        for i in self.ob_real_list:
            self.ft_dic[i] = i.replace('ob_real', 'ft')
    
    def __getitem__(self, index):
        if self.mode == 'train':
            ob_real_path = self.train_ob_real_list[index]
            ob_imag_path = self.ob_imag_dict[ob_real_path]
            gt_real_path = self.gt_dic[ob_real_path][0]
            gt_imag_path = self.gt_dic[ob_real_path][1]
            ft_path = self.ft_dic[ob_real_path]

        else:
            ob_real_path = self.val_ob_real_list[index]
            ob_imag_path = self.ob_imag_dict[ob_real_path]
            gt_real_path = self.gt_dic[ob_real_path][0]
            gt_imag_path = self.gt_dic[ob_real_path][1]
            ft_path = self.ft_dic[ob_real_path]

        if ob_real_path.startswith('chart_multi_distance.'):
            self.distance = 3.08
        
        if ob_real_path.startswith('data2.'):
            self.distance = 2.29 

        if ob_real_path.startswith('data3.'):
            self.distance = 3.19 
        
        if ob_real_path.startswith('data5.'):
            self.distance = 3.59 

        if ob_real_path.startswith('data7.'):
            self.distance = 4.18 
        
        if ob_real_path.startswith('data_unlabel_cell.'):
            self.distance = 1.21
        
        if ob_real_path.startswith('newdata1.'):
            self.distance = 3.57
        
        if ob_real_path.startswith('newdata2.'):
            self.distance = 0.825 
        
        if ob_real_path.startswith('newdata3.'):
            self.distance = 0.95  

        if ob_real_path.startswith('newdata4.'):
            self.distance = 0.98 
        
        if ob_real_path.startswith('newdata5.'):
            self.distance = 1.15 
        
        if ob_real_path.startswith('newdata6.'):
            self.distance = 0.93 
        
        if ob_real_path.startswith('newdata7.'):
            self.distance = 0.92 
        
        if ob_real_path.startswith('newdata8.'):
            self.distance = 0.89 
        
        ob_real = torch.from_numpy(np.load(os.path.join(self.data_dir, ob_real_path)))
        ob_imag = torch.from_numpy(np.load(os.path.join(self.data_dir, ob_imag_path)))
        gt_real = torch.from_numpy(np.load(os.path.join(self.data_dir, gt_real_path)))
        gt_imag = torch.from_numpy(np.load(os.path.join(self.data_dir, gt_imag_path)))
        ft = torch.from_numpy(np.load(os.path.join(self.data_dir, ft_path)))

        ob = torch.stack([ob_real, ob_imag], dim=0)
        gt = torch.stack([gt_real, gt_imag], dim=0)

        if self.mode == 'train':
            apply_trans = transforms_aug[random.getrandbits(3)]
            ob = getattr(augment, apply_trans)(ob)
            gt = getattr(augment, apply_trans)(gt)
            ft = getattr(augment, apply_trans)(ft)

        ft = ft.to(torch.float32)
            
        return {'lq': ob, 'gt': gt, 'distance': self.distance, 'ft': ft}
        
    def __len__(self):
        if self.mode == 'train':
            return len(self.train_ob_real_list)
        else:
            return len(self.val_ob_real_list)
    
    def comp_field_norm(self, comp_field):
        comp_field = comp_field[0,...] + 1j*comp_field[1,...]
        comp_field_abs = torch.abs(comp_field)
        comp_field_phase = torch.angle(comp_field)
        comp_field_abs_norm = (comp_field_abs - comp_field_abs.min()) / (comp_field_abs.max() - comp_field_abs.min())
        comp_field_phase_norm = (comp_field_phase - comp_field_phase.min()) / (comp_field_phase.max() - comp_field_phase.min())
        comp_field = comp_field_abs_norm * torch.exp(2j*comp_field_phase_norm*torch.pi) 

        return torch.stack([torch.real(comp_field), torch.imag(comp_field)], dim=0)

    def realnimag_2_ampnphase(self, comp_field):
        comp_field = comp_field[0,...] + 1j*comp_field[1,...]
        comp_field_abs = torch.abs(comp_field)
        comp_field_phase = torch.angle(comp_field)
        return torch.stack([comp_field_abs, comp_field_phase], dim=0)

    def ampnphase_2_realnimag(self, comp_field):
        comp_field = comp_field[0,...] * torch.exp(1j*comp_field[1,...])
        return torch.stack([torch.real(comp_field), torch.imag(comp_field)], dim=0)
        





        

