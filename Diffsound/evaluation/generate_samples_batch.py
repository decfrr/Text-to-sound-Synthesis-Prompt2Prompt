# ------------------------------------------
# Diffsound
# written by Dongchao Yang
# code based https://github.com/cientgu/VQ-Diffusion
# ------------------------------------------

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import torch
import cv2
import argparse
import numpy as np
import torchvision
from PIL import Image
import soundfile
#sys.path.insert(0,'/apdcephfs/share_1316500/donchaoyang/code3/DiffusionFast')
from sound_synthesis.utils.io import load_yaml_config
from sound_synthesis.modeling.build import build_model
from sound_synthesis.data.build import build_dataloader
from sound_synthesis.utils.misc import get_model_parameters_info
import datetime
from pathlib import Path
from vocoder.modules import Generator
import yaml
import pandas as pd

from prompt2prompt import Controllers

def load_vocoder(ckpt_vocoder: str, eval_mode: bool):
    ckpt_vocoder = Path(ckpt_vocoder)
    # print('ckpt_vocoder ',ckpt_vocoder)
    vocoder_sd = torch.load(ckpt_vocoder / 'best_netG.pt', map_location='cpu')
    # print('vocoder_sd ',vocoder_sd)
    with open(ckpt_vocoder / 'args.yml', 'r') as f:
        args = yaml.load(f, Loader=yaml.UnsafeLoader)
    vocoder = Generator(args.n_mel_channels, args.ngf, args.n_residual_layers)
    vocoder.load_state_dict(vocoder_sd)
    if eval_mode:
        vocoder.eval()
    return {'model': vocoder}

class Diffsound():
    def __init__(self, config, path, ckpt_vocoder):
        self.info = self.get_model(ema=True, model_path=path, config_path=config)
        self.model = self.info['model']
        self.epoch = self.info['epoch']
        self.model_name = self.info['model_name']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad=False
        if ckpt_vocoder:
            self.vocoder = load_vocoder(ckpt_vocoder, eval_mode=True)['model'].to(self.device)
        else:
            self.vocoder = None

    def get_model(self, ema, model_path, config_path):
        if 'OUTPUT' in model_path: # pretrained model
            model_name = model_path.split(os.path.sep)[-3]
        else:
            model_name = os.path.basename(config_path).replace('.yaml', '')

        config = load_yaml_config(config_path)
        model = build_model(config) #加载 dalle model
        model_parameters = get_model_parameters_info(model) #参数详情

        # print(model_parameters)
        print(f'loaded model from {model_path}')
        if os.path.exists(model_path):
            ckpt = torch.load(model_path, map_location="cpu")

        if 'last_epoch' in ckpt:
            epoch = ckpt['last_epoch']
        elif 'epoch' in ckpt:
            epoch = ckpt['epoch']
        else:
            epoch = 0

        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        if len(missing) > 0:
            print('Model missing keys:\n', missing)
        if len(unexpected) > 0:
            print('Model unexpected keys:\n', unexpected)

        if ema==True and 'ema' in ckpt:
            ema_model = model.get_ema_model()
            missing, unexpected = ema_model.load_state_dict(ckpt['ema'], strict=False)

        return {'model': model, 'epoch': epoch, 'model_name': model_name, 'parameter': model_parameters}

    def inference_generate_sample_with_condition(self, text, truncation_rate, save_root, batch_size, fast=False):
        os.makedirs(save_root, exist_ok=True)

        data_i = {}
        data_i['text'] = [text]
        data_i['image'] = None
        condition = text

        str_cond = str(condition)
        save_root_ = os.path.join(save_root, str_cond)
        os.makedirs(save_root_, exist_ok=True)

        if fast != False:
            add_string = 'r,fast'+str(fast-1)
        else:
            add_string = 'r'
        with torch.no_grad():
            model_out = self.model.generate_content(
                batch=data_i,
                filter_ratio=0,
                replicate=10, # 每个样本重复多少次?
                content_ratio=1,
                return_att_weight=False,
                sample_type="top"+str(truncation_rate)+add_string,
            ) # B x C x H x W

        # save results
        content = model_out['content']
        content = content.permute(0, 2, 3, 1).to('cpu').numpy().astype(np.uint8)
        for b in range(content.shape[0]):
            cnt = b
            save_base_name = '{}'.format(str(cnt).zfill(6))
            save_path = os.path.join(save_root_, save_base_name+'.png')
            im = Image.fromarray(content[b])
            im.save(save_path)

    def read_tsv(self, val_path):
        train_tsv = pd.read_csv(val_path, sep=',', usecols=[0,1])
        filenames = train_tsv['file_name']
        captions = train_tsv['caption']
        filenames_ls = []
        captions_ls = []
        for name in filenames:
            filenames_ls.append(name)
        for cap in captions:
            captions_ls.append(cap)
        caps_dict = {}
        for i in range(len(filenames_ls)):
            if filenames_ls[i] not in caps_dict.keys():
                caps_dict[filenames_ls[i]] = [captions_ls[i]]
            else:
                caps_dict[filenames_ls[i]].append(captions_ls[i])
        return caps_dict

    def run_and_generate(self,
                         data_i,
                         controller,
                         content_token,
                         filter_ratio=0,
                         replicate=1,
                         content_ratio=1,
                         return_att_weight=False,
                         sample_type="top0.85r",
                         ):
        with torch.no_grad():
            model_out = self.model.generate_content(
                batch=data_i,
                filter_ratio=filter_ratio,
                controller=controller,
                content_token=content_token,
                replicate=replicate,  # 每个样本重复多少次?
                content_ratio=content_ratio,
                return_att_weight=return_att_weight,
                sample_type=sample_type,
            )

        return model_out

    def generate_sample(self, val_path, truncation_rate, save_root, fast=False):
        # !see text-to-sound.ipynb

        # os.makedirs(save_root, exist_ok=True)
        # print('save_root ',save_root)
        # assert 1==2
        batch_size = 8
        if fast != False:
            add_string = 'r,fast'+str(fast-1)
        else:
            add_string = 'r'
        caps_dict = self.read_tsv(val_path)
        for key in caps_dict.keys():
            generate_num = 0
            base_name = key.split('.')[0]
            print('base_name ',base_name)
            base_name_ = base_name+'_mel_sample_'
            data_i = {}
            data_i['text'] = caps_dict[key]
            # data_i['text'] == prompt
            data_i['image'] = None

            # controller = Controllers.AttentionReplace(
            #     data_i['text'],
            #     100,
            #     cross_replace_steps=.8,
            #     self_replace_steps=.2
            # )
            controller = Controllers.AttentionStore()
            with torch.no_grad():
                model_out = self.run_and_generate(
                    data_i=data_i,
                    filter_ratio=0,
                    controller=controller,
                    replicate=1,
                    content_ratio=1,
                    return_att_weight=False,
                    sample_type="top"+str(truncation_rate)+add_string,
                )
                content = model_out['decode_content']
                # spec to sound
                os.makedirs(save_root, exist_ok=True)
                for b in range(content.shape[0]):
                    save_base_name = base_name_ + str(generate_num)
                    spec = content[b]
                    spec = spec.squeeze(0).cpu().numpy()
                    spec = (spec + 1) / 2

                    # save spec
                    save_spec_root = os.path.join(save_root, 'spec')
                    os.makedirs(save_spec_root, exist_ok=True)
                    save_spec_path = os.path.join(save_spec_root, save_base_name + '.npy')
                    np.save(save_spec_path, spec)

                    # save wav
                    if self.vocoder is not None:
                        wave_from_vocoder = self.vocoder(torch.from_numpy(spec).unsqueeze(0).to(self.device)).cpu().squeeze().detach().numpy()
                        save_sound_root = os.path.join(save_root, 'sound')
                        os.makedirs(save_sound_root, exist_ok=True)
                        save_wav_path = os.path.join(save_sound_root, save_base_name + '.wav')
                        soundfile.write(save_wav_path, wave_from_vocoder, 22050, 'PCM_24')
                    generate_num += 1

if __name__ == '__main__':
    # Note that cap_text.yaml includes the config of vagan, we must choose the right path for it.
    config_path = '/Users/hiro/Lecture/Text-to-sound-Synthesis/Diffsound/evaluation/caps_text.yaml'
    #config_path = '/apdcephfs/share_1316500/donchaoyang/code3/VQ-Diffusion/OUTPUT/caps_train/2022-02-20T21-49-16/caps_text256.yaml'
    pretrained_model_path = '/Users/hiro/Lecture/Text-to-sound-Synthesis/Diffsound/pre_model/diffsound_audiocaps.pth'
    save_root_ = '/Users/hiro/Lecture/Text-to-sound-Synthesis/Diffsound/OUT'
    random_seconds_shift = datetime.timedelta(seconds=np.random.randint(60))
    key_words = 'Real_vgg_pre_399'
    now = (datetime.datetime.now() - random_seconds_shift).strftime('%Y-%m-%dT%H-%M-%S')
    save_root = os.path.join(save_root_, key_words + '_samples_'+now, 'caps_validation')
    val_path = '/Users/hiro/Lecture/Text-to-sound-Synthesis/Diffsound/data_root/audiocaps/new_val.csv'
    ckpt_vocoder = '/Users/hiro/Lecture/Text-to-sound-Synthesis/Diffsound/vocoder/logs/vggsound/'
    Diffsound = Diffsound(config=config_path, path=pretrained_model_path, ckpt_vocoder=ckpt_vocoder)
    Diffsound.generate_sample(val_path=val_path, truncation_rate=0.85, save_root=save_root, fast=False)
