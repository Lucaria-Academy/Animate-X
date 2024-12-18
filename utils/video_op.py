import os
import os.path as osp
import sys
import cv2
import glob
import math
import torch
import gzip
import copy
import time
import json
import pickle
import base64
import imageio
import hashlib
import requests
import binascii
import zipfile
# import skvideo.io
import numpy as np
from io import BytesIO
import urllib.request
import torch.nn.functional as F
import torchvision.utils as tvutils
from multiprocessing.pool import ThreadPool as Pool
from einops import rearrange
from PIL import Image, ImageDraw, ImageFont

@torch.no_grad()
def save_video_multiple_conditions_not_gif_horizontal_1col(local_path, video_tensor, model_kwargs, source_imgs, 
                                   mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5], nrow=8, retry=5, save_fps=8):
    mean=torch.tensor(mean,device=video_tensor.device).view(1,-1,1,1,1)#ncfhw
    std=torch.tensor(std,device=video_tensor.device).view(1,-1,1,1,1)#ncfhw
    video_tensor = video_tensor.mul_(std).add_(mean)  #### unnormalize back to [0,1]
    video_tensor.clamp_(0, 1)

    b, c, n, h, w = video_tensor.shape
    source_imgs = F.adaptive_avg_pool3d(source_imgs, (n, h, w))
    source_imgs = source_imgs.cpu()

    model_kwargs_channel3 = {}
    for key, conditions in model_kwargs[0].items():

        
        if conditions.size(1) == 1:
            conditions = torch.cat([conditions, conditions, conditions], dim=1)
            conditions = F.adaptive_avg_pool3d(conditions, (n, h, w))
        if conditions.size(1) == 2:
            conditions = torch.cat([conditions, conditions[:,:1,]], dim=1)
            conditions = F.adaptive_avg_pool3d(conditions, (n, h, w))
        elif conditions.size(1) == 3:
            conditions = F.adaptive_avg_pool3d(conditions, (n, h, w))
        elif conditions.size(1) == 4: # means it is a mask.
            color = ((conditions[:, 0:3] + 1.)/2.) # .astype(np.float32)
            alpha = conditions[:, 3:4] # .astype(np.float32)
            conditions = color * alpha + 1.0 * (1.0 - alpha)
            conditions = F.adaptive_avg_pool3d(conditions, (n, h, w))
        model_kwargs_channel3[key] = conditions.cpu() if conditions.is_cuda else conditions
    
    # filename = rand_name(suffix='.gif')
    for _ in [None] * retry:
        try:
            vid_gif = rearrange(video_tensor, '(i j) c f h w -> c f (i h) (j w)', i = nrow)
            
            # cons_list = [rearrange(con, '(i j) c f h w -> c f (i h) (j w)', i = nrow) for _, con in model_kwargs_channel3.items()]
            # vid_gif = torch.cat(cons_list + [vid_gif,], dim=3)
            
            vid_gif = vid_gif.permute(1,2,3,0)
            
            images = vid_gif * 255.0
            images = [(img.numpy()).astype('uint8') for img in images]
            if len(images) == 1:
                
                local_path = local_path.replace('.mp4', '.png')
                cv2.imwrite(local_path, images[0][:,:,::-1], [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                # local_path
                # bucket.put_object_from_file(oss_key, local_path)
            else:

                outputs = []
                for image_name in images:
                    x = Image.fromarray(image_name)
                    outputs.append(x)
                from pathlib import Path
                save_fmt = Path(local_path).suffix

                if save_fmt == ".mp4":
                    with imageio.get_writer(local_path, fps=save_fps) as writer:
                        for img in outputs:
                            img_array = np.array(img)  # Convert PIL Image to numpy array
                            writer.append_data(img_array)

                elif save_fmt == ".gif":
                    outputs[0].save(
                        fp=local_path,
                        format="GIF",
                        append_images=outputs[1:],
                        save_all=True,
                        duration=(1 / save_fps * 1000),
                        loop=0,
                    )
                else:
                    raise ValueError("Unsupported file type. Use .mp4 or .gif.")

                # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                # fps = save_fps
                # image = images[0] 
                # media_writer = cv2.VideoWriter(local_path, fourcc, fps, (image.shape[1],image.shape[0]))
                # for image_name in images:
                #     im = image_name[:,:,::-1] 
                #     media_writer.write(im)
                # media_writer.release()
                
            
            exception = None
            break
        except Exception as e:
            exception = e
            continue
    if exception is not None:
        print('save video to {} failed, error: {}'.format(local_path, exception), flush=True)


@torch.no_grad()
def save_video_multiple_conditions_not_gif_horizontal_3col(local_path, video_tensor, model_kwargs, source_imgs, 
                                   mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5], nrow=8, retry=5, save_fps=8):
    mean=torch.tensor(mean,device=video_tensor.device).view(1,-1,1,1,1)#ncfhw
    std=torch.tensor(std,device=video_tensor.device).view(1,-1,1,1,1)#ncfhw
    video_tensor = video_tensor.mul_(std).add_(mean)  #### unnormalize back to [0,1]
    video_tensor.clamp_(0, 1)

    b, c, n, h, w = video_tensor.shape
    source_imgs = F.adaptive_avg_pool3d(source_imgs, (n, h, w))
    source_imgs = source_imgs.cpu()

    model_kwargs_channel3 = {}
    for key, conditions in model_kwargs[0].items():

        
        if conditions.size(1) == 1:
            conditions = torch.cat([conditions, conditions, conditions], dim=1)
            conditions = F.adaptive_avg_pool3d(conditions, (n, h, w))
        if conditions.size(1) == 2:
            conditions = torch.cat([conditions, conditions[:,:1,]], dim=1)
            conditions = F.adaptive_avg_pool3d(conditions, (n, h, w))
        elif conditions.size(1) == 3:
            conditions = F.adaptive_avg_pool3d(conditions, (n, h, w))
        elif conditions.size(1) == 4: # means it is a mask.
            color = ((conditions[:, 0:3] + 1.)/2.) # .astype(np.float32)
            alpha = conditions[:, 3:4] # .astype(np.float32)
            conditions = color * alpha + 1.0 * (1.0 - alpha)
            conditions = F.adaptive_avg_pool3d(conditions, (n, h, w))
        model_kwargs_channel3[key] = conditions.cpu() if conditions.is_cuda else conditions
    
    # filename = rand_name(suffix='.gif')
    for _ in [None] * retry:
        try:
            vid_gif = rearrange(video_tensor, '(i j) c f h w -> c f (i h) (j w)', i = nrow)
            
            cons_list = [rearrange(con, '(i j) c f h w -> c f (i h) (j w)', i = nrow) for _, con in model_kwargs_channel3.items()]
            vid_gif = torch.cat(cons_list + [vid_gif,], dim=3)
            
            vid_gif = vid_gif.permute(1,2,3,0)
            
            images = vid_gif * 255.0
            images = [(img.numpy()).astype('uint8') for img in images]
            if len(images) == 1:
                
                local_path = local_path.replace('.mp4', '.png')
                cv2.imwrite(local_path, images[0][:,:,::-1], [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                # local_path
                # bucket.put_object_from_file(oss_key, local_path)
            else:

                outputs = []
                for image_name in images:
                    x = Image.fromarray(image_name)
                    outputs.append(x)
                from pathlib import Path
                save_fmt = Path(local_path).suffix

                if save_fmt == ".mp4":
                    with imageio.get_writer(local_path, fps=save_fps) as writer:
                        for img in outputs:
                            img_array = np.array(img)  # Convert PIL Image to numpy array
                            writer.append_data(img_array)

                elif save_fmt == ".gif":
                    outputs[0].save(
                        fp=local_path,
                        format="GIF",
                        append_images=outputs[1:],
                        save_all=True,
                        duration=(1 / save_fps * 1000),
                        loop=0,
                    )
                else:
                    raise ValueError("Unsupported file type. Use .mp4 or .gif.")

                # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                # fps = save_fps
                # image = images[0] 
                # media_writer = cv2.VideoWriter(local_path, fourcc, fps, (image.shape[1],image.shape[0]))
                # for image_name in images:
                #     im = image_name[:,:,::-1] 
                #     media_writer.write(im)
                # media_writer.release()
                
            
            exception = None
            break
        except Exception as e:
            exception = e
            continue
    if exception is not None:
        print('save video to {} failed, error: {}'.format(local_path, exception), flush=True)
