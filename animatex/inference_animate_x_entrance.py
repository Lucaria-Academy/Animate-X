import os
import re
import os.path as osp
import sys
sys.path.insert(0, '/'.join(osp.realpath(__file__).split('/')[:-4]))
import json
import math
import torch
import pynvml
import logging
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.cuda.amp as amp
from importlib import reload
import torch.distributed as dist
import torch.multiprocessing as mp
import random
from einops import rearrange
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.nn.parallel import DistributedDataParallel

from .default_config import cfg
from .model.autoencoder import get_first_stage_encoding

import utils.transforms as data
from utils.seed import setup_seed
from utils.multi_port import find_free_port
from utils.distributed import generalized_all_gather
from utils.video_op import save_video_multiple_conditions_not_gif_horizontal_3col, save_video_multiple_conditions_not_gif_horizontal_1col
from utils.registry_class import INFER_ENGINE, MODEL, EMBEDDER, AUTO_ENCODER, DIFFUSION

from copy import copy
import cv2, pickle


@INFER_ENGINE.register_function()
def inference_animate_x_entrance(cfg_update,  **kwargs):
    for k, v in cfg_update.items():
        if isinstance(v, dict) and k in cfg:
            cfg[k].update(v)
        else:
            cfg[k] = v
    
    if not 'MASTER_ADDR' in os.environ:
        os.environ['MASTER_ADDR']='localhost'
        os.environ['MASTER_PORT']= find_free_port()
    cfg.pmi_rank = int(os.getenv('RANK', 0)) 
    cfg.pmi_world_size = int(os.getenv('WORLD_SIZE', 1))
    
    if cfg.debug:
        cfg.gpus_per_machine = 1
        cfg.world_size = 1
    else:
        cfg.gpus_per_machine = torch.cuda.device_count()
        cfg.world_size = cfg.pmi_world_size * cfg.gpus_per_machine
    
    if cfg.world_size == 1:
        worker(0, cfg, cfg_update)
    else:
        mp.spawn(worker, nprocs=cfg.gpus_per_machine, args=(cfg, cfg_update))
    return cfg

def process_single_pose_embedding(dwpose_source_data):


    bodies = dwpose_source_data['bodies']['candidate'][:18]

    results = np.swapaxes(bodies, 0, 1) # (32, 2, 128)
    return results

def make_masked_images(imgs, masks):
    masked_imgs = []
    for i, mask in enumerate(masks):        
        # concatenation
        masked_imgs.append(torch.cat([imgs[i] * (1 - mask), (1 - mask)], dim=1))
    return torch.stack(masked_imgs, dim=0)

def process_single_pose_embedding_katong(dwpose_source_data, index):


    bodies = dwpose_source_data['bodies'][index][:18]



    results = bodies
    results = np.swapaxes(results, 0, 1) # (32, 2, 128)
    return results

def load_video_frames(ref_image_path, pose_file_path, original_driven_video_path, pose_embedding_key, train_trans, vit_transforms, train_trans_pose, max_frames=32, frame_interval = 1, resolution=[512, 768], get_first_frame=True, vit_resolution=[224, 224]):

    pose_embedding_dim = 18

    
    for _ in range(5):
        # try:
            dwpose_all = {}
            frames_all = {}

            original_driven_video_all = {}
            original_driven_video_frame_all = {}
            pose_embedding_all = {}
            

            

            # 打开文件（以二进制读取模式）
            with open(pose_embedding_key, 'rb') as file:
                # 使用 pickle.load() 方法读取字典
                loaded_data = pickle.load(file)

            try:
                ref_pose_embedding_key = pose_embedding_key.replace(".pkl", "_ref_pose.pkl")
                with open(ref_pose_embedding_key, 'rb') as file:
                    # 使用 pickle.load() 方法读取字典
                    ref_loaded_data = pickle.load(file)
                    ref_pose_embedding = process_single_pose_embedding(ref_loaded_data)

            except:
                ref_pose_embedding = process_single_pose_embedding_katong(loaded_data, 0)


            first_image = True
            for ii_index in sorted(os.listdir(pose_file_path)):
                # ii_index = ii_index.strip()

                if ii_index != "ref_pose.jpg":

                    dwpose_all[ii_index] = Image.open(pose_file_path+"/"+ii_index)
                    frames_all[ii_index] = Image.fromarray(cv2.cvtColor(cv2.imread(ref_image_path),cv2.COLOR_BGR2RGB)) 

                    try:
                        i_index = int(ii_index.split('.')[0])
                    except:
                        i_index = int(ii_index.split('.')[0].split('_')[1])
                    try:
                        pose_embedding_all[ii_index] = process_single_pose_embedding( loaded_data[i_index]) # (2, 128)
                    except:
                        pose_embedding_all[ii_index] = process_single_pose_embedding_katong( loaded_data, i_index) # (2, 128)

            for ii_index in sorted(os.listdir(original_driven_video_path)):

                original_driven_video_all[ii_index] = Image.open(original_driven_video_path+"/"+ii_index)


                # frames_all[ii_index] = Image.open(ref_image_path)
            pose_ref_path = os.path.join(pose_file_path, "ref_pose.jpg")
            if os.path.exists(pose_ref_path) == False:
                pose_ref_path = os.path.join(pose_file_path, os.listdir(pose_file_path)[0])
            pose_ref = Image.open(pose_ref_path)
            first_eq_ref = False

            # sample max_frames poses for video generation
            stride = frame_interval
            _total_frame_num = len(frames_all)
            cover_frame_num = (stride * (max_frames-1)+1)
            if _total_frame_num < cover_frame_num:
                print('_total_frame_num is smaller than cover_frame_num, the sampled frame interval is changed')
                start_frame = 0   # we set start_frame = 0 because the pose alignment is performed on the first frame
                end_frame = _total_frame_num
                stride = max((_total_frame_num-1//(max_frames-1)),1)
                end_frame = stride*max_frames
            else:
                start_frame = 0  # we set start_frame = 0 because the pose alignment is performed on the first frame
                end_frame = start_frame + cover_frame_num
            
            frame_list = []
            dwpose_list = []
            original_driven_video_list = []
            pose_embedding_list = []
            random_ref_frame = frames_all[list(frames_all.keys())[0]]
            if random_ref_frame.mode != 'RGB':
                random_ref_frame = random_ref_frame.convert('RGB')
            random_ref_dwpose = pose_ref 
            if random_ref_dwpose.mode != 'RGB':
                random_ref_dwpose = random_ref_dwpose.convert('RGB')
            for i_index in range(start_frame, end_frame, stride):
                # import pdb; pdb.set_trace()
                if i_index == start_frame and first_eq_ref:
                    # print("i_index == start_frame and first_eq_ref:")
                    i_key = list(frames_all.keys())[i_index]
                    i_frame = frames_all[i_key]

                    if i_frame.mode != 'RGB':
                        i_frame = i_frame.convert('RGB')
                    i_dwpose = frames_pose_ref
                    if i_dwpose.mode != 'RGB':
                        i_dwpose = i_dwpose.convert('RGB')
                    frame_list.append(i_frame)
                    dwpose_list.append(i_dwpose)
                else:
                    # added 
                    
                    if first_eq_ref:
                        i_index = i_index - stride
                    # print("key = list(frames_all.keys())[i_index]")
                    i_key = list(frames_all.keys())[i_index]
                    i_frame = frames_all[i_key]
                    if i_frame.mode != 'RGB':
                        i_frame = i_frame.convert('RGB')
                    i_dwpose = dwpose_all[i_key]
                    # ii_index = ii_index.strip()
                    # print(original_driven_video_all.keys())
                    i_original_driven_video = original_driven_video_all[i_key.strip()]
                    i_pose_embedding = pose_embedding_all[i_key]
                    if i_dwpose.mode != 'RGB':
                        i_dwpose = i_dwpose.convert('RGB')

                    if i_original_driven_video.mode != 'RGB':
                        i_original_driven_video = i_original_driven_video.convert('RGB')
                    frame_list.append(i_frame)
                    dwpose_list.append(i_dwpose)
                    original_driven_video_list.append(i_original_driven_video)


                    pose_embedding_list.append(i_pose_embedding)

            have_frames = len(frame_list)>0
            middle_indix = 0
            if have_frames:
                ref_frame = frame_list[middle_indix]

                vit_frame = vit_transforms(ref_frame)
                random_ref_frame_tmp = train_trans_pose(random_ref_frame)
                random_ref_dwpose_tmp = train_trans_pose(random_ref_dwpose) 
                
                original_driven_video_data_tmp = torch.stack([vit_transforms(ss) for ss in original_driven_video_list], dim=0)

                
                ref_pose_embedding_tmp = torch.from_numpy(ref_pose_embedding)
                
                misc_data_tmp = torch.stack([train_trans_pose(ss) for ss in frame_list], dim=0)
                video_data_tmp = torch.stack([train_trans(ss) for ss in frame_list], dim=0) 
                dwpose_data_tmp = torch.stack([train_trans_pose(ss) for ss in dwpose_list], dim=0)
                
                pose_embedding_tmp = torch.stack([torch.from_numpy(ss) for ss in pose_embedding_list], dim=0)
            

            video_data = torch.zeros(max_frames, 3, resolution[1], resolution[0])
            dwpose_data = torch.zeros(max_frames, 3, resolution[1], resolution[0])

            original_driven_video_data = torch.zeros(max_frames, 3, 224, 224)

            pose_embedding = torch.zeros(max_frames, 2, pose_embedding_dim)
            ref_pose_embedding = torch.zeros(max_frames, 2, pose_embedding_dim)

            misc_data = torch.zeros(max_frames, 3, resolution[1], resolution[0])
            random_ref_frame_data = torch.zeros(max_frames, 3, resolution[1], resolution[0]) # [32, 3, 512, 768]
            random_ref_dwpose_data = torch.zeros(max_frames, 3, resolution[1], resolution[0])
            if have_frames:
                video_data[:len(frame_list), ...] = video_data_tmp      
                misc_data[:len(frame_list), ...] = misc_data_tmp
                dwpose_data[:len(frame_list), ...] = dwpose_data_tmp

                original_driven_video_data[:len(frame_list), ...] = original_driven_video_data_tmp

                pose_embedding[:len(frame_list), ...] = pose_embedding_tmp
                # print("random_ref_frame_tmp.shape", random_ref_frame_tmp.shape)
                random_ref_frame_data[:,...] = random_ref_frame_tmp
                # print("random_ref_frame_data.shape", random_ref_frame_data.shape)
                random_ref_dwpose_data[:,...] = random_ref_dwpose_tmp


                ref_pose_embedding[:,...] = ref_pose_embedding_tmp
            
            break
            
        # except Exception as e:
        #     logging.info('{} read video frame failed with error: {}'.format(pose_file_path, e))
        #     continue
    
    return vit_frame, video_data, misc_data, dwpose_data, random_ref_frame_data, random_ref_dwpose_data, pose_embedding, ref_pose_embedding, original_driven_video_data



def worker(gpu, cfg, cfg_update):
    '''
    Inference worker for each gpu
    '''
    for k, v in cfg_update.items():
        if isinstance(v, dict) and k in cfg:
            cfg[k].update(v)
        else:
            cfg[k] = v

    cfg.gpu = gpu
    cfg.seed = int(cfg.seed)
    cfg.rank = cfg.pmi_rank * cfg.gpus_per_machine + gpu
    setup_seed(cfg.seed + cfg.rank)

    if not cfg.debug:
        torch.cuda.set_device(gpu)
        torch.backends.cudnn.benchmark = True
        if hasattr(cfg, "CPU_CLIP_VAE") and cfg.CPU_CLIP_VAE:
            torch.backends.cudnn.benchmark = False
        dist.init_process_group(backend='nccl', world_size=cfg.world_size, rank=cfg.rank)

    # [Log] Save logging and make log dir
    log_dir = generalized_all_gather(cfg.log_dir)[0]
    inf_name = osp.basename(cfg.cfg_file).split('.')[0]
    test_model = osp.basename(cfg.test_model).split('.')[0].split('_')[-1]
    
    cfg.log_dir = osp.join(cfg.log_dir, '%s' % (inf_name))
    os.makedirs(cfg.log_dir, exist_ok=True)
    log_file = osp.join(cfg.log_dir, 'log_%02d.txt' % (cfg.rank))
    cfg.log_file = log_file
    reload(logging)
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler(filename=log_file),
            logging.StreamHandler(stream=sys.stdout)])
    logging.info(cfg)
    logging.info(f"Running Animate-X inference on gpu {gpu}")
    
    # [Diffusion]
    diffusion = DIFFUSION.build(cfg.Diffusion)

    # [Data] Data Transform    
    train_trans = data.Compose([
        data.Resize(cfg.resolution),
        data.ToTensor(),
        data.Normalize(mean=cfg.mean, std=cfg.std)
        ])

    train_trans_pose = data.Compose([
        data.Resize(cfg.resolution),
        data.ToTensor(),
        ]
        )

    vit_transforms = T.Compose([
                data.Resize(cfg.vit_resolution),
                T.ToTensor(),
                T.Normalize(mean=cfg.vit_mean, std=cfg.vit_std)])

    # [Model] embedder
    clip_encoder = EMBEDDER.build(cfg.embedder)
    clip_encoder.model.to(gpu)
    with torch.no_grad():
        _, _, zero_y = clip_encoder(text="")
    

    # [Model] auotoencoder 
    autoencoder = AUTO_ENCODER.build(cfg.auto_encoder)
    autoencoder.eval() # freeze
    for param in autoencoder.parameters():
        param.requires_grad = False
    autoencoder.cuda()
    
    # [Model] UNet 
    if "config" in cfg.UNet:
        cfg.UNet["config"] = cfg
    cfg.UNet["zero_y"] = zero_y
    model = MODEL.build(cfg.UNet)
    state_dict = torch.load(cfg.test_model, map_location='cpu')
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    if 'step' in state_dict:
        resume_step = state_dict['step']
    else:
        resume_step = 0
    try:
        status = model.load_state_dict(state_dict, strict=False)
    except:

        for key in list(state_dict.keys()):
            if 'pose_embedding_before.pos_embed.pos_table' in key:  
                del state_dict[key]
        status = model.load_state_dict(state_dict, strict=False)
    logging.info('Load model from {} with status {}'.format(cfg.test_model, status))
    model = model.to(gpu)
    model.eval()
    if hasattr(cfg, "CPU_CLIP_VAE") and cfg.CPU_CLIP_VAE:
        model.to(torch.float16) 
    else:
        model = DistributedDataParallel(model, device_ids=[gpu]) if not cfg.debug else model
    torch.cuda.empty_cache()


    
    test_list = cfg.test_list_path

    # test_list.reverse()

    num_videos = len(test_list)



    logging.info(f'There are {num_videos} videos. with {cfg.round} times')
    # test_list = [item for item in test_list for _ in range(cfg.round)]
    test_list = [item for _ in range(cfg.round) for item in test_list]
    
    for idx, file_path in enumerate(test_list):
        cfg.frame_interval, ref_image_key, pose_seq_key, original_driven_video_seq_key, pose_embedding_key = file_path[0], file_path[1], file_path[2], file_path[3], file_path[4]
        
        try:
            current_seed = file_path[5]
        except:
            current_seed = int(cfg.seed)
        manual_seed = int(current_seed + cfg.rank + idx//num_videos)
        setup_seed(manual_seed)

        logging.info(f"[{idx}]/[{len(test_list)}] Begin to sample {ref_image_key}, pose sequence from {pose_seq_key} init seed {manual_seed} ...")
        

        vit_frame, video_data, misc_data, dwpose_data, random_ref_frame_data, random_ref_dwpose_data, pose_embedding, ref_pose_embedding, original_driven_video_data = load_video_frames(ref_image_key, pose_seq_key, original_driven_video_seq_key, pose_embedding_key, train_trans, vit_transforms, train_trans_pose, max_frames=cfg.max_frames, frame_interval =cfg.frame_interval, resolution=cfg.resolution)
        
        

        original_driven_video_data = torch.cat([vit_frame.unsqueeze(0), original_driven_video_data], 0)

        misc_data = misc_data.unsqueeze(0).to(gpu)
        vit_frame = vit_frame.unsqueeze(0).to(gpu)
        dwpose_data = dwpose_data.unsqueeze(0).to(gpu)
        original_driven_video_data = original_driven_video_data.unsqueeze(0).to(gpu)
        random_ref_frame_data = random_ref_frame_data.unsqueeze(0).to(gpu)
        random_ref_dwpose_data = random_ref_dwpose_data.unsqueeze(0).to(gpu)

        pose_embedding = pose_embedding.unsqueeze(0).to(gpu)
        ref_pose_embedding = ref_pose_embedding[0:1].unsqueeze(0).to(gpu)

        pose_embedding = torch.cat([ref_pose_embedding, pose_embedding], dim = 1)

        # print("pose_embedding.shape: ", pose_embedding.shape)

        ### save for visualization
        misc_backups = copy(misc_data)
        frames_num = misc_data.shape[1]
        misc_backups = rearrange(misc_backups, 'b f c h w -> b c f h w')
        mv_data_video = []
        

        ### local image (first frame)
        image_local = []
        if 'local_image' in cfg.video_compositions:
            frames_num = misc_data.shape[1]
            bs_vd_local = misc_data.shape[0]
            image_local = misc_data[:,:1].clone().repeat(1,frames_num,1,1,1)
            image_local_clone = rearrange(image_local, 'b f c h w -> b c f h w', b = bs_vd_local)
            image_local = rearrange(image_local, 'b f c h w -> b c f h w', b = bs_vd_local)

            # no
            if hasattr(cfg, "latent_local_image") and cfg.latent_local_image:
                with torch.no_grad():
                    temporal_length = frames_num
                    # print("video_data[:,0].shape", video_data[:,0].shape) #
                    encoder_posterior = autoencoder.encode(video_data[:,0])
                    local_image_data = get_first_stage_encoding(encoder_posterior).detach()
                    image_local = local_image_data.unsqueeze(1).repeat(1,temporal_length,1,1,1) # [10, 16, 4, 64, 40]
                    # print("image_local.shape", image_local.shape) #

        
        ### encode the video_data
        bs_vd = misc_data.shape[0]
        misc_data = rearrange(misc_data, 'b f c h w -> (b f) c h w')
        misc_data_list = torch.chunk(misc_data, misc_data.shape[0]//cfg.chunk_size,dim=0)
        

        with torch.no_grad():
            
            random_ref_frame = []
            if 'randomref' in cfg.video_compositions:
                random_ref_frame_clone = rearrange(random_ref_frame_data, 'b f c h w -> b c f h w')
                if hasattr(cfg, "latent_random_ref") and cfg.latent_random_ref:
                    
                    temporal_length = random_ref_frame_data.shape[1]
                    encoder_posterior = autoencoder.encode(random_ref_frame_data[:,0].sub(0.5).div_(0.5))
                    random_ref_frame_data = get_first_stage_encoding(encoder_posterior).detach()
                    random_ref_frame_data = random_ref_frame_data.unsqueeze(1).repeat(1,temporal_length,1,1,1) # [10, 16, 4, 64, 40]
                    # print("random_ref_frame_data.shape", random_ref_frame_data.shape) #
                random_ref_frame = rearrange(random_ref_frame_data, 'b f c h w -> b c f h w')


            if 'dwpose' in cfg.video_compositions:
                bs_vd_local = dwpose_data.shape[0]
                dwpose_data_clone = rearrange(dwpose_data.clone(), 'b f c h w -> b c f h w', b = bs_vd_local)
                if 'randomref_pose' in cfg.video_compositions:
                    dwpose_data = torch.cat([random_ref_dwpose_data[:,:1], dwpose_data], dim=1)
                dwpose_data = rearrange(dwpose_data, 'b f c h w -> b c f h w', b = bs_vd_local)
                # print("dwpose_data = rearrange(dwpose_dat.shape", dwpose_data.shape) #
            
            y_visual = []
            if 'image' in cfg.video_compositions:
                with torch.no_grad():
                    vit_frame = vit_frame.squeeze(1)
                    y_visual = clip_encoder.encode_image(vit_frame).unsqueeze(1) # [60, 1024]
                    y_visual0 = y_visual.clone()

            batch_size, seq_len = original_driven_video_data.shape[0], original_driven_video_data.shape[1]

            original_driven_video_data = original_driven_video_data.reshape(batch_size*seq_len,3,224,224)
            original_driven_video_data_embedding = clip_encoder.encode_image(original_driven_video_data).unsqueeze(1) # [60, 1024]
            
            # print("original_driven_video_data_embedding.shape: ", original_driven_video_data_embedding.shape)
            original_driven_video_data_embedding = original_driven_video_data_embedding.clone()

        with amp.autocast(enabled=True):
            pynvml.nvmlInit()
            handle=pynvml.nvmlDeviceGetHandleByIndex(0)
            meminfo=pynvml.nvmlDeviceGetMemoryInfo(handle)
            cur_seed = torch.initial_seed()
            logging.info(f"Current seed {cur_seed} ...")

            noise = torch.randn([1, 4, cfg.max_frames, int(cfg.resolution[1]/cfg.scale), int(cfg.resolution[0]/cfg.scale)])
            noise = noise.to(gpu)

            if hasattr(cfg.Diffusion, "noise_strength"):
                b, c, f, _, _= noise.shape
                offset_noise = torch.randn(b, c, f, 1, 1, device=noise.device)
                noise = noise + cfg.Diffusion.noise_strength * offset_noise


            full_model_kwargs=[{
                                        'y': None,
                                        'pose_embeddings': [pose_embedding, original_driven_video_data_embedding],
                                        "local_image": None if len(image_local) == 0 else image_local[:],
                                        'image': None if len(y_visual) == 0 else y_visual0[:],
                                        'dwpose': None if len(dwpose_data) == 0 else dwpose_data[:],
                                        'randomref': None if len(random_ref_frame) == 0 else random_ref_frame[:],
                                       }, 
                                       {
                                        'y': None,
                                        "local_image": None, 
                                        'image': None,
                                        'randomref': None,
                                        'dwpose': None, 
                                        "pose_embeddings": None, 
                                       }]

            # for visualization
            full_model_kwargs_vis =[{
                                        'y': None,
                                        "local_image": None if len(image_local) == 0 else image_local_clone[:],
                                        'image': None,
                                        'pose_embeddings': [pose_embedding, original_driven_video_data_embedding],
                                        'dwpose': None if len(dwpose_data_clone) == 0 else dwpose_data_clone[:],
                                        'randomref': None if len(random_ref_frame) == 0 else random_ref_frame_clone[:, :3],
                                       }, 
                                       {
                                        'y': None,
                                        "local_image": None, 
                                        'image': None,
                                        'randomref': None,
                                        'dwpose': None, 
                                        "pose_embeddings": None, 
                                       }]

            
            partial_keys = [
                    ['image', 'randomref', "dwpose","pose_embeddings"],
                ]
            if hasattr(cfg, "partial_keys") and cfg.partial_keys:
                partial_keys = cfg.partial_keys

            for partial_keys_one in partial_keys:
                model_kwargs_one = prepare_model_kwargs(partial_keys = partial_keys_one,
                                    full_model_kwargs = full_model_kwargs,
                                    use_fps_condition = cfg.use_fps_condition)




                model_kwargs_one_vis = prepare_model_kwargs(partial_keys = partial_keys_one,
                                    full_model_kwargs = full_model_kwargs_vis,
                                    use_fps_condition = cfg.use_fps_condition)
                noise_one = noise
                
                if hasattr(cfg, "CPU_CLIP_VAE") and cfg.CPU_CLIP_VAE:
                    clip_encoder.cpu() # add this line
                    autoencoder.cpu() # add this line
                    torch.cuda.empty_cache() # add this line
                    
                video_data = diffusion.ddim_sample_loop(
                    noise=noise_one,
                    model=model.eval(), 
                    model_kwargs=model_kwargs_one,
                    guide_scale=cfg.guide_scale,
                    ddim_timesteps=cfg.ddim_timesteps,
                    eta=0.0)
                
                # print("video_data = diffusion.ddim_sample_", video_data.shape) #torch.Size([1, 4, 32, 96, 64])

                if hasattr(cfg, "CPU_CLIP_VAE") and cfg.CPU_CLIP_VAE:
                    # if run forward of  autoencoder or clip_encoder second times, load them again
                    clip_encoder.cuda()
                    autoencoder.cuda()
                video_data = 1. / cfg.scale_factor * video_data 
                video_data = rearrange(video_data, 'b c f h w -> (b f) c h w')
                chunk_size = min(cfg.decoder_bs, video_data.shape[0])
                video_data_list = torch.chunk(video_data, video_data.shape[0]//chunk_size, dim=0)
                decode_data = []
                for vd_data in video_data_list:
                    gen_frames = autoencoder.decode(vd_data)
                    decode_data.append(gen_frames)
                video_data = torch.cat(decode_data, dim=0)
                video_data = rearrange(video_data, '(b f) c h w -> b c f h w', b = cfg.batch_size).float()
                
                text_size = cfg.resolution[-1]
                cap_name = re.sub(r'[^\w\s]', '', ref_image_key.split("/")[-1].split('.')[0]) # .replace(' ', '_')
                pose_name = re.sub(r'[^\w\s]', '', pose_seq_key.split("/")[-1].split('.')[0])  
                name = f'seed_{cur_seed}'
                file_name = f'{cap_name}_{pose_name}_{name}_rank_{cfg.world_size:02d}_{cfg.rank:02d}_{idx:02d}_{cfg.resolution[1]}x{cfg.resolution[0]}.mp4'
                local_path = os.path.join(cfg.log_dir, f'{file_name}')
                local_path_1col = os.path.join(cfg.log_dir, f'{file_name[:-4]}_results_1col.mp4')
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                captions = "human"
                del model_kwargs_one_vis[0][list(model_kwargs_one_vis[0].keys())[0]]
                del model_kwargs_one_vis[1][list(model_kwargs_one_vis[1].keys())[0]]

                del model_kwargs_one_vis[0]["pose_embeddings"]
                del model_kwargs_one_vis[1]["pose_embeddings"]

                

                save_video_multiple_conditions_not_gif_horizontal_3col(local_path, video_data.cpu(), model_kwargs_one_vis, misc_backups, 
                                                cfg.mean, cfg.std, nrow=1, save_fps=cfg.save_fps)

                save_video_multiple_conditions_not_gif_horizontal_1col(local_path_1col, video_data.cpu(), model_kwargs_one_vis, misc_backups, 
                                                cfg.mean, cfg.std, nrow=1, save_fps=cfg.save_fps)      
                logging.info(f'video saved in {local_path}!')

    
    logging.info('Congratulations! The inference is completed!')
    # synchronize to finish some processes
    if not cfg.debug:
        torch.cuda.synchronize()
        dist.barrier()

def prepare_model_kwargs(partial_keys, full_model_kwargs, use_fps_condition=False):
    
    if use_fps_condition is True:
        partial_keys.append('fps')

    partial_model_kwargs = [{}, {}]
    for partial_key in partial_keys:
        partial_model_kwargs[0][partial_key] = full_model_kwargs[0][partial_key]
        partial_model_kwargs[1][partial_key] = full_model_kwargs[1][partial_key]

    return partial_model_kwargs
