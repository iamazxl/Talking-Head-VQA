import argparse
import os

import numpy as np

import torch
import torch.nn
from torchvision import transforms
from torch.utils import data

from model import TB_DHQA

from utils import performance_fit

import pandas as pd
import cv2
import scipy.io as scio

from PIL import Image

from extract_slowfast_features_THQA import slowfast,pack_pathway_output

#--框架提取--
def extract_frame(videos_dir, video_name,save_folder):
    filename = os.path.join(videos_dir, video_name)
    # print("filename:",filename)

    video_name_str = video_name[:-4]
    video_capture = cv2.VideoCapture()
    video_capture.open(filename)
    cap=cv2.VideoCapture(filename)

    
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_frame_rate = int(round(cap.get(cv2.CAP_PROP_FPS)))
    
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # the heigh of frames
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # the width of frames
    
    if video_height > video_width:
        video_width_resize = 520
        video_height_resize = int(video_width_resize/video_width*video_height)
    else:
        video_height_resize = 520
        video_width_resize = int(video_height_resize/video_height*video_width)
        
    dim = (video_width_resize, video_height_resize)

    video_read_index = 0

    frame_idx = 0
    
    video_length_min = 8
    
    for i in range(video_length):
        has_frames, frame = video_capture.read()
        if has_frames:
            # key frame
            if (video_read_index < video_length) and (frame_idx % video_frame_rate == 0):
                read_frame = cv2.resize(frame, dim)
                exit_folder(os.path.join(save_folder, video_name_str))
                cv2.imwrite(os.path.join(save_folder, video_name_str, \
                                         '{:03d}'.format(video_read_index) + '.png'), read_frame)          
                video_read_index += 1
            frame_idx += 1

    if video_read_index < video_length_min:
        for i in range(video_read_index, video_length_min):
            cv2.imwrite(os.path.join(save_folder, video_name_str, \
                                     '{:03d}'.format(i) + '.png'), read_frame)

    return

def exit_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)    
        
    return


#--特征提取--
class TB_VideoFeature_Extract(data.Dataset):
    """Read data from the original dataset for feature extraction"""
    def __init__(self, data_dir, filename, transform, resize):
        super(TB_VideoFeature_Extract, self).__init__()

        self.video_names = filename
        self.video_dir = data_dir
        self.transform = transform
        self.resize = resize
        self.length = 1

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        video_name = self.video_names
        filename=os.path.join(self.video_dir, video_name)
        video_capture = cv2.VideoCapture()
        video_capture.open(filename)
        cap=cv2.VideoCapture(filename)

        video_channel = 3
        
        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_frame_rate = int(round(cap.get(cv2.CAP_PROP_FPS)))

        video_clip = int(video_length/video_frame_rate)
       
        video_clip_min = 8

        video_length_clip = 32

        transformed_frame_all = torch.zeros([video_length, video_channel, self.resize, self.resize])

        transformed_video_all = []
        
        video_read_index = 0
        for i in range(video_length):
            has_frames, frame = video_capture.read()
            if has_frames:
                read_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                read_frame = self.transform(read_frame)
                transformed_frame_all[video_read_index] = read_frame
                video_read_index += 1


        if video_read_index < video_length:
            for i in range(video_read_index, video_length):
                transformed_frame_all[i] = transformed_frame_all[video_read_index - 1]
 
        video_capture.release()

        for i in range(video_clip):
            transformed_video = torch.zeros([video_length_clip, video_channel, self.resize, self.resize])
            if (i*video_frame_rate + video_length_clip) <= video_length:
                transformed_video = transformed_frame_all[i*video_frame_rate : (i*video_frame_rate + video_length_clip)]
            else:
                transformed_video[:(video_length - i*video_frame_rate)] = transformed_frame_all[i*video_frame_rate :]
                for j in range((video_length - i*video_frame_rate), video_length_clip):
                    transformed_video[j] = transformed_video[video_length - i*video_frame_rate - 1]
            transformed_video_all.append(transformed_video)

        if video_clip < video_clip_min:
            for i in range(video_clip, video_clip_min):
                transformed_video_all.append(transformed_video_all[video_clip - 1])
       
        return transformed_video_all, video_name


class TB_MotionFeature_Extract(data.Dataset):
    """Read data from the original dataset for feature extraction"""
    def __init__(self, data_dir, data_dir_3D ,filename, transform, database_name, crop_size, feature_type):
        super(TB_MotionFeature_Extract, self).__init__()
        if database_name == 'THQA':
            self.video_name = filename
        
        self.crop_size = crop_size
        self.videos_dir = data_dir
        self.data_dir_3D = data_dir_3D
        self.transform = transform
        self.length = 1
        self.feature_type = feature_type
        self.database_name = database_name
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        video_name = self.video_name
        video_name_str = video_name[:-4]
        path_name = os.path.join(self.videos_dir, video_name_str)
        
        video_channel = 3

        video_height_crop = self.crop_size
        video_width_crop = self.crop_size
        
        video_length_read = 8
        
        transformed_video = torch.zeros([video_length_read, video_channel, video_height_crop, video_width_crop])
        
        for i in range(video_length_read):
            imge_name = os.path.join(path_name, '{:03d}'.format(i) + '.png')
            read_frame = Image.open(imge_name)
            read_frame = read_frame.convert('RGB')
            read_frame = self.transform(read_frame)
            transformed_video[i] = read_frame
            
        if self.feature_type == 'SlowFast':
            feature_folder_name = os.path.join(self.data_dir_3D, video_name_str)
            transformed_feature = torch.zeros([video_length_read, 2048+256])
            for i in range(video_length_read):
                i_index = i
                feature_3D_slow = np.load(os.path.join(feature_folder_name+".mp4", 'feature_' + str(i_index) + '_slow_feature.npy'))
                feature_3D_slow = torch.from_numpy(feature_3D_slow)
                feature_3D_slow = feature_3D_slow.squeeze()
                feature_3D_fast = np.load(os.path.join(feature_folder_name+".mp4", 'feature_' + str(i_index) + '_fast_feature.npy'))
                feature_3D_fast = torch.from_numpy(feature_3D_fast)
                feature_3D_fast = feature_3D_fast.squeeze()
                feature_3D = torch.cat([feature_3D_slow, feature_3D_fast])
                transformed_feature[i] = feature_3D

        #print("test_loader:",transformed_video, "  ",transformed_feature,"  ", video_name)
        return transformed_video, transformed_feature, video_name

            
    

        
        
def main(config):
    
    videos_dir = "videos"
    video_name = "1_voice_en_35042678_mkittalk.mp4"
    save_folder = os.path.join(videos_dir,"frames")
    # print("save_folder:",save_folder)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #提取frame
    # print('start extract video: {}'.format(video_name))
    extract_frame(videos_dir, video_name, save_folder)
    # print("finished extract frame!!!")
    
    #提取特征
    # print("extracting featrue!!!")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ex_model = slowfast()
    ex_model = ex_model.to(device)
    
    resize = config.resize
    feature_save_folder = os.path.join(videos_dir,'feature/')

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transformations_test = transforms.Compose([transforms.Resize([resize, resize]),transforms.ToTensor(),\
        transforms.Normalize(mean = [0.45, 0.45, 0.45], std = [0.225, 0.225, 0.225])])
    trainset = TB_VideoFeature_Extract(videos_dir, video_name, transformations_test, resize)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=1,
        shuffle=False, num_workers=config.num_workers)
        
    
    
    with torch.no_grad():
        ex_model.eval()

        for i, (video, video_name) in enumerate(train_loader):
            video_name = video_name[0]
            print(video_name)

            if not os.path.exists(feature_save_folder + video_name):
                os.makedirs(feature_save_folder + video_name)
            
            for idx, ele in enumerate(video):
                # ele = ele.to(device)
                ele = ele.permute(0, 2, 1, 3, 4)
                inputs = pack_pathway_output(ele, device)
                slow_feature, fast_feature = ex_model(inputs)
                np.save(feature_save_folder + video_name + '/' + 'feature_' + str(idx) + '_slow_feature', slow_feature.to('cpu').numpy())
                np.save(feature_save_folder + video_name + '/' + 'feature_' + str(idx) + '_fast_feature', fast_feature.to('cpu').numpy())

    
    
    #print("finished extracting featrue!!!")
    

    model = TB_DHQA.resnet50(pretrained=False)


    model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)
    model = model.to(device)

    # load the trained model

    #print('loading the trained model')
    #model.load_state_dict(torch.load(config.trained_model))
    state_dict = torch.load(config.trained_model)
    state_dict = {f'module.{k}': v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    #print("finished loading model")


    #feature_dir = os.path.join(config.data_path, 'feature/')
    
    
    #print("load data")
    
    transformations_test = transforms.Compose([transforms.Resize(config.resize),transforms.CenterCrop(config.crop_size),transforms.ToTensor(),\
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
    
    testset = TB_MotionFeature_Extract(save_folder, feature_save_folder, video_name , transformations_test, 'THQA', config.crop_size, 'SlowFast')
  
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1,
        shuffle=False, num_workers=config.num_workers)
    
    # print("finish loading")
    
    

    with torch.no_grad():
        model.eval()  
        #label = np.zeros([len(testset)])
        y_output = np.zeros([len(testset)])
        videos_name = []
        for i,(video, feature_3D, video_name) in enumerate(test_loader):
            print(video_name)
            videos_name.append(video_name)
            video = video.to(device)
            feature_3D = feature_3D.to(device)
            #label[i] = mos.item()
            outputs = model(video, feature_3D)

            y_output = outputs.item()
    print("The score of this video:",y_output)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # input parameters
    parser.add_argument('--database', type=str,default= 'THQA')
    parser.add_argument('--train_database', type=str)

    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--resize', type=int, default=256)
    parser.add_argument('--crop_size', type=int, default=448)

    # misc
    parser.add_argument('--trained_model', type=str, default='tb_THQA800_SRCC_0.806901.pth')
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--feature_type', type=str)
    parser.add_argument('--multi_gpu', action='store_true')
    parser.add_argument('--gpu_ids', type=list, default=None)

    
    config = parser.parse_args()

    main(config)



