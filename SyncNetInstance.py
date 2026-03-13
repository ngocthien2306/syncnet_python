#!/usr/bin/python
#-*- coding: utf-8 -*-
# Video 25 FPS, Audio 16000HZ

import torch
import numpy
import time, pdb, argparse, subprocess, math
import cv2
import python_speech_features

from SyncNetModel import *


# ==================== Get OFFSET ====================

def calc_pdist(feat1, feat2, vshift=10):
    win_size = vshift * 2 + 1
    feat2p   = torch.nn.functional.pad(feat2, (0, 0, vshift, vshift))

    # Vectorized: build sliding windows in one shot instead of per-frame loop
    # feat2p.unfold(0, win_size, 1) → (T, D, win_size) → permute → (T, win_size, D)
    feat2_win = feat2p.unfold(0, win_size, 1).permute(0, 2, 1)          # (T, W, D)
    feat1_exp = feat1.unsqueeze(1).expand(-1, win_size, -1)              # (T, W, D)
    dists_all = torch.norm(feat1_exp - feat2_win, p=2, dim=2)            # (T, W)

    return [dists_all[i] for i in range(len(feat1))]

# ==================== MAIN DEF ====================

class SyncNetInstance(torch.nn.Module):

    def __init__(self, dropout = 0, num_layers_in_fc_layers = 1024, device = 'cuda'):
        super(SyncNetInstance, self).__init__();

        self.device = device
        self.__S__ = S(num_layers_in_fc_layers = num_layers_in_fc_layers).to(device);

    # ------------------------------------------------------------------
    # CPU phase — pure IO + numpy, no GPU; safe to call in thread pool
    # ------------------------------------------------------------------
    def prepare_data(self, videofile):
        """Read video + audio and compute CPU tensors.  Returns None on error."""

        # Read video frames via VideoCapture
        cap = cv2.VideoCapture(videofile)
        images = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            images.append(frame)
        cap.release()

        if not images:
            return None

        im = numpy.stack(images, axis=3)
        im = numpy.expand_dims(im, axis=0)
        im = numpy.transpose(im, (0, 3, 4, 1, 2))
        imtv_cpu = torch.from_numpy(im.astype(numpy.float32))  # stays on CPU

        # Read audio via ffmpeg pipe
        proc = subprocess.Popen(
            ['ffmpeg', '-y', '-i', videofile, '-vn', '-acodec', 'pcm_s16le',
             '-ar', '16000', '-ac', '1', '-f', 's16le', '-'],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        audio_bytes, _ = proc.communicate()
        audio = numpy.frombuffer(audio_bytes, dtype=numpy.int16).copy()

        if len(audio) == 0:
            return None

        mfcc = zip(*python_speech_features.mfcc(audio, 16000))
        mfcc = numpy.stack([numpy.array(i) for i in mfcc])
        cc = numpy.expand_dims(numpy.expand_dims(mfcc, axis=0), axis=0)
        cct_cpu = torch.from_numpy(cc.astype(numpy.float32))  # stays on CPU

        min_length = min(len(images), math.floor(len(audio) / 640))
        return imtv_cpu, cct_cpu, min_length

    # ------------------------------------------------------------------
    # GPU phase — moves pre-computed CPU tensors to device, runs model
    # ------------------------------------------------------------------
    def evaluate_tensors(self, opt, imtv_cpu, cct_cpu, min_length):
        """GPU-only inference on tensors prepared by prepare_data()."""
        imtv_dev = imtv_cpu.to(self.device)
        cct_dev  = cct_cpu.to(self.device)

        lastframe = min_length - 5
        im_feat, cc_feat = [], []

        for i in range(0, lastframe, opt.batch_size):
            im_batch = [imtv_dev[:, :, v:v+5, :, :] for v in range(i, min(lastframe, i+opt.batch_size))]
            im_out   = self.__S__.forward_lip(torch.cat(im_batch, 0))
            im_feat.append(im_out.data.cpu())

            cc_batch = [cct_dev[:, :, :, v*4:v*4+20] for v in range(i, min(lastframe, i+opt.batch_size))]
            cc_out   = self.__S__.forward_aud(torch.cat(cc_batch, 0))
            cc_feat.append(cc_out.data.cpu())

        im_feat = torch.cat(im_feat, 0)
        cc_feat = torch.cat(cc_feat, 0)

        dists  = calc_pdist(im_feat, cc_feat, vshift=opt.vshift)
        mdist  = torch.mean(torch.stack(dists, 1), 1)

        minval, minidx = torch.min(mdist, 0)
        offset = opt.vshift - minidx
        conf   = torch.median(mdist) - minval

        dists_npy = numpy.array([dist.numpy() for dist in dists])
        return offset.numpy(), conf.numpy(), dists_npy

    # ------------------------------------------------------------------
    # Convenience wrapper (backward-compatible)
    # ------------------------------------------------------------------
    def evaluate(self, opt, videofile):
        data = self.prepare_data(videofile)
        if data is None:
            vshift = getattr(opt, 'vshift', 15)
            return numpy.array(0), numpy.array(0.0), numpy.zeros((1, vshift * 2 + 1))
        return self.evaluate_tensors(opt, *data)

    def extract_feature(self, opt, videofile):

        self.__S__.eval();
        
        # ========== ==========
        # Load video 
        # ========== ==========
        cap = cv2.VideoCapture(videofile)

        frame_num = 1;
        images = []
        while frame_num:
            frame_num += 1
            ret, image = cap.read()
            if ret == 0:
                break

            images.append(image)

        im = numpy.stack(images,axis=3)
        im = numpy.expand_dims(im,axis=0)
        im = numpy.transpose(im,(0,3,4,1,2))

        imtv = torch.autograd.Variable(torch.from_numpy(im.astype(float)).float())
        
        # ========== ==========
        # Generate video feats
        # ========== ==========

        lastframe = len(images)-4
        im_feat = []

        tS = time.time()
        for i in range(0,lastframe,opt.batch_size):
            
            im_batch = [ imtv[:,:,vframe:vframe+5,:,:] for vframe in range(i,min(lastframe,i+opt.batch_size)) ]
            im_in = torch.cat(im_batch,0)
            im_out  = self.__S__.forward_lipfeat(im_in.cuda());
            im_feat.append(im_out.data.cpu())

        im_feat = torch.cat(im_feat,0)

        # ========== ==========
        # Compute offset
        # ========== ==========
            
        print('Compute time %.3f sec.' % (time.time()-tS))

        return im_feat


    def loadParameters(self, path):
        loaded_state = torch.load(path, map_location=lambda storage, loc: storage);

        self_state = self.__S__.state_dict();

        for name, param in loaded_state.items():

            self_state[name].copy_(param);
