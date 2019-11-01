#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 11:52:48 2017

@author: wuyiming
"""

import util


"""
Code example for training U-Net
"""

"""
import network

Xlist,Ylist = util.LoadDataset(target="vocal")
print("Dataset loaded.")
network.TrainUNet(Xlist,Ylist,savefile="unet.model",epoch=30)
"""


"""
Code example for performing vocal separation with U-Net
"""
import glob
fl = glob.glob('audio/*')
fname = fl[1]
mag, phase = util.LoadAudio('audio/01 Calling (2).wav')
start = 0
end = 2048+1024

mask = util.ComputeMask(mag[:, start:end], unet_model="unet.model", hard=False)

util.SaveAudio(
    "%s-vocal.wav" % fname, mag[:, start:end]*mask, phase[:, start:end])
util.SaveAudio(
    "%s-inst.wav" % fname, mag[:, start:end]*(1-mask), phase[:, start:end])
util.SaveAudio(
    "%s-orig.wav" % fname, mag[:, start:end], phase[:, start:end])
