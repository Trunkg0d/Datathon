import torch
from models.dior_model import DIORModel
import os
import matplotlib.pyplot as plt
import numpy as np
from datasets.deepfashion_datasets import DFVisualDataset

dataroot = 'data'
exp_name = 'DIORv1_64' # DIOR_64
epoch = 'latest'
netG = 'diorv1' # dior
ngf = 64
absolute_dir = "D:\Datathon\dressing-in-order\output"

## this is a dummy "argparse"
class Opt:
    def __init__(self):
        pass
if True:
    opt = Opt()
    opt.dataroot = dataroot
    opt.isTrain = False
    opt.phase = 'test'
    opt.n_human_parts = 8; opt.n_kpts = 18; opt.style_nc = 64
    opt.n_style_blocks = 4; opt.netG = netG; opt.netE = 'adgan'
    opt.ngf = ngf
    opt.norm_type = 'instance'; opt.relu_type = 'leakyrelu'
    opt.init_type = 'orthogonal'; opt.init_gain = 0.02; opt.gpu_ids = [0]
    opt.frozen_flownet = True; opt.random_rate = 1; opt.perturb = False; opt.warmup=False
    opt.name = exp_name
    opt.vgg_path = ''; opt.flownet_path = ''
    opt.checkpoints_dir = 'checkpoints'
    opt.frozen_enc = True
    opt.load_iter = 0
    opt.epoch = epoch
    opt.verbose = False

# create model
model = DIORModel(opt)
model.setup(opt)

# load data
Dataset = DFVisualDataset
ds = Dataset(dataroot=dataroot, dim=(256,176), n_human_part=8)

# # preload a set of pre-selected models defined in "standard_test_anns.txt" for quick visualizations
inputs = dict()
# for attr in ds.attr_keys:
#     inputs[attr] = ds.get_attr_visual_input(attr)

# define some tool functions for I/O
def load_img(pid, ds):
    if len(pid[0]) < 10: # load pre-selected models
        person = inputs[pid[0]]
        person = (i.cuda() for i in person)
        pimg, parse, to_pose = person
        pimg, parse, to_pose = pimg[pid[1]], parse[pid[1]], to_pose[pid[1]]
    else: # load model from scratch
        person = ds.get_inputs_by_key(pid[0])
        person = (i.cuda() for i in person)
        pimg, parse, to_pose = person
    return pimg.squeeze(), parse.squeeze(), to_pose.squeeze()

def plot_img(pimg=[], gimgs=[], oimgs=[], gen_img=[], pose=None):
    if pose != None:
        import utils.pose_utils as pose_utils
        # print(pose.size())
        kpt = pose_utils.draw_pose_from_map(pose.cpu().numpy().transpose(1,2,0),radius=6)
        kpt = kpt[0]
    if not isinstance(pimg, list):
        pimg = [pimg]
    if not isinstance(gen_img, list):
        gen_img = [gen_img]
    out = pimg + gimgs + oimgs + gen_img
    if out:
        out = torch.cat(out, 2).float().cpu().detach().numpy()
        out = (out + 1) / 2 # denormalize
        out = np.transpose(out, [1,2,0])

        if pose != None:
            out = np.concatenate((kpt, out),1)
    else:
        out = kpt
    fig = plt.figure(figsize=(6,4), dpi= 100, facecolor='w', edgecolor='k')
    plt.axis('off')
    plt.imshow(out)
    plt.savefig("out")

def plot_gen_img(pimg=[], gimgs=[], oimgs=[], gen_img=[], pose=None, dir="", abs_dir = absolute_dir):
    if pose != None:
        import utils.pose_utils as pose_utils
        # print(pose.size())
        kpt = pose_utils.draw_pose_from_map(pose.cpu().numpy().transpose(1,2,0),radius=6)
        kpt = kpt[0]
    if not isinstance(pimg, list):
        pimg = [pimg]
    if not isinstance(gen_img, list):
        gen_img = [gen_img]
    out = gen_img
    if out:
        out = torch.cat(out, 2).float().cpu().detach().numpy()
        out = (out + 1) / 2 # denormalize
        out = np.transpose(out, [1,2,0])
    else:
        out = kpt
    fig = plt.figure(figsize=(6,4), dpi= 100, facecolor='w', edgecolor='k')
    plt.axis('off')
    plt.imshow(out)
    abs_dir += dir
    plt.savefig(abs_dir)

# define dressing-in-order function (the pipeline)
def dress_in_order(model, pid, pose_id=None, gids=[], ogids=[], order=[5,1,3,2]):
    PID = [0,4,6,7]
    GID = [2,5,1,3]
    # encode person
    pimg, parse, from_pose = load_img(pid, ds)
    if not pose_id:
        to_pose = from_pose
    else:
        to_img, _, to_pose = load_img(pose_id, ds)
    psegs = model.encode_attr(pimg[None], parse[None], from_pose[None], to_pose[None], PID)

    # encode base garments
    gsegs = model.encode_attr(pimg[None], parse[None], from_pose[None], to_pose[None])


    # swap base garment if any
    gimgs = []
    for gid in gids:
        _,_,k = gid
        gimg, gparse, pose =  load_img(gid, ds)
        seg = model.encode_single_attr(gimg[None], gparse[None], pose[None], to_pose[None], i=gid[2])
        gsegs[gid[2]] = seg
        gimgs += [gimg * (gparse == gid[2])]

    # encode garment (overlay)
    garments = []
    over_gsegs = []
    oimgs = []
    for gid in ogids:
        oimg, oparse, pose = load_img(gid, ds)
        oimgs += [oimg * (oparse == gid[2])]
        seg = model.encode_single_attr(oimg[None], oparse[None], pose[None], to_pose[None], i=gid[2])
        over_gsegs += [seg]

    gsegs = [gsegs[i] for i in order] + over_gsegs
    gen_img = model.netG(to_pose[None], psegs, gsegs)

    return pimg, gimgs, oimgs, gen_img[0], to_pose

def pose_transfer(pid, pose_id, dir=""):
    # generate
    pimg, gimgs, oimgs, gen_img, pose = dress_in_order(model, pid, pose_id=pose_id)
    dir += ("/" + pid[0])
    plot_gen_img(pimg, gimgs, oimgs, gen_img, pose, dir=dir)

# pose_transfer_dir = "\pose_transfer"
# pid = ("test_image_1.jpg", None, None)
# pose_id = ("test_image_1.jpg", None, None)
# pose_transfer(pid=pid, pose_id=pose_id, dir=pose_transfer_dir)

def tucking_in_out(pid, gids, tuking_in=0, dir=""):
    if tuking_in:
        # tuck in (dressing order: hair, top, bottom)
        pimg, gimgs, oimgs, gen_img, pose = dress_in_order(model, pid, gids=gids, order=[2, 5, 1])
    else:
        # not tuckin (dressing order: hair, bottom, top)
        pimg, gimgs, oimgs, gen_img, pose = dress_in_order(model, pid, gids=gids, order=[2, 1, 5])
    dir += ("/" + pid[0])
    plot_gen_img(pimg, gimgs, gen_img=gen_img, pose=pose, dir=dir)

# tuking_dir = "\layer_tucking_in_out"
# pid = ("test_image_1.jpg", None, None)
# gids = [("test_image_1.jpg",0,5), ("test_image_1.jpg",3,1)]
# tucking_in_out(pid=pid, gids=gids, tuking_in=1, dir=tuking_dir)

def layering(pid, gids, ogids=[], multi_layer=0, dir=""):
    dir += ("/" + pid[0])
    if multi_layer:
        pimg, gimgs, oimgs, gen_img, pose = dress_in_order(model, pid, gids=gids, ogids=ogids)
    else:
        pimg, gimgs, oimgs, gen_img, pose = dress_in_order(model, pid, gids=gids)

    plot_gen_img(pimg, gimgs, oimgs, gen_img, pose, dir=dir)

# layering_dir = "\layering"
# pid = ('test_image_1.jpg',None, None)
# gids = [('garment_nike.jpg', None, 5)]
# layering(pid, gids, dir=layering_dir)

multi_layering_dir = "\multi_layering"
pid = ('fashionMENDenimid0000056501_7additional.jpg', None, None)
gids = [('garment_nike.jpg', None, 5), ('pant_test_2.jpg', None, 1)]
ogids = [('fashionMENDenimid0000056501_7additional.jpg', None, 5)]
layering(pid, gids, multi_layer=1, dir=multi_layering_dir)

