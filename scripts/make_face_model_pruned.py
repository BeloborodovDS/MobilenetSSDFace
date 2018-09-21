import caffe
from caffe.proto.caffe_pb2 import NetParameter, LayerParameter
import google.protobuf.text_format as txtf
import numpy as np
from collections import OrderedDict
from matplotlib import pyplot as plt

ref_net = caffe.Net('models/ssd_face/ssd_face_deploy_bn.prototxt', 
                    'models/ssd_face/best_bn_full.caffemodel', caffe.TRAIN)

l2prevmask = OrderedDict()
l2nextmask = OrderedDict()
l2mask = OrderedDict()
prev_mask = None
block = []
eps = 1e-6
for l in ref_net.params.keys():
    if 'scale' in l:
        mask = np.abs(ref_net.params[l][0].data) < eps
        plt.plot(np.sort(np.abs(ref_net.params[l][0].data)), 'bx-')
        plt.savefig('./models/tmp/'+l.replace('/','_')+'.png')
        plt.clf()
        if prev_mask is not None:
            prev_mask = np.logical_or(prev_mask, mask)
        else:
            prev_mask = mask
        block.append(l)
    elif 'bn' in l or 'dw' in l:
        block.append(l)
    else:
        if prev_mask is not None:
            for e in block:
                if 'scale' in e or 'bn' in e or 'dw' in e:
                    l2mask[e] = prev_mask
                else:
                    l2nextmask[e] = prev_mask
                    l2mask[e] = None
            l2prevmask[l] = prev_mask
        else:
            l2prevmask[l] = np.zeros((ref_net.params[l][0].data.shape[1],)).astype(bool)
        l2nextmask[l] = np.zeros((ref_net.params[l][0].data.shape[0],)).astype(bool)
        prev_mask = None
        block = []
        block.append(l)
        if 'mbox' in l:
            l2nextmask[l] = np.zeros((ref_net.params[l][0].data.shape[0],)).astype(bool)
            bottom = l.replace('_mbox_loc','').replace('_mbox_conf','')
            l2prevmask[l] = l2nextmask[bottom]

for l in ref_net.params.keys():
    if 'scale' in l or 'bn' in l or 'dw' in l:
        l2mask[l] = np.logical_not(l2mask[l])
    else:
        l2mask[l] = (np.logical_not(l2prevmask[l]), np.logical_not(l2nextmask[l]))
               
for mode in ['train','test','deploy','deploy_bn']:
    with open('models/ssd_face/ssd_face_'+mode+'.prototxt', 'r') as f:
        net_par = NetParameter()
        txtf.Merge(f.read(), net_par)
        
    new_layers = []
    for l in net_par.layer:
        newl = LayerParameter()
        newl.CopyFrom(l)
        
        if newl.name in l2mask:
            if 'bn' in newl.name or 'scale' in newl.name:
                pass
            elif 'dw' in newl.name:
                newl.convolution_param.num_output = sum(l2mask[newl.name])
                newl.convolution_param.group = sum(l2mask[newl.name])
            else:
                newl.convolution_param.num_output = sum(l2mask[newl.name][1])
        
        if l.name in {'mbox_loc','mbox_conf','mbox_priorbox'}:
            newbot = [e for e in l.bottom if ('16' not in e) and ('17' not in e)]
            del(newl.bottom[:])
            newl.bottom.extend(newbot)
            new_layers.append(newl)
        elif ('16' not in l.name) and ('17' not in l.name):
            new_layers.append(newl)
        
    newnet_par = NetParameter()
    newnet_par.CopyFrom(net_par)
    del(newnet_par.layer[:])
    newnet_par.layer.extend(new_layers)
    
    with open('models/ssd_face_pruned/face_'+mode+'.prototxt', 'w') as f:
        f.write(txtf.MessageToString(newnet_par))
        
new_net = caffe.Net('models/ssd_face_pruned/face_deploy_bn.prototxt', 
                    'models/empty.caffemodel', caffe.TEST)
                    
for l in new_net.params.keys():
    if 'mbox' in l:
        inmask, outmask = l2mask[l]
        new_net.params[l][0].data[...] = ref_net.params[l][0].data[outmask,:,:,:][:,inmask,:,:]
        new_net.params[l][1].data[...] = ref_net.params[l][1].data[outmask]
    elif 'scale' in l:
        mask = l2mask[l]
        new_net.params[l][0].data[...] = ref_net.params[l][0].data[mask]
        new_net.params[l][1].data[...] = ref_net.params[l][1].data[mask]
    elif 'bn' in l:
        mask = l2mask[l]
        new_net.params[l][0].data[...] = ref_net.params[l][0].data[mask]
        new_net.params[l][1].data[...] = ref_net.params[l][1].data[mask]
        new_net.params[l][2].data[...] = ref_net.params[l][2].data[...]
    elif 'dw' in l:
        mask = l2mask[l]
        new_net.params[l][0].data[...] = ref_net.params[l][0].data[mask,:,:,:]
    else:
        inmask, outmask = l2mask[l]
        new_net.params[l][0].data[...] = ref_net.params[l][0].data[outmask,:,:,:][:,inmask,:,:]

#save pruned net parameters
new_net.save('models/ssd_face_pruned/short_init.caffemodel')


for l, v in l2mask.items():
    if '16' not in l and '17' not in l:
        if ('bn' in l) or ('scale' in l):
            pass
        else:
            if 'dw' in l: 
                if v.size != sum(v):
                    print(l,'\t({0},) --> ({1},)'.format(v.size,sum(v)))
            else: 
                if (v[1].size != sum(v[1])) or (v[0].size != sum(v[0])):
                    print(l,'\t({0},{1}) --> ({2},{3})'.format(v[1].size,v[0].size,sum(v[1]),sum(v[0])))

print('\nDeleting layers 16-17')
