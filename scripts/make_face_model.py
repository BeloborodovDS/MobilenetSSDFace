import sys
import cv2
import numpy as np
import caffe
from caffe.proto.caffe_pb2 import NetParameter, LayerParameter
import google.protobuf.text_format as txtf

def transform_input(img, transpose=True, dtype=np.float32):
    inpt = cv2.resize(img, (300,300))
    inpt = inpt - 127.5
    inpt = inpt / 127.5
    inpt = inpt.astype(dtype)
    if transpose:
        inpt = inpt.transpose((2, 0, 1))
    return inpt

def get_masks(net, percentile=50):
    bnames = [e for e in net.blobs.keys() if ('data' not in e) and ('split' not in e) 
              and ('mbox' not in e) and ('detection' not in e)]
    blobmask = {}
    prev = None
    for b in bnames:
        blob = net.blobs[b].data
        mean = blob.mean(axis=(0,2,3))
        perc = np.percentile(mean, percentile)
        mask = mean>perc
        blobmask[b] = mask
        if ('dw' in b) and (prev is not None):
            blobmask[prev] = mask
        prev = b
    return blobmask

def resize_network(netdef, name2num, verbose=True):
    new_layers = []
    for l in netdef.layer:
        newl = LayerParameter()
        newl.CopyFrom(l)
        if (l.name in name2num):
            if (l.type == 'Convolution'):
                if verbose:
                    print(l.name+': \t'+
                          'Changing num_output from '+str(l.convolution_param.num_output)+' to '+str(name2num[l.name]))
                newl.convolution_param.num_output = name2num[l.name]
                if newl.convolution_param.group > 1:
                    newl.convolution_param.group = name2num[l.name]
            else:
                if verbose:
                    print('Layer '+l.name+' is not convolution, skipping')
        new_layers.append(newl)
    new_pnet = NetParameter()
    new_pnet.CopyFrom(netdef)
    del(new_pnet.layer[:])
    new_pnet.layer.extend(new_layers)
    return new_pnet

def set_params(model, newmodel, newnetdef, blob2mask):
    l2bot = {l.name:l.bottom for l in newnetdef.layer}
    l2top = {l.name:l.top for l in newnetdef.layer}
    l2group = {l.name:l.convolution_param.group for l in newnetdef.layer}
    
    for name in model.params.keys():
        if 'mbox' in name:
            continue
        top = l2top[name][0]
        bot = l2bot[name][0]
        topmask = blob2mask[top] if top in blob2mask else None
        botmask = blob2mask[bot] if bot in blob2mask else None
        conv = model.params[name][0].data
        bias = model.params[name][1].data
        if topmask is not None:
            conv = conv[topmask,:,:,:]
            bias = bias[topmask]
        if (botmask is not None) and (l2group[name]==1):
            conv = conv[:,botmask,:,:]
        newmodel.params[name][0].data[...] = conv
        if name+'/scale' in newmodel.params:
            newmodel.params[name+'/scale'][1].data[...] = bias
        else:
            newmodel.params[name][1].data[...] = bias
            
if __name__ == "__main__":
    percentile = int(sys.argv[1])    
    
    ref_net = caffe.Net('models/ssd_voc/deploy.prototxt', 
                    'models/ssd_voc/MobileNetSSD_deploy.caffemodel', 
                    caffe.TEST) 

    with open('models/ssd_voc/deploy.prototxt', 'r') as f:
        ref_par = NetParameter()
        txtf.Merge(f.read(), ref_par)
        
    with open('models/ssd_face/ssd_face_train.prototxt', 'r') as f:
        train_par = NetParameter()
        txtf.Merge(f.read(), train_par)   
    with open('models/ssd_face/ssd_face_test.prototxt', 'r') as f:
        test_par = NetParameter()
        txtf.Merge(f.read(), test_par)  
    with open('models/ssd_face/ssd_face_deploy.prototxt', 'r') as f:
        dep_par = NetParameter()
        txtf.Merge(f.read(), dep_par)
        
    faces = cv2.imread('images/faces.png')
    
    inpt = transform_input(faces)
    ref_net.blobs['data'].data[...] = inpt
    output = ref_net.forward()
    
    blobmask = get_masks(ref_net, percentile)
    sizes = {k:sum(v) for k,v in blobmask.items()}
    
    train_par = resize_network(train_par, sizes)
    test_par = resize_network(test_par, sizes, verbose=False)
    dep_par = resize_network(dep_par, sizes, verbose=False)
    
    with open('models/ssd_face_pruned/face_train.prototxt', 'w') as f:
        f.write(txtf.MessageToString(train_par))
    with open('models/ssd_face_pruned/face_test.prototxt', 'w') as f:
        f.write(txtf.MessageToString(test_par))
    with open('models/ssd_face_pruned/face_deploy.prototxt', 'w') as f:
        f.write(txtf.MessageToString(dep_par))
        
    new_net = caffe.Net('models/ssd_face_pruned/face_train.prototxt', 
                        'models/empty.caffemodel', caffe.TRAIN)
    
    set_params(ref_net, new_net, train_par, blobmask)
    
    new_net.save('models/ssd_face_pruned/face_init.caffemodel')