import numpy as np  
import sys
import caffe  

def merge_bn(net, nob):
    '''
    merge the batchnorm, scale layer weights to the conv layer, to  improve the performance
    var = var + scaleFactor
    rstd = 1. / sqrt(var + eps)
    w = w * rstd * scale
    b = (b - mean) * rstd * scale + shift
    '''
    for key in net.params.keys():
        if type(net.params[key]) is caffe._caffe.BlobVec:
            if key.endswith("/bn") or key.endswith("/scale"):
                continue
            else:
                conv = net.params[key]
                if not key + "/bn" in net.params:
                    for i, w in enumerate(conv):
                        nob.params[key][i].data[...] = w.data
                else:
                    bn = net.params[key + "/bn"]
                    scale = net.params[key + "/scale"]
                    wt = conv[0].data
                    channels = wt.shape[0]
                    bias = np.zeros(wt.shape[0])
                    if len(conv) > 1:
                        bias = conv[1].data
                    mean = bn[0].data
                    var = bn[1].data
                    scalef = bn[2].data

                    scales = scale[0].data
                    shift = scale[1].data

                    if scalef != 0:
                        scalef = 1. / scalef
                    mean = mean * scalef
                    var = var * scalef
                    rstd = 1. / np.sqrt(var + 1e-5)
                    rstd1 = rstd.reshape((channels,1,1,1))
                    scales1 = scales.reshape((channels,1,1,1))
                    wt = wt * rstd1 * scales1
                    bias = (bias - mean) * rstd * scales + shift
                    
                    nob.params[key][0].data[...] = wt
                    nob.params[key][1].data[...] = bias
  
train_proto = sys.argv[1]  
train_model = sys.argv[2]

deploy_proto = sys.argv[3]  
save_model = sys.argv[4]

net = caffe.Net(train_proto, train_model, caffe.TRAIN)  
net_deploy = caffe.Net(deploy_proto, caffe.TEST)  

merge_bn(net, net_deploy)
net_deploy.save(save_model)

