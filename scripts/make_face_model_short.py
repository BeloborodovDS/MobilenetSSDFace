import caffe
from caffe.proto.caffe_pb2 import NetParameter, LayerParameter
import google.protobuf.text_format as txtf
import sys

pattern = sys.argv[1:]
print(pattern)

ref_net = caffe.Net('models/ssd_face/ssd_face_deploy_bn.prototxt', 
                    'models/ssd_face/best_bn_full.caffemodel', caffe.TRAIN)
               
for mode in ['train','test','deploy','deploy_bn']:
    with open('models/ssd_face/ssd_face_'+mode+'.prototxt', 'r') as f:
        net_par = NetParameter()
        txtf.Merge(f.read(), net_par)
        
    new_layers = []
    for l in net_par.layer:
        newl = LayerParameter()
        newl.CopyFrom(l)
        
        if l.name in {'mbox_loc','mbox_conf','mbox_priorbox'}:
            newbot = [e for e in l.bottom if (('14' not in e) and ('15' not in e) and 
                                                ('16' not in e) and ('17' not in e))]
            del(newl.bottom[:])
            newl.bottom.extend(newbot)
            new_layers.append(newl)
        elif (('14' not in l.name) and ('15' not in l.name) and 
                ('16' not in l.name) and ('17' not in l.name)):
            new_layers.append(newl)
        
    newnet_par = NetParameter()
    newnet_par.CopyFrom(net_par)
    del(newnet_par.layer[:])
    newnet_par.layer.extend(new_layers)
    
    with open('models/ssd_face_pruned/face_'+mode+'.prototxt', 'w') as f:
        f.write(txtf.MessageToString(newnet_par))
        
new_net = caffe.Net('models/ssd_face_pruned/face_deploy_bn.prototxt', 
                    'models/ssd_face/best_bn_full.caffemodel', caffe.TEST)

#save pruned net parameters
new_net.save('models/ssd_face_pruned/short_init.caffemodel')

print('\nDeleting layers 14-17')
