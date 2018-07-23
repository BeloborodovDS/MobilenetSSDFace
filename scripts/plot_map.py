import subprocess
import os, sys
from matplotlib import pyplot as plt

solver = sys.argv[1]

files = os.listdir('snapshots')
with open('train_files/weights.txt') as f:
    snapshot = f.read().strip()
snapshot = snapshot.split('/')[-1]
snapshot = '_'.join(snapshot.split('_')[:-2])

old_cache = {}
if snapshot+'_cache.txt' in files:
    with open('snapshots/'+snapshot+'_cache.txt') as f:
        t = f.read().strip('\n').split('\n')
        t = [e.split(':') for e in t]
        t = [(int(e[0]),float(e[1])) for e in t]
        old_cache.update(t)
cache = {}

with open('Makefile') as f:
    caf = f.read().split('\n')
caf = [e for e in caf if e.startswith('caffe_exec := ')][0]
caf = caf[len('caffe_exec := '):]

files = [e for e in files if e.startswith(snapshot+'_iter_') and e.endswith('.caffemodel')]
iters = [int(e[len(snapshot+'_iter_'):-len('.caffemodel')]) for e in files]
fi = sorted(list(zip(files,iters)), key=lambda x: x[1])
iters = [e[1] for e in fi]
maps = []

for fn, it in fi:
    print(fn)
    if it in old_cache:
        val = old_cache[it]
        print('\t(cached)\t'+str(val))
    else:
        command = caf + ' train -solver ' + solver + ' -weights snapshots/' + fn
        #command = caf + ' train -solver train_files/solver_test.prototxt -weights snapshots/'+fn
        #command = caf + ' test -model models/ssd_face_pruned/face_test.prototxt -weights snapshots/'+fn+' -iterations 200'
        stdout, stderr = subprocess.Popen(command.split(), 
                                         stdout=subprocess.PIPE, 
                                         stderr=subprocess.PIPE, 
                                         stdin=subprocess.PIPE).communicate()
        res = [e for e in stderr.decode('utf-8').split('\n') if 'Test net output ' in e]
        if len(res)>0:
            res = [e[e.find(']')+1:] for e in res]
            val = float(res[0].split()[-1])
            print('\n'.join(res))
        else:
            val = 0
            print('No detections, mAP = 0')
    maps.append(val)
    cache[it] = val
    
cache = [str(k)+':'+str(v) for k,v in cache.items()]
cache = '\n'.join(cache) 
with open('snapshots/'+snapshot+'_cache.txt', 'w') as f:
    f.write(cache)
    
plt.plot(iters, maps, 'bx-')
plt.xlabel('Iteration')
plt.ylabel('mAP')
plt.title(snapshot)
plt.savefig('snapshots/'+snapshot+'_map.png')
plt.clf()