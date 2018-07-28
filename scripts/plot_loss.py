import os
from matplotlib import pyplot as plt
import numpy as np

logs = os.listdir('snapshots')
logs = [e for e in logs if e.endswith('_log.txt')]

for fn in logs:
    with open('snapshots/'+fn) as f:
        log = f.read().split('\n')
    log = [e.split() for e in log if 'Iteration' in e and 'loss' in e]
    loss = [float(e[-1]) for e in log]
    iters = [int(e[-4][:-1]) for e in log]
    
    plt.plot(iters, loss, 'b-')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title(fn[:-8])
    plt.savefig('snapshots/'+fn[:-8]+'_loss.png')
    plt.clf()
    
    plt.plot(iters, np.log(np.array(loss)), 'b-')
    plt.xlabel('Iteration')
    plt.ylabel('Log loss')
    plt.title(fn[:-8])
    plt.savefig('snapshots/'+fn[:-8]+'_logloss.png')
    plt.clf()