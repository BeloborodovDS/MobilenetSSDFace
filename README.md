# Mobilenet+SSD face detector training

This repo contains code for Mobilenet+SSD face detector training. This detector is compatible with Movidius Neural Compute Stick. You need <a href="https://github.com/movidius/ncsdk" target="_blank">NCSDK</a> to test it with Neural Compute Stick.

To train this detector (<a href="https://github.com/weiliu89/caffe/tree/ssd" target="_blank">SSD-Caffe</a> is needed):

1. Download <a href="http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/" target="_blank">WIDER</a> and <a href="http://vis-www.cs.umass.edu/fddb/" target="_blank">FDDB</a> datasets.

2. Edit Makefile: set data_dir, lmdb_pyscript, caffe_exec, datasets names and path to data folder.

3. Make LMBD database:
~~~
make lmdb
~~~

4. Make face model (generate templates and get pre-trained weights):
~~~
make face_model_full
~~~

5. Edit train_files/solver_train_full.prototxt if necessary and train net:
~~~
make train_full
~~~

Or resume from snapshot:
~~~
echo /path/to/snapshot > train_files/snapshot.txt
make resume_full
~~~

6. Test model:
~~~
echo /path/to/snapshot > train_files/snapshot.txt
make test_full
~~~

Test best model from this repo:
~~~
make test_best_full
~~~

7. (Optional) Make long-range (shorter) model:
~~~
make face_model_short
~~~

And test it:
~~~
make test_short_init
~~~

8. Plot loss from Caffe logs:
~~~
make plot_loss
~~~

Plot Average Precision from snapshots:
~~~
echo /path/to/any/snapshot > train_files/snapshot.txt
make plot_map_full
~~~

9. Profile initial VOC net, best face net, short face net for Neural Compute Stick:
~~~
make profile_initial
~~~
or
~~~
make profile_face_full
~~~
or
~~~
make profile_short_init
~~~


Also see Caffe_face.ipynb for details.

See images/output to see how nets perform on examples (test network to get these results).

See <a href="https://colab.research.google.com/drive/1LExcFZO8vN46xrJ8deG159eIUaW0kB-H" target="_blank">this notebook</a> for training this model in Google Colaboratory.


