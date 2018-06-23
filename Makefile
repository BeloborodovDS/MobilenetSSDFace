cur_dir := "$(shell pwd)"
wider_name := WIDERfacedet
fddb_name := FDDBfacedet
combine_name := WIDER_FDDB_facedet
data_dir := /home/dmitrii/Work/Datasets
wider_dir := $(data_dir)/$(wider_name)
fddb_dir := $(data_dir)/$(fddb_name)

lmdb_pyscript := /opt/movidius/ssd-caffe/scripts/create_annoset.py
caffe_exec := /opt/movidius/ssd-caffe/build/tools/caffe

wider_train_id := 0B6eKvaijfFUDQUUwd21EckhUbWs
wider_val_id := 0B6eKvaijfFUDd3dIRmpvSk8tLUk

profile_initial:
	cd models/ssd_voc_profile; \
	mvNCProfile ../ssd_voc/deploy.prototxt -w ../ssd_voc/MobileNetSSD_deploy.caffemodel -s 12; \
	cd ../..
compile_initial:
	cd models/ssd_voc_profile; \
	mvNCCompile ../ssd_voc/deploy.prototxt -w ../ssd_voc/MobileNetSSD_deploy.caffemodel -s 12; \
	cd ../..
profile_face:
	cd models/ssd_face_pruned; \
	mvNCProfile ./face_deploy.prototxt -w ../empty.caffemodel -s 12; \
	cd ../..

wider_load:
	mkdir -p -v $(wider_dir) && cd $(wider_dir) && \
	wget --quiet --save-cookies cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$(wider_train_id)" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p' > confirm.txt && \
	wget --load-cookies cookies.txt "https://docs.google.com/uc?export=download&confirm=`cat confirm.txt`&id=$(wider_train_id)" -O WIDER_train.zip && \
	wget --quiet --save-cookies cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$(wider_val_id)" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p' > confirm.txt && \
	wget --load-cookies cookies.txt "https://docs.google.com/uc?export=download&confirm=`cat confirm.txt`&id=$(wider_val_id)" -O WIDER_val.zip && \
	wget http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/support/bbx_annotation/wider_face_split.zip -O wider_face_split.zip && \
	rm -rf cookies.txt && rm -rf confirm.txt
fddb_load:
	mkdir -p -v $(fddb_dir) && cd $(fddb_dir) && \
	wget http://tamaraberg.com/faceDataset/originalPics.tar.gz -O originalPics.tar.gz && \
	wget http://vis-www.cs.umass.edu/fddb/FDDB-folds.tgz -O FDDB-folds.tgz
datasets:
	cd $(wider_dir) && unzip -u WIDER_train && unzip -u WIDER_val && unzip -u wider_face_split && \
	cd $(fddb_dir) && tar -xvzf originalPics.tar.gz && tar -xvzf FDDB-folds.tgz	

wider_xml:
	cd $(wider_dir) && \
	cd WIDER_train && \
	mkdir -p -v xml && \
	cd ../WIDER_val && \
	mkdir -p -v xml && \
	cd $(cur_dir) && \
	python3 ./scripts/make_wider_xml.py $(wider_dir)/ WIDER_train/xml/ WIDER_val/xml/
wider_lmdb: wider_xml
	python3 $(lmdb_pyscript) --anno-type=detection --label-map-file=$(wider_dir)/labelmap.prototxt \
	--min-dim=0 --max-dim=0 --resize-width=0 --resize-height=0 --check-label --encode-type=jpg --encoded \
	--redo \
	$(wider_dir) $(wider_dir)/trainval.txt $(wider_dir)/WIDER_train/lmdb/wider_train_lmdb \
	./data; \
	python3 $(lmdb_pyscript) --anno-type=detection --label-map-file=$(wider_dir)/labelmap.prototxt \
	--min-dim=0 --max-dim=0 --resize-width=0 --resize-height=0 --check-label --encode-type=jpg --encoded \
	--redo \
	$(wider_dir) $(wider_dir)/test.txt $(wider_dir)/WIDER_val/lmdb/wider_test_lmdb \
	./data \

fddb_xml:
	cd $(fddb_dir) && \
	mkdir -p -v xml && \
	cd xml && \
	mkdir -p -v trainval && \
	mkdir -p -v test && \
	cd $(cur_dir) && \
	python3 ./scripts/make_fddb_xml.py $(fddb_dir)/ xml/trainval/ xml/test/
fddb_lmdb: fddb_xml
	python3 $(lmdb_pyscript) --anno-type=detection --label-map-file=$(fddb_dir)/labelmap.prototxt \
	--min-dim=0 --max-dim=0 --resize-width=0 --resize-height=0 --check-label --encode-type=jpg --encoded \
	--redo \
	$(fddb_dir) $(fddb_dir)/trainval.txt $(fddb_dir)/lmdb/fddb_train_lmdb \
	./data; \
	python3 $(lmdb_pyscript) --anno-type=detection --label-map-file=$(fddb_dir)/labelmap.prototxt \
	--min-dim=0 --max-dim=0 --resize-width=0 --resize-height=0 --check-label --encode-type=jpg --encoded \
	--redo \
	$(fddb_dir) $(fddb_dir)/test.txt $(fddb_dir)/lmdb/fddb_test_lmdb \
	./data \

merge_datasets: 
	cd $(data_dir) && \
	mkdir -p -v $(combine_name) && \
	cd $(cur_dir) && \
	python3 ./scripts/merge_wider_fddb.py $(data_dir)/ $(wider_name)/ $(fddb_name)/ $(combine_name)/

lmdb: wider_xml fddb_xml merge_datasets
	python3 $(lmdb_pyscript) --anno-type=detection --label-map-file=$(data_dir)/$(combine_name)/labelmap.prototxt \
	--min-dim=0 --max-dim=0 --resize-width=0 --resize-height=0 --check-label --encode-type=jpg --encoded \
	--redo \
	$(data_dir) $(data_dir)/$(combine_name)/trainval.txt \
	$(data_dir)/$(combine_name)/lmdb/wider_fddb_train_lmdb \
	./data; \
	python3 $(lmdb_pyscript) --anno-type=detection --label-map-file=$(data_dir)/$(combine_name)/labelmap.prototxt \
	--min-dim=0 --max-dim=0 --resize-width=0 --resize-height=0 --check-label --encode-type=jpg --encoded \
	--redo \
	$(data_dir) $(data_dir)/$(combine_name)/test.txt \
	$(data_dir)/$(combine_name)/lmdb/wider_fddb_test_lmdb \
	./data \

gen_templates:
	python3 ./models/ssd_voc/gen.py --stage=train --lmdb=$(cur_dir)/data/wider_fddb_train_lmdb/ \
	--label-map=$(cur_dir)/models/labelmap.prototxt --class-num=2 \
	> models/ssd_face/ssd_face_train.prototxt; \
	python3 ./models/ssd_voc/gen.py --stage=test --lmdb=$(cur_dir)/data/wider_fddb_test_lmdb/ \
	--label-map=$(cur_dir)/models/labelmap.prototxt --class-num=2 \
	> models/ssd_face/ssd_face_test.prototxt; \
	python3 ./models/ssd_voc/gen.py --stage=deploy --class-num=2 \
	> models/ssd_face/ssd_face_deploy.prototxt; \
	python3 ./scripts/check_proto.py 

face_model: gen_templates
	python3 ./scripts/make_face_model.py 50; \
	python3 ./scripts/check_face_model.py

train:
	$(caffe_exec) train -solver train_files/solver_train.prototxt -weights models/ssd_face_pruned/face_init.caffemodel 2>&1 | \
	tee `cat train_files/solver_train.prototxt | grep snapshot_prefix | grep -o \".* | tr -d \"`_log.txt
train_noinit:
	$(caffe_exec) train -solver train_files/solver_train.prototxt 2>&1 | \
	tee `cat train_files/solver_train.prototxt | grep snapshot_prefix | grep -o \".* | tr -d \"`_log.txt
resume:
	$(caffe_exec) train -solver train_files/solver_train.prototxt -snapshot `cat train_files/snapshot.txt` 2>&1 | \
	tee -a `cat train_files/solver_train.prototxt | grep snapshot_prefix | grep -o \".* | tr -d \"`_log.txt
test:
	mkdir -p models/tmp && mkdir -p images/output && \
	python3 models/ssd_voc/merge_bn.py models/ssd_face_pruned/face_train.prototxt `cat train_files/weights.txt` \
	models/ssd_face_pruned/face_deploy.prototxt models/tmp/test.caffemodel && \
	python3 scripts/test_on_examples.py `cat train_files/weights.txt` && \
	echo "\n\nmAP:" && \
	$(caffe_exec) train -solver train_files/solver_test.prototxt -weights `cat train_files/weights.txt` 2>&1 | \
	grep -o "Test net output .* = [0-9]*\.[0-9]*"
plot_loss:
	python3 scripts/plot_loss.py
plot_map:
	python3 scripts/plot_map.py






