cur_dir := "$(shell pwd)"
wider_name := WIDERfacedet
fddb_name := FDDBfacedet
combine_name := WIDER_FDDB_facedet
data_dir := /home/dmitrii/Work/Datasets
wider_dir := $(data_dir)/$(wider_name)
fddb_dir := $(data_dir)/$(fddb_name)

lmdb_pyscript := /opt/movidius/ssd-caffe/scripts/create_annoset.py
caffe_exec := /opt/movidius/ssd-caffe/build/tools/caffe

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

wider_xml:
	cd $(wider_dir) && \
	cd WIDER_train && \
	mkdir -p -v xml && \
	cd ../WIDER_val && \
	mkdir -p -v xml && \
	cd $(cur_dir) && \
	python3 ./scripts/make_wider_xml.py $(wider_dir)/ WIDER_train/xml/ WIDER_val/xml/
wider_lmdb:
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
fddb_lmdb:
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
	$(caffe_exec) train -solver solver_train.prototxt -weights models/ssd_face_pruned/face_init.caffemodel






