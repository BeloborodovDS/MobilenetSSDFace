import caffe

try:
    caffe.Net('models/ssd_face_pruned/face_train.prototxt', 
              'models/ssd_face_pruned/face_init.caffemodel', 
              caffe.TRAIN)
    caffe.Net('models/ssd_face_pruned/face_test.prototxt', 
              'models/ssd_face_pruned/face_init.caffemodel', 
              caffe.TEST)
    print('Model check COMPLETE')
except Exception as e:
    print(repr(e))
    print('Model check FAILED')