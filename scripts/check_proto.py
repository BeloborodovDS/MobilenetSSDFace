import caffe

try:
    caffe.Net('models/ssd_face/ssd_face_train.prototxt', 
              'models/empty.caffemodel', 
              caffe.TRAIN)
    caffe.Net('models/ssd_face/ssd_face_test.prototxt', 
              'models/empty.caffemodel', 
              caffe.TEST)
    caffe.Net('models/ssd_face/ssd_face_deploy.prototxt', 
              'models/empty.caffemodel', 
              caffe.TEST)
    caffe.Net('models/ssd_face/ssd_face_deploy_bn.prototxt', 
              'models/empty.caffemodel', 
              caffe.TEST)
    print('Model check COMPLETE')
except Exception as e:
    print(repr(e))
    print('Model check FAILED')
    