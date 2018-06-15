import sys
import PIL
from PIL import Image
import random
import xml.etree.ElementTree as ET
from xml.dom import minidom
random.seed(42)

def parse_fddb(text, path, min_face=6, min_ratio=0.02, all_valid=True, 
               min_im_ratio=0.5, max_im_ratio=2.0):
    """Parse FDDB annotations from text and images
    text: text annotations
    path: path to dataset
    min_face: discard faces smaller than this
    min_ratio: discard faces with ratio face_w/max(im_w, im_h) smaller than this
    all_valid: if True, include image only if all faces are accepted
    min_im_ratio, max_im_ratio: reject images with width/height outside this range
    Returns: [(filename, width, height, numchannels, [[x1,y1,x2,y2]...])...]"""
    total_faces = 0
    total_ims = 0
    data = text.split('\n')
    data = [e for e in data if len(e)>0]
    i = 0
    res = []
    while (i<len(data)):
        file = data[i]
        im = Image.open(path+file+'.jpg')
        total_ims += 1
        check = True
        if im.bits!=8:
            print('Warning: image '+file+' data is '+str(im.bits)+'-bit, skipping')
            check = False
        if im.layers not in {3,4}:
            print('Warning: image '+file+' has '+str(im.layers)+' channels, skipping')
            check = False
        width, height = im.size
        channels = im.layers
        i+=1
        num = int(data[i])
        i+=1
        faces = []
        total_faces += num
        im_ratio = 1.0*width/height
        if (im_ratio<min_im_ratio) or (im_ratio>max_im_ratio):
            check = False
        for j in range(num):
            face = data[i].split()
            face = [int(float(e)) for e in face if len(e)>0]
            rmax,rmin,ang,cx,cy,_ = face
            w = 2*rmin
            x = cx-rmin
            y = cy-rmin
            ratio = 1.0*w/max(width, height)
            if (w>=min_face) and (ratio>=min_ratio):
                faces.append([x,y,x+w,y+w])
            elif all_valid:
                check = False
            i+=1
        if (len(faces)>0) and check:
            res.append((file+'.jpg', width, height, channels, faces))
    print('Total images: '+str(total_ims))
    print('Total faces: '+str(total_faces))
    return res

def write_xml_fddb(data, path, xml_path, filename, write_sizes=True, shuffle=True, verbose=True):
    """Create .xml annotations and file-list for data
    data: dataset from parse_fddb(...)
    path: path to dataset
    xml_path: subpath to folder for .xml
    filename: name for file-list (no extension)
    write_sizes: if True, create file with (filename, height, width) records
    shuffle: if True, shuffle lines in data
    verbose: if True, report every 1000 files"""
    img_files = [e[0] for e in data]
    xml_files = ['.'.join(e.split('.')[:-1])+'.xml' for e in img_files]
    xml_files = [xml_path + e.replace('/','_') for e in xml_files]
    all_files = [u+' '+v for u,v in zip(img_files, xml_files)]
    
    if shuffle:
        random.shuffle(all_files)
    with open(path+filename+'.txt', 'w') as f:
        f.write('\n'.join(all_files))
        
    if write_sizes:
        sizes = [(e[0].split('/')[-1].split('.')[0], str(e[2]), str(e[1])) for e in data]
        sizes = [' '.join(e) for e in sizes]
        with open(path+filename+'_name_size.txt', 'w') as f:
            f.write('\n'.join(sizes))
     
    cnt = 0
    for item, sp in zip(data, xml_files):
        e_anno = ET.Element('annotation')
        e_size = ET.SubElement(e_anno, 'size')
        e_width = ET.SubElement(e_size, 'width')
        e_height = ET.SubElement(e_size, 'height')
        e_depth = ET.SubElement(e_size, 'depth')
        e_width.text = str(item[1])
        e_height.text = str(item[2])
        e_depth.text = str(item[3])
        for xmin,ymin,xmax,ymax in item[4]:
            e_obj = ET.SubElement(e_anno, 'object')
            e_name = ET.SubElement(e_obj, 'name')
            e_name.text = 'face'
            e_bndbox = ET.SubElement(e_obj, 'bndbox')
            e_xmin = ET.SubElement(e_bndbox, 'xmin')
            e_ymin = ET.SubElement(e_bndbox, 'ymin')
            e_xmax = ET.SubElement(e_bndbox, 'xmax')
            e_ymax = ET.SubElement(e_bndbox, 'ymax')
            e_xmin.text = str(xmin)
            e_ymin.text = str(ymin)
            e_xmax.text = str(xmax)
            e_ymax.text = str(ymax)
        txt = minidom.parseString(ET.tostring(e_anno, 'utf-8')).toprettyxml(indent='\t')
        with open(path+sp,'w') as f:
            f.write(txt)
        cnt += 1
        if (cnt%1000==0) and verbose:
            print(filename+': '+str(cnt)+' of '+str(len(data)))

# create labelmap for face detection            
def make_labelmap(path):
    txt = """item {
  name: "none_of_the_above"
  label: 0
  display_name: "background"
}
item {
  name: "face"
  label: 1
  display_name: "face"
}"""
    with open(path+'labelmap.prototxt', 'w') as f:
        f.write(txt)
        
if __name__ == "__main__":
    fddb_path = sys.argv[1] #dataset path
    train_xml_path = sys.argv[2] #xml subpath
    test_xml_path = sys.argv[3] #xml subpath
    
    train_folds = ['01','02','03','04','05','06','07','08','09']
    test_folds = ['10']
    
    make_labelmap(fddb_path)
    
    print('Parsing FDDB dataset...')
    
    #concat data from all folds and parse
    print('Train:')
    train_fddb = ''
    for fld in train_folds:
        with open(fddb_path+'/FDDB-folds/FDDB-fold-'+fld+'-ellipseList.txt') as f:
            data = f.read()
            train_fddb = train_fddb+data
    train_fddb = parse_fddb(train_fddb, fddb_path)
    print('Accepted images: '+str(len(train_fddb)))
    print('Accepted faces: '+str(sum([len(e[4]) for e in train_fddb])))
    
    print('Test:')
    test_fddb = ''
    for fld in test_folds:
        with open(fddb_path+'/FDDB-folds/FDDB-fold-'+fld+'-ellipseList.txt') as f:
            data = f.read()
            test_fddb = test_fddb+data
    test_fddb = parse_fddb(test_fddb, fddb_path)
    print('Accepted images: '+str(len(test_fddb)))
    print('Accepted faces: '+str(sum([len(e[4]) for e in test_fddb])))
    
    print('Creating .xml annotations and file lists...')
    
    print('Train:')
    write_xml_fddb(train_fddb, fddb_path, train_xml_path, 'trainval', write_sizes=False)
    print('Test:')
    write_xml_fddb(test_fddb,  fddb_path, test_xml_path,  'test',     shuffle=False)
    
    print('Done')