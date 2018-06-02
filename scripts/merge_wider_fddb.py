import sys
import random
random.seed(42)

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

#add path to each item of 'str1 str2\nstr3 str4\n...' sequence
def add_path(text, path):
    data = text.split('\n')
    data = [e.split() for e in data if len(e)>0]
    data = [[path+v for v in u] for u in data]
    data = [' '.join(e) for e in data]
    return '\n'.join(data)

#shuffle lines of text
def shuffle_lines(text):
    res = text.split('\n')
    random.shuffle(res)
    return '\n'.join(res)
   
if __name__ == "__main__":
    data_path = sys.argv[1]
    wider_path = sys.argv[2]
    fddb_path = sys.argv[3]
    new_path = sys.argv[4]
    
    make_labelmap(data_path+new_path)
    
    #merge three files from each of two datasets
    for fn in ["trainval.txt", "test.txt", "test_name_size.txt"]:
        with open(data_path + wider_path + fn) as f:
            wider_text = f.read()
        with open(data_path + fddb_path  + fn) as f:
            fddb_text = f.read()
        #file with sizes does not need paths
        if fn != "test_name_size.txt":
            wider_text = add_path(wider_text, wider_path)
            fddb_text = add_path(fddb_text, fddb_path)
        merged_text = wider_text + '\n' + fddb_text
        #shuffle only train data
        if fn == "trainval.txt":
            merged_text = shuffle_lines(merged_text)
        with open(data_path + new_path   + fn, 'w') as f:
            f.write(merged_text)
