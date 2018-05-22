# coding: utf-8

import cPickle as pik
import sys
import cv2
import numpy as np
import tensorflow as tf
import decoder
# import decoderrn
import re

TAG_SIZE = 20
IMG_SIZE = 256
IMG_SIZE_HEIGHT = 456
BATCH_SIZE = 64
MAX_CAP = 15

def load_img(imgf):
    im = cv2.imread(imgf)
    im = cv2.resize(im, (IMG_SIZE, IMG_SIZE_HEIGHT))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = im.astype(np.float32)
    im = im/256*2-1

    img = im.reshape(1, IMG_SIZE_HEIGHT, IMG_SIZE, 1)
    return img
    
def decode(dic_tag, arr):
    tags = []
    for t in arr:
        i = np.argmax(t)
        tags.append(dic_tag[i])
    return tags

def main():
    
    if len(sys.argv) < 2:
        print "cmd image_path"
        return
    else:
        imgf = sys.argv[1]
    
    img = load_img(imgf)
    with open("dict.pik") as f:
        dic = pik.load(f)
    size = len(dic)

    config = tf.ConfigProto(device_count = {'GPU': 0})
    sess = tf.Session(config=config)

    with open("models/checkpoint") as f:
        fn = f.readline()
        fn = re.match(r'model_checkpoint_path\: \"(.*?)\"', fn, re.M|re.I).group(1) 
        
    saver = tf.train.import_meta_graph("models/%s.meta"%fn)
    saver.restore(sess,tf.train.latest_checkpoint('models'))
    # sess.run(tf.global_variables_initializer())

    model = tf.get_default_graph()

    start_tag = np.zeros(size, dtype=np.float32)
    start_tag[0] = 1
    start_tag = start_tag.reshape(1, -1)
    out = sess.run([
        model.get_tensor_by_name("tags_out:0"),
    ], feed_dict={
        model.get_tensor_by_name("img:0"): img,
        model.get_tensor_by_name("tag:0"): start_tag,
    })

    tags = decode(dic, out[0])
    print tags
    dec = decoder.Decoder()
    page = dec.decode(tags)
    # print page
    rpage = page.react()
    print rpage

    # with open("ReactApp/show-app/src/App.js", "w") as f:
    #     f.write(rpage)
    # print "write react js file finish"

        # print tags
    # dec = decoderrn.Decoder()
    # page = dec.decode(tags)
    # print page
    # rpage = page.react()

    # with open("pix2app/src/App.js", "w") as f:
    #     f.write(rpage)
    # #print rpage
    # print "write app file finish"

    sess.close()  
    
if __name__ == "__main__":
    main()
