# coding:utf-8 
# @Time : 2019-04-03 09:43
# @Author : SuRui

import os
import sys
import time

#sys.path.append("/your_build_caffe/python")
import caffe
import numpy as np
import shutil as sh

os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

def initial_graph(proto, model):
	caffe.set_mode_cpu()
	#caffe.set_mode_gpu()
	net = caffe.Net(proto, model, caffe.TEST)
	return net


def eval(net, image):
	nh, nw = 161, 198 # keep consistent with deploy.prototxt
	im = caffe.io.load_image(image)
	# for gray
	# im = im[:, :, 0]
	# im = np.expand_dims(im, 2)
	h, w, _ = im.shape
	print(h, w, _)
	transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
	transformer.set_transpose('data', (2, 0, 1))  # row to col
	# transformer.set_channel_swap('data', (2, 1, 0))  # RGB to BGR
	transformer.set_raw_scale('data', 255)  # [0,1] to [0,255]
	# transformer.set_mean('data', img_mean)  #
	transformer.set_input_scale('data', 0.0078125)
	start = time.time()
	net.blobs['data'].reshape(1, 1, nh, nw)
	net.blobs['data'].data[...] = transformer.preprocess('data', im)
	out = net.forward()
	end = time.time()
	print("Cost time : %s" % (end - start))
	
	prob = out['prob']
	prob = np.squeeze(prob)
	idx = np.argsort(-prob)
	print("\n-+-+-+")
	print("class : %s" % str(idx[0]))
	print("prob : %s" % str(prob[idx[0]]))
	print("-+-+-+")
	print("-+-+-+")
	return idx[0] == 1


def calc_inference_performance(all_data_dir):
	# binary classification
	TP = 0
	TN = 0
	FP = 0
	FN = 0
	name_list = os.listdir(all_data_dir)
	total = len(name_list)
	net = initial_graph(deploy_prototxt_path, deploy_caffemodel_path)
	for file_name in name_list:
		ground_truth = file_name.split("_")[0]
		flag = eval(net, os.path.join(all_data_dir, file_name))
		if flag and ground_truth == '1':
			TP += 1
			sh.copy(os.path.join(all_data_dir, file_name),
			        os.path.join("./model_analysis/TP", file_name))
		elif not flag and ground_truth == '0':
			TN += 1
			sh.copy(os.path.join(all_data_dir, file_name),
			        os.path.join("./model_analysis/TN", file_name))
		elif flag and ground_truth == '0':
			FP += 1
			sh.copy(os.path.join(all_data_dir, file_name),
			        os.path.join("./model_analysis/FP", file_name))
		elif not flag and ground_truth == '1':
			FN += 1
			sh.copy(os.path.join(all_data_dir, file_name),
			        os.path.join("./model_analysis/FN", file_name))
		print("TP: %s|TN: %s|FP: %s|FN: %s" % (TP, TN, FP, FN))
		print("precision: %s" % str((TP) / (TP + FP + 1e-5)))
		print("recall: %s" % str((TP) / (TP + FN + 1e-5)))
		print("total: %s" % total)
		print("success: %s" % str(TP + TN))
		print("acc %s" % str(((TP + TN) * 1.0 / total)))
	
	print("TP: %s" % TP)
	print("TN: %s" % TN)
	print("precision: %s" % str((TP) * 1.0 / (TP + FP)))
	print("recall: %s" % str((TP) * 1.0 / (TP + FN)))
	print("total: %s" % total)
	print("success: %s" % str(TP + TN))
	print("acc %s" % str(((TP + TN) * 1.0 / total)))



if __name__ == "__main__":
	deploy_prototxt_path = "xxx/mobilenet_v2_deploy.prototxt"
	deploy_caffemodel_path = "xxx/mo2_iter_20000.caffemodel"
	net = initial_graph(deploy_prototxt_path, deploy_caffemodel_path)
	data_dir = "" 
	calc_inference_performance(data_dir)

