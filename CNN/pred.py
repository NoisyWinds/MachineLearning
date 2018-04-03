import re
import numpy as np
from PIL import Image
import io
import base64
import tensorflow as tf

class Pred(object):
	def __init__(self):
			self.sess = tf.Session()

			# 载入模型并且填充参数
			saver = tf.train.import_meta_graph('tmp/model.ckpt.meta')
			saver.restore(self.sess, 'tmp/model.ckpt')
			graph = tf.get_default_graph()
			self.x = graph.get_operation_by_name('x').outputs[0]
			self.keep_prob = graph.get_operation_by_name('keep_prob').outputs[0]

			# tf.get_colliection 会返回 tf.add_to_collection 所设置的预测方法
			self.pred = tf.get_collection('pred')[0]

	def response(self,dataURL):
				arr_x = self.changeImage(dataURL)
				result = self.sess.run(self.pred,feed_dict={self.x:arr_x,self.keep_prob: 1.0})[0]
				return str(result)

	def changeImage(self,dataUrl):
		dataUrl = re.sub('^data:image/.+;base64,','',dataUrl)
		image_s = base64.b64decode(dataUrl)
		image=io.BytesIO(image_s)
		img = Image.open(image).convert('L')
		if img.size[0] != 28 or img.size[1] != 28:
			img = img.resize((28, 28))
		arr = []
		for i in range(28):
			for j in range(28):
				pixel = 1.0 - float(img.getpixel((j, i)))/255.0
				arr.append(pixel)
		arr_x = np.array(arr).reshape((1,784))
		return arr_x