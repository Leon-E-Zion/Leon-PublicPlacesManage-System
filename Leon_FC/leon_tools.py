import os
import sys

def leon_mk(path):
	try:
		if path.find('.') == -1:
			os.makedirs(path)
		else:
			with open(path,mode="w",encoding="utf-8") as f:
	   			pass
	except:
		pass

# 基于现有项目路径加载深层路径
def leon_path_in(path):
	root = os.getcwd()
	sys.path.append(os.path.join(root,path))
	return os.path.join(root,path)
