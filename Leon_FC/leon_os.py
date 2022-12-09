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
    # 判断操作系统
    if sys.platform == 'win32':
        path = path.split(r'\\')
    else:
        path = path.split(r'/')
    path_ = ''
    for pt in path:
        path_ = os.path.join(path_,pt)
    path_ = os.path.join(root,path_)
    sys.path.append(path_)
    return path_

# 生成指定文件夹的目录树 并写入保存到指定文件中
def dfs_showdir(f,path, depth):
    if depth == 0:
        print("root:[" + path + "]")
        f.write("root:[" + path + "]"+'\n')
    for item in os.listdir(path):
        if item in ['.git', '.idea', '__pycache__']:
            continue
        print("| " * depth + "+--" + item)
        f.write("| " * depth + "+--" + item+'\n')
        new_item = path + '/' + item
        if os.path.isdir(new_item):
            dfs_showdir(f,new_item, depth + 1)
def get_filetree(path):
    with open(os.path.join(path,'Leon-file_tree.txt'),"w+") as f:    #设置文件对象
        str_ = dfs_showdir(f,path, 0)
        

if __name__ == '__main__':
    pass

