import os

# init_test
def init_test():
	return(r"Welcome To Leon's WorkStation.")

# info_mes(key,mes)
def info_mes(key,mes):
	print('--------\n{}:{}\n--------'.format(key,mes))

# get dict_dirs
def fc_0(str_,prefix):
    k_0 = str_.find("'")
    k_1 = str_.find("'",(k_0+1))
    return 'value_type:'+str_[k_0+1:k_1]
def get_dict_dirs(text, prefix='\t'):
    for i in text.keys():
        if isinstance(text[i], dict):
            print(prefix + '+ ' + i)
            get_dict_dirs(text[i], prefix + '|   ')  # 最后一个└
        else:
            print(prefix + '- ' + i+' '+fc_0(str(type(text[i])),prefix))


if __name__=='__main__':
    # 初始化测试集
    a = {"a":{"a":1,"b":2,"c":3},"b":2,"c":3}
    get_dict_dirs(a)

 

 

