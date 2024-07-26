import os
import re

# 原txt文件目录（不要以\结尾）
input_dir = r'./labels/train/'
# 改标签后txt文件目录（不要以\结尾）
out_dir = r"./labels/train1/"

if not os.path.exists(out_dir):
    os.mkdir(out_dir)


def file_name(input_dir):
    for dir_path, _, files in os.walk(input_dir):
        for i_name in files:
            i_path = os.path.join(dir_path, i_name)
            o_path = os.path.join(out_dir, i_name)
            with open(i_path, 'r+', encoding='utf-8') as f:
                for i in f:
                    print(i)
                    # 需要换几个，写几个if(1变0)
                    if re.match('1', i):
                        with open(o_path, 'a+') as g:
                            data = '0 ' + i[2:]
                            g.write(data)
                    # elif re.match('16', i):
                    #     with open(o_path, 'a+') as g:
                    #         data = '1' + i[2:]
                    #         g.write(data)
                    else:
                        with open(o_path, 'a+') as g:
                            g.write(i)


if __name__ == '__main__':
    file_name(input_dir)
