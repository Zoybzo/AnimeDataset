#!/bin/bash

# 遍历当前目录下的所有.safetensors文件
for file in *.safetensors; do
    # 检查文件名是否不包含形如 -000000 到 -999999 的部分
    if [[ ! "$file" =~ -[0-9]{6}\.safetensors$ ]]; then
        # 提取文件名的前缀部分（去除.safetensors扩展名）
        prefix="${file%.safetensors}"
        # 构造新的文件名
        new_name="${prefix}-000050.safetensors"
        # 重命名文件
        mv "$file" "$new_name"
        echo "Renamed $file to $new_name"
    fi
done

echo "All applicable files have been renamed."

