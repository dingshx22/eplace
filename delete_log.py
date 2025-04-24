import os

folder_path = 'logs'# 设置 images 文件夹路径

# 图片文件扩展名列表
info_extensions = ['.csv', '.json']

# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    if os.path.isfile(file_path) and os.path.splitext(filename)[1].lower() in info_extensions:
        os.remove(file_path)
        print(f"已删除: {file_path}")

print("记录清理完成。")
