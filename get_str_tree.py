import os

# 在这里配置你想跳过的垃圾文件夹，避免输出结果太长
IGNORE_DIRS = {'.git', '__pycache__', 'venv', 'env', 'node_modules', '.idea', '.vscode', '.pytest_cache'}
# 在这里配置你想跳过的隐藏文件（比如 Mac 的 .DS_Store）
IGNORE_FILES = {'.DS_Store'}


def print_tree(dir_path, prefix=""):
    """递归打印目录树"""
    try:
        # 获取目录下所有文件和文件夹
        items = os.listdir(dir_path)
    except PermissionError:
        print(prefix + "├── [无权限访问]")
        return

    # 过滤掉不需要的文件夹和文件
    valid_items = []
    for item in items:
        path = os.path.join(dir_path, item)
        if os.path.isdir(path) and item in IGNORE_DIRS:
            continue
        if os.path.isfile(path) and item in IGNORE_FILES:
            continue
        valid_items.append(item)

    # 排序：让文件夹排在前面，文件排在后面，看起来更整齐
    valid_items.sort(key=lambda x: (not os.path.isdir(os.path.join(dir_path, x)), x.lower()))

    # 遍历并打印
    for i, item in enumerate(valid_items):
        path = os.path.join(dir_path, item)
        is_last = (i == len(valid_items) - 1)

        # 选择连接符
        connector = "└── " if is_last else "├── "
        print(prefix + connector + item)

        # 如果是文件夹，继续往下一层递归
        if os.path.isdir(path):
            extension = "    " if is_last else "│   "
            print_tree(path, prefix + extension)


if __name__ == "__main__":
    # 默认扫描当前所在目录
    target_dir = "."
    absolute_path = os.path.abspath(target_dir)
    project_name = os.path.basename(absolute_path)

    print(f"📁 {project_name}/")
    print_tree(target_dir)