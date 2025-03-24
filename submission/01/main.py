import os
import shutil
import json
from datetime import datetime

def makedir():
    """
    函数实现对于文件夹以及日志文件的创建
    """
    os.mkdir('copy')
    os.mkdir('copy/quantum_core/')
    os.mkdir('copy/quantum_core/SECTOR-7G/')

    os.mkdir('copy/hologram_vault/')
    os.mkdir('copy/hologram_vault/CHAMBER-12F/')

    os.mkdir('copy/exobiology_lab/')
    os.mkdir('copy/exobiology_lab/POD-09X/')

    os.mkdir('copy/temporal_archive/')
    os.mkdir('copy/temporal_archive/VAULT-00T/')

    os.mkdir('copy/quantum_quarantine/')

    open('copy/hologram_log.txt','w').close()
    open('copy/hologram_log.json', 'w').close()

def rename(name, prefix):
    """
    :param name: 需要增加前缀的文件名
    :param prefix: 需要增加的前缀名
    :return: 更名之后文件所在的位置
    """
    folder = "incoming_data"
    new_name = prefix + name
    os.rename(os.path.join(folder, name), os.path.join(folder, new_name))
    return os.path.join(folder, new_name)

def classify():
    """
    函数实现对incoming_data文件夹内的文件按照后缀名进行分类
    分类结束之后的文件保存在copy文件夹正确的目录下
    最后删除原incoming_data文件夹，将copy文件夹更名为incoming_data
    从而实现目标
    """
    folder = "incoming_data"
    # 通过os.walk()进行对incoming_data的遍历
    for _, _, files in os.walk(folder):
        for file in files:
            # 获得incoming_data 文件夹里每一个文件的路径(file_path)、文件名(file_name)、后缀名(file_extension)
            file_path = os.path.join(folder, file)
            _, file_extension = os.path.splitext(file_path)

            # 获得当前时间信息（时 + 秒 + 微秒），以便后续的增加时间前缀
            dt = datetime.now()
            time_str = dt.strftime('%H%M%S%f') + '_'

            # 对文件后缀名进行分类，调用rename（）函数进行增加前缀操作，接着将其移动到copy文件夹正确的目录下
            if file_extension == '.quantum':
                new = rename(file, time_str)
                shutil.move(new, 'copy/quantum_core/SECTOR-7G/')
            elif file_extension == '.holo':
                new = rename(file, time_str)
                shutil.move(new, 'copy/hologram_vault/CHAMBER-12F/')
            elif file_extension == '.exo':
                new = rename(file, time_str)
                shutil.move(new, 'copy/exobiology_lab/POD-09X/')
            elif file_extension == '.chrono':
                new = rename(file, time_str)
                shutil.move(new, 'copy/temporal_archive/VAULT-00T/')
            else:
                # 未知文件需再增加一截前缀
                new = rename(file, time_str + 'ENCRYPTED_')
                shutil.move(new, 'copy/quantum_quarantine/')

    # 删除原 incoming_data 文件夹(此时应为空), 将已经分类好的文件夹 copy 更名为 incoming_data
    os.removedirs('incoming_data')
    os.rename('copy', 'incoming_data')


def write(start_path, file, prefix="", data=None):
    """
    递归遍历目录结构，并写入日志文件，同时生成 JSON 结构化数据。
    :param start_path: 要遍历的目录路径。
    :param file: 目标日志文件对象。
    :param prefix: 用于格式化输出的前缀（默认值为空）。
    :param data: 存储目录结构的 JSON 数据（默认为 None，函数内部初始化）。
    :return: 包含目录结构的 JSON 数据。
    """
    # 如果 data 为空，则初始化为字典
    if data is None:
        data = {}

    # 获取 start_path 目录下的所有文件和文件夹
    items = os.listdir(start_path)

    # 遍历当前目录中的所有文件和文件夹
    for item in items:
        path = os.path.join(start_path, item)  # 构造完整的路径

        # 在日志文件中写入当前项的前缀
        file.write(prefix + "├─ ")

        # 处理特殊文件 hologram_log.txt，直接写入名称
        if item == "hologram_log.txt":
            file.write(item + '\n')

        # 处理文件夹
        elif os.path.isdir(path):
            file.write('🚀')  # 目录前加上火箭符号表示
            data[item] = {}  # 在 JSON 结构中创建该目录的字典
            file.write(item + '\n')
            new_prefix = prefix + "│   "  # 生成新的前缀，用于子项缩进
            write(path, file, new_prefix, data[item])  # 递归处理子目录

        # 处理文件
        elif os.path.isfile(path):
            # 判断文件类型，并在日志中加上对应的图标
            if item.endswith(".quantum") or item.endswith(".holo") or item.endswith(".exo") or item.endswith(".chrono"):
                file.write('🔮')
            else:
                file.write('⚠️')

            file.write(item + '\n')

            if "files" not in data:  # 如果 data 字典中没有 "files" 这个键
                data["files"] = []  # 初始化 "files" 为一个空列表
            data.setdefault("files", []).append(item)

    return data  # 返回 JSON 结构化数据


def save():
    """
    函数实现 hologram_log.txt 和 hologram_log.json 两个日志文件的写入数据
    """

    # 写入 hologram_log.txt
    with open("incoming_data/hologram_log.txt", "w", encoding="utf-8") as f:
        f.write("┌─────────────────────────────┐\n")
        f.write("│ 🛸 Xia-III 空间站数据分布全息图 │\n")
        f.write("└─────────────────────────────┘\n\n")
        f.write("├─🚀 incoming_data" + '\n')

        # 此处调用 write（）函数，获得返回值 data 储存了文件夹中数据的目录信息
        data = {"incoming_data" : write("incoming_data", f, "│   ")}

        f.write('\n\n')
        f.write('🤖 SuperNova · 地球标准时 2142-10-25T12:03:47\n')
        f.write('⚠️ 警告：请勿直视量子文件核心')

    # 写入 hologram_log.json
    with open("incoming_data/hologram_log.json", 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    print("🌌 量子文件舱管理系统开始工作🚀🚀🚀")
    print("------------------------")
    makedir()
    print("已创建好目标舱室🎖️🎖️🎖️")
    print("------------------------")
    classify()
    print("已将外星数据文件归档分类🎇🎇🎇")
    print("------------------------")
    save()
    print("已保存3D全息日志到‘hologram_log.txt 和 hologram_log.json’，请查看📜📜📜")
    print("------------------------")
    print("已完成量子数据流处理🚀🚀🚀")



"""
.quantum → 存入 quantum_core/SECTOR-7G/
.holo → 存入 hologram_vault/CHAMBER-12F/
.exo → 存入 exobiology_lab/POD-09X/
.chrono → 存入 temporal_archive/VAULT-00T/
未知文件 → 存入 quantum_quarantine/ 并重命名（前缀加 ENCRYPTED_）
"""