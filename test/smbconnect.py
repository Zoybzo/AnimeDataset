import os
import toml

from smb.SMBConnection import SMBConnection


# 配置文件路径
config_path = os.path.expanduser("./.private/smb.toml")
# 读取配置文件
with open(config_path, "r") as f:
    config = toml.load(f)["smb"]

# 提取配置信息
server_ip = config["server_ip"]
server_name = config.get("server_name", "")  # 可选字段
share_name = config["share_name"]
username = config["username"]
password = config["password"]

# 创建 SMB 连接
# conn = SMBConnection('用户名', '密码', '客户端名称', '服务器名称', use_ntlm_v2=True)
# conn.connect('172.16.17.42', 445)

# 创建 SMB 连接
conn = SMBConnection(
    username, password, "client_machine_name", server_name, use_ntlm_v2=True
)

try:
    # 连接到服务器
    if conn.connect(server_ip, 445):
        print("成功连接到 SMB 服务器！")

        # 列出共享文件夹中的文件
        files = conn.listPath(share_name, "/")
        print(f"共享文件夹 '{share_name}' 中的文件：")
        for file in files:
            print(file.filename)
    else:
        print("无法连接到 SMB 服务器。")
finally:
    # 关闭连接
    conn.close()
