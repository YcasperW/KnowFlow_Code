# -*- coding: utf-8 -*-
# gen_password_hash.py
# 用途：把管理员明文口令转成哈希串，用于替换 .env 里的 KNOWFLOW_ADMIN_PASSWORD 值。
# 用法：
#   方式一（明文传参，会显示在命令行历史，慎用）：python gen_password_hash.py 你的口令
#   方式二（不回显，推荐）：python gen_password_hash.py   → 按提示输入
import hashlib, secrets, base64, sys, getpass

def hash_password(plain, iterations=200000):
    """把明文口令变成可安全存储的哈希串（含随机盐，不可逆）。"""
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", plain.encode("utf-8"), salt, iterations)
    return f"pbkdf2${iterations}${base64.b64encode(salt).decode()}${base64.b64encode(dk).decode()}"

if __name__ == "__main__":
    if len(sys.argv) > 1:
        plain = sys.argv[1]
    else:
        plain = getpass.getpass("请输入管理员明文口令（不回显）：")
    h = hash_password(plain)
    print("\n✅ 哈希串已生成，请把 .env 里改为：")
    print(f'KNOWFLOW_ADMIN_PASSWORD={h}')
    print("\n（多个管理员用逗号分隔多个哈希串即可）")
