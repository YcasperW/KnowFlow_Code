import os
import importlib.util

# 修复：原为硬编码绝对路径，仓库换个位置就跑不起来。改为相对本文件定位。
_PAGE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Web Frame.py")
spec = importlib.util.spec_from_file_location("wf", _PAGE_PATH)
wf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wf)
app = wf.app
client = app.test_client()

# 1. 游客态
r = client.get('/auth_status')
print('1) guest auth_status:', r.get_json())
assert r.get_json()['is_admin'] is False

# 2. 游客上传 → 必须 403
r = client.post('/upload')
print('2) guest upload ->', r.status_code, r.get_json())
assert r.status_code == 403

# 3. 错误口令登录 → 401
r = client.post('/admin_login', json={'password': 'wrong'})
print('3) wrong login ->', r.status_code)
assert r.status_code == 401

# 4. 正确口令登录 → 200 且 is_admin=True
r = client.post('/admin_login', json={'password': 'admin123'})
print('4) admin login ->', r.status_code, r.get_json())
assert r.status_code == 200 and r.get_json()['is_admin'] is True

# 5. 同一会话再查状态 → True（会话保持）
r = client.get('/auth_status')
print('5) admin auth_status:', r.get_json())
assert r.get_json()['is_admin'] is True

# 6. 登录后上传（无文件）→ 越过 403，返回 400（缺文件，证明管理员闸门已放行）
r = client.post('/upload')
print('6) admin upload(no file) ->', r.status_code, r.get_json())
assert r.status_code == 400

# 7. 登出 → 状态变 False
r = client.post('/admin_logout')
print('7) logout:', r.get_json())
r = client.get('/auth_status')
assert r.get_json()['is_admin'] is False

print('\n✅ ALL AUTH SMOKE TESTS PASSED')
