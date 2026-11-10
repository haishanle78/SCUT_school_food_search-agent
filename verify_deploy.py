"""核对「线上运行的代码」与「本地 SDK 源码」是否一致。

用法：
    python verify_deploy.py

原理：web/server.py 启动时会把固定清单里的源码文件做 SHA256 指纹，
通过 /api/status 的 build 字段对外暴露。本脚本在本地算同样的指纹，
再去请求线上接口比对。相同 = 线上就是当前这份 SDK 代码；不同 = 需要重新发布。

注意：不读取、不上传任何密钥或用户数据；指纹只由源码字节计算。
若本机开了全局代理导致请求失败，脚本会给出可直接在手机浏览器打开的核对链接。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from urllib import request

sys.path.insert(0, str(Path(__file__).resolve().parent))

from web.server import BUILD  # noqa: E402

STATUS_URL = (
    "https://campus-food-advisor.app.workbuddy.host"
    "/api/status?sharecode=NDHJM5B3NI"
)


def main() -> None:
    print(f"本地源码指纹 : {BUILD['hash']}  （{BUILD['files']} 个源码文件）")
    print(f"线上核对地址 : {STATUS_URL}")
    try:
        with request.urlopen(STATUS_URL, timeout=12) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        print(f"\n本机无法直连线上接口（{type(exc).__name__}），常见于全局代理/VPN 劫持。")
        print("请用手机流量打开上面的核对地址，把 build 字段与本地指纹比对即可。")
        return

    remote = data.get("build", "")
    print(f"线上源码指纹 : {remote}")
    print(f"模型：{data.get('llm_model')} | 地图：{'已连接' if data.get('map') else '未配置'}"
          f" | 出发点：{data.get('origin')}")
    if remote == BUILD["hash"]:
        print("\n✅ 一致：线上运行的就是当前 SDK 目录里的这份代码。")
    else:
        print("\n⚠️ 不一致：线上是旧版本。请在 WorkBuddy 里重新发布本工程后再核对。")


if __name__ == "__main__":
    main()
