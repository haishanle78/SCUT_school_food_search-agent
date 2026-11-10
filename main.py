"""CLI 入口：交互对话 / 演示模式。

用法：
    python main.py           # 交互对话（Ctrl+C 或输入 /quit 退出）
    python main.py --demo    # 跑 3 个预置示例问题（无 Key 时走离线演示）
"""

from __future__ import annotations

import sys

# Windows 控制台中文输出兜底
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import config
from agent.core import FoodAdvisorAgent
from agent.memory import UserProfile

DEMO_QUESTIONS = [
    "今天中午想吃辣的，预算 30 以内，不想走超过 15 分钟",
    "周末想和同学聚餐，大概 6 个人，人均 50 左右，哪家能坐得下？",
    "懒得出门，外卖有什么不踩雷的？",
]


def banner(agent: FoodAdvisorAgent) -> None:
    mode = "LLM 模式（DeepSeek/OpenAI 兼容）" if agent.llm.enabled else "离线模式（规则兜底，未配置 .env）"
    print("=" * 56)
    print("  国际校区干饭军师 · Food Advisor Agent")
    print(f"  当前模式：{mode}")
    print("  命令：/profile 查看画像  /reset 重开会话  /quit 退出")
    print("=" * 56)


def main() -> None:
    profile = UserProfile()
    agent = FoodAdvisorAgent(profile)
    banner(agent)

    if "--demo" in sys.argv:
        for i, question in enumerate(DEMO_QUESTIONS, 1):
            print(f"\n【示例 {i}】{question}")
            print("-" * 56)
            print(agent.run(question))
        print("\n演示结束。")
        return

    while True:
        try:
            user_input = input("\n你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n下次再聊，干饭愉快！")
            break
        if not user_input:
            continue
        if user_input == "/quit":
            print("下次再聊，干饭愉快！")
            break
        if user_input == "/profile":
            print(f"画像：{profile.as_prompt_text()}")
            continue
        if user_input == "/reset":
            agent = FoodAdvisorAgent(UserProfile())
            print("已重开会话与画像。")
            continue

        answer = agent.run(user_input)
        print(f"\n军师> {answer}")


if __name__ == "__main__":
    main()
