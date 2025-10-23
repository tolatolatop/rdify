#!/usr/bin/env python3
"""
测试 Moonshot API 连接的脚本
"""
import os
import sys
from openai import OpenAI

def test_moonshot_connection():
    """测试 Moonshot API 连接"""
    
    # 从环境变量获取 API 密钥
    api_key = os.getenv("MOONSHOT_API_KEY")
    if not api_key:
        print("❌ 错误: 请设置 MOONSHOT_API_KEY 环境变量")
        print("   例如: export MOONSHOT_API_KEY='your-api-key-here'")
        return False
    
    # 创建客户端
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.moonshot.cn/v1"
    )
    
    try:
        print("🔍 正在测试 Moonshot API 连接...")
        
        # 测试模型列表
        print("📋 获取可用模型列表...")
        models = client.models.list()
        print(f"✅ 成功获取 {len(models.data)} 个模型")
        
        # 显示前几个模型
        for i, model in enumerate(models.data[:3]):
            print(f"   {i+1}. {model.id}")
        
        # 测试聊天完成
        print("\n💬 测试聊天完成功能...")
        response = client.chat.completions.create(
            model="moonshot-v1-8k",
            messages=[
                {"role": "user", "content": "你好，请简单介绍一下自己"}
            ],
            max_tokens=100
        )
        
        print("✅ 聊天完成测试成功")
        print(f"回复: {response.choices[0].message.content}")
        
        return True
        
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        return False

if __name__ == "__main__":
    success = test_moonshot_connection()
    sys.exit(0 if success else 1)
