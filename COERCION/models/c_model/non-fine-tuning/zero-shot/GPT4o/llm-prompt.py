import os
import json
import requests
from tqdm import tqdm
from volcenginesdkarkruntime import Ark
from openai import OpenAI  # 新增：引入OpenAI库用于调用阿里云

# ==================== 1. 基础工具函数 ====================

def handle_streaming_response(response):
    """
    处理普通中转API的流式响应
    """
    full_response = ""
    for line in response.iter_lines():
        if line:
            line_str = line.decode('utf-8').lstrip("data: ").strip()
            if line_str == "[DONE]":
                break
            try:
                data = json.loads(line_str)
                chunk = data["choices"][0]["delta"].get("content", "")
                full_response += chunk
                print(chunk, end="", flush=True)
            except json.JSONDecodeError:
                continue
    return full_response.strip()

# ==================== 2. 各平台调用函数 ====================

def chat_with_streaming_prompt(prompt, user_key, model):
    """
    渠道A: 向中转 API 发送请求
    """
    url = "https://autobak.zaiwen.top/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {user_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "stream": True,
        "messages": [{"role": "user", "content": prompt}]
    }

    response = requests.post(url, headers=headers, json=payload, stream=True)
    if response.status_code == 200:
        return {"content": handle_streaming_response(response), "reasoning_content": ""}
    else:
        print("❌ 请求出错:", response.text)
        return {"content": "", "reasoning_content": ""}


def chat_with_ark(prompt, api_key, model):
    """
    渠道B: 使用火山引擎方舟SDK
    """
    try:
        client = Ark(api_key=api_key, timeout=1800)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            temperature=0.6
        )
        
        full_response = ""
        reasoning_content = ""
        is_reasoning = False
        
        for chunk in response:
            # 处理思维链
            if hasattr(chunk.choices[0].delta, 'reasoning_content') and chunk.choices[0].delta.reasoning_content:
                r_chunk = chunk.choices[0].delta.reasoning_content
                reasoning_content += r_chunk
                if not is_reasoning:
                    print("\n🤔 思维链内容 (Ark):")
                    is_reasoning = True
                print(r_chunk, end="", flush=True)
            
            # 处理正文
            if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                c_chunk = chunk.choices[0].delta.content
                full_response += c_chunk
                if is_reasoning:
                    print("\n\n💡 最终回答:")
                    is_reasoning = False
                print(c_chunk, end="", flush=True)
        
        return {"content": full_response, "reasoning_content": reasoning_content}
        
    except Exception as e:
        print(f"❌ 方舟API调用出错: {e}")
        return {"content": "", "reasoning_content": ""}


def chat_with_aliyun(prompt, api_key, model):
    """
    渠道C: 使用阿里云百炼 (OpenAI兼容接口)
    """
    try:
        client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            extra_body={"enable_thinking": True}, # 开启思考模式的关键参数
            stream=True,
            stream_options={"include_usage": True}
        )

        full_response = ""
        reasoning_content = ""
        is_answering = False # 用于标记是否已经从思考转入回答

        print("\n" + "=" * 10 + " 思考过程 (Aliyun) " + "=" * 10 + "\n")

        for chunk in response:
            # 过滤掉 usage 信息块
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            # 1. 收集思考内容
            if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
                r_chunk = delta.reasoning_content
                reasoning_content += r_chunk
                print(r_chunk, end="", flush=True)

            # 2. 收集回复内容
            if hasattr(delta, "content") and delta.content is not None:
                if not is_answering:
                    print("\n\n" + "=" * 10 + " 完整回复 " + "=" * 10 + "\n")
                    is_answering = True
                
                c_chunk = delta.content
                full_response += c_chunk
                print(c_chunk, end="", flush=True)

        return {"content": full_response, "reasoning_content": reasoning_content}

    except Exception as e:
        print(f"❌ 阿里云API调用出错: {e}")
        return {"content": "", "reasoning_content": ""}


# ==================== 3. 核心处理逻辑 ====================

def process_jsonl(input_path, output_path, config):
    """
    批量处理 jsonl 文件
    """
    # 读取数据
    if not os.path.exists(input_path):
        print(f"❌ 输入文件不存在: {input_path}")
        return

    with open(input_path, "r", encoding="utf-8") as fin:
        lines = [json.loads(line) for line in fin if line.strip()]

    provider = config["provider"]
    
    with open(output_path, "w", encoding="utf-8") as fout:
        for item in tqdm(lines, desc=f"处理中 [{provider}]"):
            prompt = item["prompt"]
            result = {}

            # 根据配置选择不同的处理函数
            if provider == "aliyun":
                result = chat_with_aliyun(prompt, config["api_key"], config["model"])
            elif provider == "ark":
                result = chat_with_ark(prompt, config["api_key"], config["model"])
            else: # standard / zaiwen
                result = chat_with_streaming_prompt(prompt, config["api_key"], config["model"])

            print(result)
            # 打印换行，保持控制台整洁
            print(f"\n✅ End.\n")

            # 保存结果
            item["completion"] = result["content"]
            # 如果有思考过程，也保存下来
            if result.get("reasoning_content"):
                item["reasoning_content"] = result["reasoning_content"]
            
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")


# ==================== 4. 入口函数 ====================

if __name__ == "__main__":
    
    # 👉👉👉在此处配置你的参数 👈👈👈
    
    # 选项: "aliyun", "ark", "zaiwen"
    API_PROVIDER = "zaiwen" 

    # 配置字典
    CONFIG = {
        "aliyun": {
            "api_key": "sk-REPLACE_WITH_ALIYUN_API_KEY", # 填入阿里云 API Key
            "model": "deepseek-r1"
        },
        "ark": {
            "api_key": "a09c9147-f195-485f-8271-ac553d30f032",  # me
            # "api_key": "c1803690-cf39-4985-9c1d-71b138299198",  # zys
            # "api_key": "43a74914-87b7-4735-bc73-db98a6674b16", # zxy
            # "api_key": "0e1fac76-c082-415b-bb3f-96814669d03b",  # sa
            # "api_key": "ce756c98-c605-406f-907e-266ce8ee57ae", # fsj
            "model": "deepseek-r1-250528"
        },
        "zaiwen": {
            "api_key": "5eathqef2qu582hy61fa581px43k2kpp",
            "model": "gpt-4o"
        }
    }

    # 路径配置
    input_jsonl = "GPT4o/zero-shot/prompt/prompt_test.jsonl"
    output_jsonl = f"GPT4o/zero-shot/clean/uncleaned_data_{API_PROVIDER}_test.jsonl"

    # 检查 Key
    current_config = CONFIG.get(API_PROVIDER)
    if not current_config or not current_config["api_key"]:
        print(f"❌ 错误: 未找到 {API_PROVIDER} 的配置或 API Key 为空。")
        exit()

    # 打印信息
    print(f"🚀 开始任务")
    print(f"📂 输入: {input_jsonl}")
    print(f"📂 输出: {output_jsonl}")
    print(f"🤖 渠道: {API_PROVIDER} | 模型: {current_config['model']}")
    print("-" * 40)

    # 构造传递给处理函数的统一配置对象
    process_config = {
        "provider": API_PROVIDER,
        "api_key": current_config["api_key"],
        "model": current_config["model"]
    }

    process_jsonl(input_jsonl, output_jsonl, process_config)