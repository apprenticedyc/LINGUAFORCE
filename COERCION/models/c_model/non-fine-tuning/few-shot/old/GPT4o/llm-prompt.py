import os
import json
import time
import requests
from tqdm import tqdm
from volcenginesdkarkruntime import Ark
from openai import OpenAI  # 用于调用阿里云或标准 OpenAI 接口

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
# 【修改点1】将参数从 prompt 改为 messages，并直接传递

def chat_with_streaming_prompt(messages, user_key, model, max_retries=3):
    """
    渠道A: 向中转 API 发送请求 (加入自动重试和异常捕获)
    """
    url = "https://autobak.zaiwen.top/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {user_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "stream": True,
        "messages": messages
    }

    for attempt in range(max_retries):
        try:
            # 加上 timeout，防止死等
            response = requests.post(url, headers=headers, json=payload, stream=True, timeout=60)
            
            if response.status_code == 200:
                content = handle_streaming_response(response)
                # 如果成功获取到内容，直接返回
                if content: 
                    return {"content": content, "reasoning_content": ""}
            else:
                print(f"❌ 请求出错 (状态码: {response.status_code}): {response.text}")
                
        except requests.exceptions.ChunkedEncodingError as e:
            print(f"\n⚠️ 服务器断流 (第 {attempt + 1}/{max_retries} 次尝试)...")
        except requests.exceptions.RequestException as e:
            print(f"\n⚠️ 网络请求异常 (第 {attempt + 1}/{max_retries} 次尝试): {e}")
        except Exception as e:
            print(f"\n⚠️ 未知错误 (第 {attempt + 1}/{max_retries} 次尝试): {e}")

        # 如果失败了，休息 2 秒再重试
        if attempt < max_retries - 1:
            time.sleep(2)

    # 如果重试了 max_retries 次还是失败，返回空字符串，避免阻断后续数据的处理
    print("\n❌ 该条数据多次重试均失败，已跳过。")
    return {"content": "", "reasoning_content": ""}


def chat_with_ark(messages, api_key, model, max_retries=3):
    """
    渠道B: 使用火山引擎方舟SDK (加入自动重试机制)
    """
    for attempt in range(max_retries):
        try:
            client = Ark(api_key=api_key, timeout=1800)
            response = client.chat.completions.create(
                model=model,
                messages=messages,  # 直接传入完整的多轮对话数组
                stream=True,
                temperature=0.0     # 建议分类任务将温度设为0或很低
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
            
            # 如果成功走完流式读取，直接返回结果
            return {"content": full_response, "reasoning_content": reasoning_content}
            
        except Exception as e:
            print(f"\n⚠️ 方舟API调用异常 (第 {attempt + 1}/{max_retries} 次尝试): {e}")
            if attempt < max_retries - 1:
                time.sleep(2)  # 失败后休息2秒再试
                
    print("\n❌ 方舟API多次重试均失败，已跳过该条数据。")
    return {"content": "", "reasoning_content": ""}


def chat_with_aliyun(messages, api_key, model, max_retries=3):
    """
    渠道C: 使用阿里云百炼 (OpenAI兼容接口) (加入自动重试机制)
    """
    for attempt in range(max_retries):
        try:
            client = OpenAI(
                api_key=api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )

            response = client.chat.completions.create(
                model=model,
                messages=messages,  # 直接传入完整的多轮对话数组
                extra_body={"enable_thinking": True}, # 开启思考模式的关键参数
                stream=True,
                stream_options={"include_usage": True}
            )

            full_response = ""
            reasoning_content = ""
            is_answering = False

            print("\n" + "=" * 10 + " 思考过程 (Aliyun) " + "=" * 10 + "\n")

            for chunk in response:
                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta

                if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
                    r_chunk = delta.reasoning_content
                    reasoning_content += r_chunk
                    print(r_chunk, end="", flush=True)

                if hasattr(delta, "content") and delta.content is not None:
                    if not is_answering:
                        print("\n\n" + "=" * 10 + " 完整回复 " + "=" * 10 + "\n")
                        is_answering = True
                    
                    c_chunk = delta.content
                    full_response += c_chunk
                    print(c_chunk, end="", flush=True)

            # 如果成功走完流式读取，直接返回结果
            return {"content": full_response, "reasoning_content": reasoning_content}

        except Exception as e:
            print(f"\n⚠️ 阿里云API调用异常 (第 {attempt + 1}/{max_retries} 次尝试): {e}")
            if attempt < max_retries - 1:
                time.sleep(2)  # 失败后休息2秒再试
                
    print("\n❌ 阿里云API多次重试均失败，已跳过该条数据。")
    return {"content": "", "reasoning_content": ""}


# ==================== 3. 核心处理逻辑 ====================

def process_jsonl(input_path, output_path, config):
    if not os.path.exists(input_path):
        print(f"❌ 输入文件不存在: {input_path}")
        return

    with open(input_path, "r", encoding="utf-8") as fin:
        lines = [json.loads(line) for line in fin if line.strip()]

    provider = config["provider"]
    
    with open(output_path, "w", encoding="utf-8") as fout:
        for item in tqdm(lines, desc=f"处理中 [{provider}]"):
            
            # 【修改点2】智能兼容层：判断数据格式
            if "messages" in item:
                # 兼容你新生成的 Few-shot 数据
                messages = item["messages"]
            elif "prompt" in item:
                # 兼容你以前老的 Zero-shot 数据
                messages = [{"role": "user", "content": item["prompt"]}]
            else:
                print(f"⚠️ 跳过格式异常的数据: {item.get('dialogue_id', 'unknown')}")
                continue

            result = {}

            # 将 messages 传递给大模型
            if provider == "aliyun":
                result = chat_with_aliyun(messages, config["api_key"], config["model"])
            elif provider == "ark":
                result = chat_with_ark(messages, config["api_key"], config["model"])
            else: 
                result = chat_with_streaming_prompt(messages, config["api_key"], config["model"])

            print(f"\n✅ End.\n")

            # 保存结果
            item["completion"] = result["content"]
            if result.get("reasoning_content"):
                item["reasoning_content"] = result["reasoning_content"]
            
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")


# ==================== 4. 入口函数 ====================

if __name__ == "__main__":
    
    API_PROVIDER = "zaiwen" # 这里切换提供商
    
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

    K_SHOT = 1
    # 【修改点3】记得把这里的路径改成你新生成的 Few-shot 数据的路径
    input_jsonl = f"GPT4o/few-shot/prompt/prompt_test_fewshot_{K_SHOT}.jsonl" 
    output_jsonl = f"GPT4o/few-shot/clean/fewshot_{K_SHOT}/uncleaned_data_{API_PROVIDER}_test.jsonl"

    current_config = CONFIG.get(API_PROVIDER)
    if not current_config or not current_config["api_key"]:
        print(f"❌ 错误: 未找到 {API_PROVIDER} 的配置或 API Key 为空。")
        exit()

    print(f"🚀 开始任务")
    print(f"📂 输入: {input_jsonl}")
    print(f"📂 输出: {output_jsonl}")
    print(f"🤖 渠道: {API_PROVIDER} | 模型: {current_config['model']}")
    print("-" * 40)

    process_config = {
        "provider": API_PROVIDER,
        "api_key": current_config["api_key"],
        "model": current_config["model"]
    }

    process_jsonl(input_jsonl, output_jsonl, process_config)