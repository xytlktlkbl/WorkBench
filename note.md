# python to tool
一个python函数是怎么被LLM调用的？
首先进行工具注册，即获得对应的openai json schema格式， 并且建立起映射。

在向LLM发送信息的时候，会连工具列表一起发送过去，对应的，LLM的返回HTTP是工具调用。

接受到HTTP响应之后，在python后端开始工具查找，参数解析，调用。并把结果再传回LLM。LLM看到结果，开始下一次循环。

# 循环
标准React 循环：action和final answer2选1
本框架延续这个思路，直到没有工具调用，输出最终答案时，循环结束。
此外还有一个兜底策略，循环上线超出了限制就强制结束，避免死循环。

python scripts/inference/generate_multi_agent_results.py \
    --model_name deepseek-v4-flash \
    --queries_path data/processed/queries_and_answers/multi_domain_queries_and_answers.csv \
    --max_queries 3 --mode single_agent

python scripts/inference/generate_multi_agent_results.py \
    --model_name deepseek-v4-flash \
    --all_domains \
    --max_queries 3 --mode multi_agent_shared