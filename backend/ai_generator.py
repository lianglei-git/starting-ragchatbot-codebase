import anthropic
from typing import List, Optional, Dict, Any

class AIGenerator:
    """处理与Anthropic Claude API的交互以生成响应"""
    
    # 静态系统提示，避免每次调用时重新构建
    SYSTEM_PROMPT = """ 你是一个专门处理课程材料和教育内容的AI助手，可以访问用于课程信息的全面搜索工具。

搜索工具使用：
- 仅对特定课程内容或详细教育材料的问题使用搜索工具
- 每次查询最多进行一次搜索
- 将搜索结果综合成准确的、基于事实的响应
- 如果搜索没有结果，请明确说明，不要提供替代方案

响应协议：
- **一般知识问题**：使用现有知识回答，无需搜索
- **课程特定问题**：先搜索，然后回答
- **无元评论**：
 - 仅提供直接答案——不提供推理过程、搜索解释或问题类型分析
 - 不要提及"基于搜索结果"


所有响应必须：
1. **简洁和专注** - 快速切入重点
2. **具有教育性** - 保持教学价值
3. **清晰** - 使用易于理解的语言
4. **包含示例支持** - 在有助于理解时包含相关示例
仅提供所问内容的直接答案。
"""
    
    def __init__(self, api_key: str, model: str):
        """
        初始化AI生成器
        
        Args:
            api_key: Anthropic API密钥
            model: Claude模型名称
        """
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        
        # 预构建基础API参数以提高效率
        self.base_params = {
            "model": self.model,
            "temperature": 0,  # 温度设置为0以确保确定性响应
            "max_tokens": 800  # 最大令牌数
        }
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        生成AI响应，支持可选工具使用和对话上下文
        
        Args:
            query: 用户的问题或请求
            conversation_history: 先前的对话消息作为上下文
            tools: AI可以使用的可用工具
            tool_manager: 执行工具的管理器
            
        Returns:
            生成的响应字符串
        """
        
        # 高效构建系统内容 - 尽可能避免字符串操作
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history 
            else self.SYSTEM_PROMPT
        )
        
        # 准备API调用参数
        api_params = {
            **self.base_params,
            "messages": [{"role": "user", "content": query}],
            "system": system_content
        }
        
        # 如果可用，添加工具
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}  # 自动选择工具
        
        # 从Claude获取响应
        response = self.client.messages.create(**api_params)
        print(response)
        
        # 如果需要，处理工具执行
        if response.stop_reason == "tool_use" and tool_manager:
            return self._handle_tool_execution(response, api_params, tool_manager)
        
        # 返回直接响应
        return response.content[0].text
    
    def _handle_tool_execution(self, initial_response, base_params: Dict[str, Any], tool_manager):
        """
        处理工具调用的执行并获取后续响应
        
        Args:
            initial_response: 包含工具使用请求的响应
            base_params: 基础API参数
            tool_manager: 执行工具的管理器
            
        Returns:
            工具执行后的最终响应文本
        """
        # 从现有消息开始
        messages = base_params["messages"].copy()
        
        # 添加AI的工具使用响应
        messages.append({"role": "assistant", "content": initial_response.content})
        
        # 执行所有工具调用并收集结果
        tool_results = []
        for content_block in initial_response.content:
            if content_block.type == "tool_use":
                tool_result = tool_manager.execute_tool(
                    content_block.name, 
                    **content_block.input
                )
                
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": content_block.id,
                    "content": tool_result
                })
        
        # 将工具结果作为单个消息添加
        if tool_results:
            messages.append({"role": "user", "content": tool_results})
        
        # 准备不带工具的最终API调用
        final_params = {
            **self.base_params,
            "messages": messages,
            "system": base_params["system"]
        }
        
        # 获取最终响应
        final_response = self.client.messages.create(**final_params)
        return final_response.content[0].text