#!/usr/bin/env python
"""예제 LangChain 서버는 대화형 검색 에이전트를 노출합니다.

관련 LangChain 문서:

* 사용자 정의 에이전트 생성: https://python.langchain.com/docs/modules/agents/how_to/custom_agent
* 에이전트와 함께 스트리밍: https://python.langchain.com/docs/modules/agents/how_to/streaming#custom-streaming-with-events
* 일반 스트리밍 문서: https://python.langchain.com/docs/expression_language/streaming

**주의**
1. 개별 토큰 스트리밍을 지원하려면 스트리밍 엔드포인트 대신 astream 이벤트 엔드포인트를 사용해야 합니다.
2. 이 예제는 메시지 기록을 잘라내지 않으므로 너무 많은 메시지를 보내면 (토큰 길이를 초과) 충돌이 발생합니다.
3. 현재 플레이그라운드는 에이전트 출력을 잘 렌더링하지 않습니다! 플레이그라운드를 사용하려면 astream 이벤트를 사용하여 서버 측에서 출력을 사용자 정의해야 합니다.
4. 클라이언트 노트북을 참조하면 클라이언트 측에서 stream_events를 사용하는 예제가 있습니다.
"""
from typing import Any
from fastapi import FastAPI
from langchain.agents import AgentExecutor
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.pydantic_v1 import BaseModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langserve import add_routes
# OpenAI LLM 초기화
llm = ChatOpenAI(
    model='deepseek-chat', 
    openai_api_key='apikey', 
    openai_api_base='https://api.deepseek.com/beta',
    max_tokens=8000,
    streaming=True
)
# 프롬프트 템플릿 정의
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        # 사용자 입력과 에이전트 스크래치패드의 순서가 중요합니다.
        # 에이전트 스크래치패드는 에이전트가 생각하고,
        # 도구를 호출하고, 도구 출력을 확인하여 주어진
        # 사용자 입력에 응답하는 작업 공간입니다. 사용자 입력 뒤에 와야 합니다.
        ("user", "{input}"),
        # MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# LLM에서 개별 토큰 스트리밍을 지원하려면 streaming=True로 설정해야 합니다.
# 토큰은 stream_log / stream events 엔드포인트를 사용할 때 사용할 수 있지만,
# 에이전트가 작업 관찰 쌍을 스트리밍하기 때문에 stream 엔드포인트를 사용할 때는 사용할 수 없습니다.
# stream events 엔드포인트를 사용하는 방법을 보여주는 클라이언트 노트북을 참조하세요.

tools = [] 

agent = (
    {
        "input": lambda x: x["input"],
        # "agent_scratchpad": lambda x: "",  # 스크래치패드 사용 안 함
    }
    | prompt
    | llm
    | OpenAIFunctionsAgentOutputParser()
)

agent_executor = AgentExecutor(agent=agent, tools=tools)

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="LangChain의 Runnable 인터페이스를 사용하여 간단한 API 서버를 실행합니다."
)

# 현재 AgentExecutor는 스키마가 부족하므로 입력/출력 스키마를 추가해야 합니다.
class Input(BaseModel):
    input: str

class Output(BaseModel):
    output: Any
    
# 체인을 사용하기 위해 앱에 경로 추가:
# /invoke
# /batch
# /stream
# /stream_events

add_routes(
    app,
    agent_executor.with_types(input_type=Input, output_type=Output).with_config(
        {"run_name": "agent"}
    ),
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)

