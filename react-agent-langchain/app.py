from typing import Union, List
from dotenv import load_dotenv

from langchain.prompts import PromptTemplate
from langchain.agents import tool, Tool
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.tools.render import render_text_description
from langchain.agents.format_scratchpad.log import format_log_to_str
from langchain_core.agents import AgentAction, AgentFinish
from langchain_openai.chat_models import ChatOpenAI

load_dotenv()


@tool
def get_text_length(text: str) -> int:
    """
    Returns the length of a text by characters
    """

    print(f"DEBUG: get_text_length enter with {text=}")
    # 주석이 추론 과정에 필수적으로 들어감. : get_text_length.description
    return len(text)


def find_tool_by_name(tools: List[Tool], tool_name: str) -> Tool:
    for _tool in tools:
        if _tool.name == tool_name:
            return _tool
        raise ValueError(f"Tool with name {tool_name} in tools")


def main():
    tools = [get_text_length]

    template = """
    Answer the following questions as best you can. You have access to the following tools:

    {tools}
    
    Use the following format:
    
    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question
    
    Begin!
    
    Question: {input}
    Thought: {agent_scratchpad}
    """

    # tools 도 LLM에게 string 형태로 제공해야 함
    prompt = PromptTemplate.from_template(template=template).partial(
        tools=render_text_description(tools=tools),
        tool_names=", ".join([t.name for t in tools]),
    )

    """
    stop argument: Observation
    LLM에게 단어 생성을 멈추고 작동을 끝내라는 명령을 보냄
    stop 토큰을 넣지 않으면 계속해서 텍스트를 생성하고,
    관찰할 때마다 단어를 하나씩 추측하게 되기 때문임
    관찰은 도구의 결과물이고, 도구를 실행할 때 따라오는 것
    
    LLM 실행에서 stop 토큰의 중요성을 이해하게 될 것임!!
    """
    llm = ChatOpenAI(temperature=0, stop=["\nObservation"])

    """
    #
    이 파이프가 뭘까? 
    -> LangChain Expression Language, LCEL
    선언적으로 정의하고 체인과 함께 구성할 수 있음
    - 코드를 읽고 구성하는 게 더 쉬워지고
    - 내부적으로 자세히 확인 가능
    - 병렬처리, 일괄처리, 스트리밍 지원 대체 등등 

    #
    파이프 오퍼레이터(|)는 왼쪽의 출력을 오른쪽의 입력으로 플러그인 함 
    직관적으로 생각하면 프롬프트를 받아 LLM에 플러그인한 다음 실행하는 것임
    (I/O Type은 마크다운 Component Input/Output Type 참조)

    #
    chain을 invoke 하기 위해서는 자리를 replace 할 분을 dict로 제시해줘야 함
    """

    """
    history를  추적하려면 intermediate_steps라는 새 변수가 필요함
    """
    intermediate_steps = []

    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_log_to_str(x["agent_scratchpad"]),
        }
        | prompt
        | llm
        | ReActSingleInputOutputParser()
    )

    question_message = "What is the length in characters of the text DOG?"

    cnt = 1
    while cnt < 5:
        agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
            {"input": question_message, "agent_scratchpad": intermediate_steps}
        )

        print(f"=== agent_step {cnt} answer ===")
        print(f"{agent_step=}\n\n")

        if isinstance(agent_step, AgentAction):
            tool_name = agent_step.tool
            tool_to_use = find_tool_by_name(tools, tool_name)
            tool_input = agent_step.tool_input
            observation = tool_to_use.func(str(tool_input))

            print(f"{observation=}")
            intermediate_steps.append((agent_step, str(observation)))
            cnt += 1

        elif isinstance(agent_step, AgentFinish):
            print(f"{agent_step.return_values=}")
            break


if __name__ == "__main__":
    print("Hello ReAct LangChain!")
    main()
