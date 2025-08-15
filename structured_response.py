# https://blog.csdn.net/lovechris00/article/details/148014663
from pydantic import BaseModel
from langgraph.prebuilt import create_react_agent

def get_person() -> str:  # (1)!
    """Get weather for a given city."""
    return f"唐僧、猪八戒、孙悟空、二郎神、沙悟净"
class WeatherResponse(BaseModel):
    conditions: str

agent = create_react_agent(
    model="anthropic:claude-3-7-sonnet-latest",
    tools=[get_person()],
    response_format=WeatherResponse  # (1)!
)

response = agent.invoke(
    {"messages": [{"role": "user", "content": "西游记中师徒5人的出身"}]}
)

response["structured_response"]
