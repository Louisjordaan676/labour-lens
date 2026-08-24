from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langgraph.graph import MessagesState

load_dotenv()


def chatbot_node(state: MessagesState):
    # Because MessagesState behaves like a dictionary-like state object, we
    # can access the messages using the "messages" key.
    messages = state["messages"]
    # instead of invoking the llm with user input, we invoke it with the
    #  messages objects list
    response = llm.invoke(messages)
    return {"messages": [response]}


llm = ChatOpenAI(model="gpt-4o-mini")


while True:

    user_input = input("you:  ")

    if user_input == "e" or user_input == "bye":
        print("goodbye, chat to you soon.")
        break

    response = chatbot.invoke(user_input,
                              config={"configurable": {"session_id":
                                                       "conversation_1"}})

    print(response.content)
