from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()

chat_history = InMemoryChatMessageHistory()


def get_session_history(session_id):
    return chat_history


llm = ChatOpenAI(model="gpt-4o-mini")

chatbot = RunnableWithMessageHistory(
    llm,
    get_session_history
)

while True:

    user_input = input("you:  ")

    if user_input == "e" or user_input == "bye":
        print("goodbye, chat to you soon.")
        break

    response = chatbot.invoke(user_input,
                              config={"configurable": {"session_id":
                                                       "conversation_1"}})

    print(response.content)
