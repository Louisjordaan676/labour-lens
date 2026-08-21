from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()


llm = ChatOpenAI(model="gpt-4o-mini")

while True:

    user_input = input("you:  ")

    if user_input == "e" or user_input == "bye":
        print("goodbye, chat to you soon.")
        break

    response = llm.invoke(user_input)

    print(response.content)
