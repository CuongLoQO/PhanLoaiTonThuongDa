from openai import OpenAI

client = OpenAI(
    api_key="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"
)

def ask_chatgpt():
    question = input("Bạn muốn hỏi gì về bệnh da liễu? ")
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": question}
        ]
    )
    print("\nChatGPT trả lời:")
    print(response.choices[0].message.content)

if __name__ == "__main__":
    ask_chatgpt()