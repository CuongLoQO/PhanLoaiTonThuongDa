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