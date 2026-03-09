from flask import Flask, request, jsonify, render_template
from embeddings import load_documents
from rag import retrieve
from openai import OpenAI

app = Flask(__name__)

# OpenAI client (PUT YOUR API KEY HERE)
client = OpenAI(api_key="YOUR_API_KEY")

# Load documents with embeddings
documents = load_documents()


# Function to call LLM
def get_llm_response(context, question):

    prompt = f"""
Use the following context to answer the question.

Context:
{context}

Question:
{question}

Answer:
"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    return response.choices[0].message.content


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        data = request.json
        message = data.get("message")

        # Retrieve similar documents
        chunks = retrieve(message, documents)

        context = chunks[0]["content"]

        # Generate answer using LLM
        answer = "Based on the documents:\n\n" + context

        return jsonify({
            "reply": answer,
            "retrievedChunks": len(chunks)
        })

    except Exception as e:
        print("SERVER ERROR:", e)

        return jsonify({
            "reply": "Server error occurred",
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(debug=True)