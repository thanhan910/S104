from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration


# Initialize tokenizer and model
tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
retriever = RagRetriever.from_pretrained("facebook/rag-token-nq", index_name="custom", passages_path="test-text.txt")
model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq", retriever=retriever)

# Function to ask a question
def ask_question(question):
    # Encode the question
    input_ids = tokenizer(question, return_tensors="pt").input_ids

    # Generate the answer
    generated_ids = model.generate(input_ids)

    # Decode and print the answer
    print("Answer:", tokenizer.decode(generated_ids[0], skip_special_tokens=True))

# Example question
ask_question("What is the capital of Victoria?")
