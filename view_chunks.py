import pickle

with open(
    "vector_store/college_chunks.pkl",
    "rb"
) as file:

    chunks = pickle.load(file)

print("Total chunks:", len(chunks))

for i, chunk in enumerate(chunks[:5]):

    print("\n==============================")
    print("CHUNK", i + 1)
    print("==============================")

    print(chunk)