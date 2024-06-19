from sentence_transformers import SentenceTransformer
sentences = ["Since becoming a parent, one of the simplest ways to describe the experience is that so many things happen all the time.  Which is kind of what the stock market feels like right now. And is kind of what defines the market all the time.  Late Sunday, the latest round of GameStop (GME) fervor was kicked up again when an account believed to be tied to investor Keith Gill, who ignited the meme stock rally in 2021, revealed it spent nearly $175 million building a position in the video game retailer.  This led to a pop in the stock on Monday, with meme darling AMC (AMC) following suit."]

#model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
#model = SentenceTransformer('Salesforce/SFR-Embedding-Mistral',device_map='auto')
#model = SentenceTransformer('Snowflake/snowflake-arctic-embed-m',device='cpu')
model = SentenceTransformer('Snowflake/snowflake-arctic-embed-l',device='cpu')
embeddings = model.encode(sentences)
print(embeddings)

print(len(embeddings[0]))
