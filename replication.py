
#%%
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')




from torch.nn.functional import cosine_similarity
def find_closest_token(target_embedding, embedding_matrix):
    target_norm = target_embedding /target_embedding.norm(dim=-1, keepdim=True) 
    embedding_norm = embedding_matrix / embedding_matrix.norm(dim=-1, keepdim=True)
    similarities = cosine_similarity(target_norm, embedding_norm)
    closest_idx = similarities.argmax()
    return closest_idx
#%%
# Convert target word to token ID
target_token_id = tokenizer.encode("5", return_tensors="pt")[0]
print(target_token_id)

#%%
input_embeddings = torch.randn((1,1,model.config.n_embd), requires_grad=True)
close_token = find_closest_token(input_embeddings, model.get_input_embeddings().weight)
tokenizer.decode(close_token.item())




#%%


# %%

from torch.optim import Adam
from torch.nn.functional import cross_entropy

optimizer = Adam([input_embeddings], lr=0.1)
steps = 100


for step in range(steps):
    # Zero gradients
    optimizer.zero_grad()
    
    # Pass embeddings directly to the model
    outputs = model(inputs_embeds=input_embeddings)
    logits = outputs.logits
    # print("logits", logits.shape)
    
    # Calculate loss between the output logits and the target token ID
    # Assume the target token is the first token in the sequence
    loss = cross_entropy(logits[:, -1, :], target_token_id)
    
    # Backpropagate the loss
    loss.backward()
    
    # Update the embeddings
    optimizer.step()
    
    if step % 10 == 0:
        print(f"Step {step}: Loss = {loss.item()}")
# %%

close_token = find_closest_token(input_embeddings, model.get_input_embeddings().weight)
tokenizer.decode(close_token.item()) 
# %%
tokenizer.decode(20)