import torch
from model import CharRNN, CharLSTM
from dataset import Shakespeare  

def generate(model, idx_to_char, char_to_idx, seed_characters, temperature, max_length):
    """
    Generate characters
    
    Args:
        model: trained model
        idx_to_char: dictionary mapping indices to characters
        char_to_idx: dictionary mapping characters to indices
        seed_characters: seed characters
        temperature: T (temperature for sampling)
        max_length: maximum length of generated sequence
        
    Returns:
        samples: generated characters
    """
    model.eval()
    
    samples = []
    
    for seed in seed_characters:
        generated = seed
        inp = torch.tensor([char_to_idx[c] for c in seed], dtype=torch.long).unsqueeze(0)
        
        hidden = model.init_hidden(1)
        
        for _ in range(max_length):
            output, hidden = model(inp, hidden)
            output_dist = output.squeeze().div(temperature).exp()
            top_char = torch.multinomial(output_dist, 1)[0].item()
            
            predicted_char = idx_to_char[top_char]
            
            generated += predicted_char
            inp = torch.tensor([[top_char]], dtype=torch.long)
            
        samples.append(generated)
        
    return samples

if __name__ == '__main__':
    rnn_model_path = 'rnn_model.pth' 
    lstm_model_path = 'lstm_model.pth' 
    
    dataset = Shakespeare(input_file='shakespeare_train.txt')
    
    input_size = len(dataset.chars)  
    hidden_size = 256
    num_layers = 2
    output_size = input_size
    
    rnn_model = CharRNN(input_size, hidden_size, output_size, num_layers)
    lstm_model = CharLSTM(input_size, hidden_size, output_size, num_layers)
    
    rnn_model.load_state_dict(torch.load(rnn_model_path))
    lstm_model.load_state_dict(torch.load(lstm_model_path))
    
    seed_characters = [
        'K',
        'B',
        'T',
        'C',
        'Q'
    ]
    
    temperature = [0.5,1,2]
    max_length = 100

    for T in temperature:
        print(f"Temperature : "+ str(T))
        print(f"RNN 모델 생성 결과:")
        rnn_samples = generate(rnn_model, dataset.idx_to_char, dataset.char_to_idx, seed_characters, T, max_length)
        for i, sample in enumerate(rnn_samples):
            print(f'Sample {i+1}:')
            print(sample)
            print()

    for T in temperature: 
        print(f"Temperature : "+ str(T))
        print(f"LSTM 모델 생성 결과:")
        lstm_samples = generate(lstm_model, dataset.idx_to_char, dataset.char_to_idx, seed_characters, T, max_length)
        for i, sample in enumerate(lstm_samples):
            print(f'Sample {i+1}:')
            print(sample)
            print()