from torch import nn, optim, tensor, device

from neural_machine_translation import NMTEncoder, NMTDecoder
from prepare_data import NMTDataset

BATCH_SIZE = 64

def train(data, encoder, decoder, criterion, e_optimizer, de_optimizer, tag_ix, save_path, debug_steps=10, epoch=10,
          device=device('cpu')):
    for e in range(epoch):
      encoder.train(True)
      decoder.train(True)
      ovr_loss = 0
      hidden = encoder.init_hidden(BATCH_SIZE)
      for i, batch in enumerate(data):
          print(i)
          e_optimizer.zero_grad()
          de_optimizer.zero_grad()
          src = batch.Russian.to(device)
          trg = batch.English.to(device)

          if src.shape[0] != BATCH_SIZE:
              continue

          loss = 0
          encoder_output, hidden = encoder(src, hidden)
          decoder_input = tensor([[tag_ix]] * BATCH_SIZE)
          

          for t in range(1, trg.size(1)):
              
              out, hidden = decoder(decoder_input,
                                    hidden,
                                    encoder_output)
              
              loss += criterion(out, trg[:, t])
              decoder_input = trg[:, t].unsqueeze(1)

          loss.backward()
          e_optimizer.step()
          de_optimizer.step()
          ovr_loss += loss.item()
          if i % debug_steps == 0:
              print(f'Epoch {e}: loss={ovr_loss / (i + 1)}')
              torch.save(encoder.state_dict(), save_path+"/encoder.pth")
              torch.save(decoder.state_dict(), save_path+"/decoder.pth")


def main():
    learning_rate = 1e-3
    hidden_dim = 256

    dataset = NMTDataset('data/corpus.en_ru.1m.en',
                         'data/corpus.en_ru.1m.ru')

    train_data, validation_data = dataset.get_iterators(batch_size=BATCH_SIZE, debug=True)
    input_dim, output_dim = dataset.INPUT_DIM, dataset.OUTPUT_DIM

    encoder = NMTEncoder(input_dim, hidden_dim)
    decoder = NMTDecoder(hidden_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    e_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    de_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    train(train_data, encoder, decoder, criterion, e_optimizer, de_optimizer, dataset.get_src_tag_idx('<eos>'),".")


if __name__ == '__main__':
    main()
