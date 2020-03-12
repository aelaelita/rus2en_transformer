from torch import nn, optim, tensor

from neural_machine_translation import NMTEncoder, NMTDecoder
from prepare_data import get_iterators

BATCH_SIZE = 128


def train(data, encoder, decoder, criterion, e_optimizer, de_optimizer, debug_steps=100, epoch=-1):
    encoder.train(True)
    decoder.train(True)
    e_optimizer.zero_grad()
    de_optimizer.zero_grad()
    ovr_loss = 0
    hidden = encoder.init_hidden(BATCH_SIZE)
    for i, batch in enumerate(data):
        src = batch.Russian
        trg = batch.English
        encoder_output, hidden = encoder(src, hidden)
        # decoder_input = tensor()
        # for t in range(1, trg.size(1)):
        output, hidden = decoder(src, hidden, encoder_output)
        loss = criterion(output, trg)
        loss.backward()
        e_optimizer.step()
        de_optimizer.step()
        ovr_loss += loss.item()
        if i % debug_steps == 0:
            print(f'Epoch {epoch} ({i}/{len(data)}): loss={ovr_loss / i}')


def main():
    learning_rate = 1e-3
    hidden_dim = 256

    train_data, validation_data, input_dim, output_dim = get_iterators('data/corpus.en_ru.1m.en',
                                                                       'data/corpus.en_ru.1m.ru',
                                                                       batch_size=BATCH_SIZE, debug=True)

    encoder = NMTEncoder(input_dim, hidden_dim)
    decoder = NMTDecoder(hidden_dim, output_dim)
    criterion = nn.NLLLoss()
    e_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    de_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    train(train_data, encoder, decoder, criterion, e_optimizer, de_optimizer)


if __name__ == '__main__':
    main()
