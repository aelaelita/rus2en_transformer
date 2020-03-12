from torch import nn, optim

from neural_machine_translation import NMTEncoder, NMTDecoder
from prepare_data import get_iterators


def train(data, encoder, decoder, criterion, e_optimizer, de_optimizer, debug_steps=100, epoch=-1):
    encoder.train(True)
    decoder.train(True)
    e_optimizer.zero_grad()
    de_optimizer.zero_grad()
    ovr_loss = 0
    hidden = encoder.init_hidden()
    for i, batch in enumerate(data):
        src = batch.Russian
        trg = batch.English
        encoder_output, hidden = encoder(src, hidden)
        output, hidden = decoder(src, hidden, encoder_output)
        loss = criterion(output, trg)
        loss.backward()
        e_optimizer.step()
        de_optimizer.step()
        ovr_loss += loss.item()
        if i % debug_steps == 0:
            print(f'Epoch {epoch} ({i}/{len(data)}): loss={ovr_loss / i}')


def main():
    train_data, validation_data = get_iterators('corpus.en_ru.1m.en', 'corpus.en_ru.1m.ru', batch_size=128,
                                                debug=True)
    encoder = NMTEncoder()
    decoder = NMTDecoder()
    criterion = nn.NLLLoss()
    learning_rate = 1e-3
    e_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    de_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    train(train_data, encoder, decoder, criterion, e_optimizer, de_optimizer)


if __name__ == '__main__':
    main()