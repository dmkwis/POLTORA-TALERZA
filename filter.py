import dataset

FINETUNE_SIZE = 100000
PRETRAIN_SIZE = 300000


def filter_unique_word(data_x, data_y, word_cnt):
    new_data_x, new_data_y = [], []
    reduced = 0
    for i, y in enumerate(data_y):
        keep = True
        for w in y:
            if word_cnt[w] < 50:
                keep = False
                reduced += 1
                break
        if keep:
            new_data_x.append(data_x[i])
            new_data_y.append(y)

    print('reduced words by:', reduced)
    return new_data_x, new_data_y


def filter_rare(data_x, data_y, word_cnt, data_size):
    data_scores = []
    for y in data_y:
        score = 0
        for w in y:
            score += word_cnt[w]
        score /= len(y)
        data_scores.append(score)
    
    data = list(zip(data_x, data_y))
    data_with_scores = list(zip(data, data_scores))
    data_with_scores = sorted(data_with_scores, key = lambda a: a[1])
    new_data = list(zip(*data_with_scores))
    new_data = list(new_data[0])
    new_data = new_data[(-data_size):]
    new_data = list(zip(*new_data))
    new_data_x, new_data_y = list(new_data[0]), list(new_data[1])

    print('reduced some more')
    return new_data_x, new_data_y


def fill_word_cnt():
    word_cnt = {}
    for verse in pretrain_y + finetune_y:
        words = set(verse)
        for w in words:
            if w not in word_cnt:
                word_cnt[w] = 0
            word_cnt[w] += 1
    return word_cnt


if __name__ == '__main__':
    pretrain_x, pretrain_y = dataset.get_data('pretrain')
    finetune_x, finetune_y = dataset.get_data('finetune')

    word_cnt = fill_word_cnt()
    
    print('words counted')
    pretrain_x, pretrain_y = filter_unique_word(pretrain_x, pretrain_y, word_cnt)
    finetune_x, finetune_y = filter_unique_word(finetune_x, finetune_y, word_cnt)

    word_cnt = fill_word_cnt()

    pretrain_x, pretrain_y = filter_rare(pretrain_x, pretrain_y, word_cnt, PRETRAIN_SIZE)
    finetune_x, finetune_y = filter_rare(finetune_x, finetune_y, word_cnt, FINETUNE_SIZE)

    with open('data/pretrain_x.txt', 'w') as file:
        pretrain_x = [' '.join(l) for l in pretrain_x]
        pretrain_x = [v.replace('\n ', '\n') for v in pretrain_x]
        pretrain_x = '\n\n'.join(pretrain_x)
        file.write(pretrain_x)
    with open('data/pretrain_y.txt', 'w') as file:
        pretrain_y = [' '.join(l) for l in pretrain_y]
        pretrain_y = [v.replace('\n ', '\n') for v in pretrain_y]
        pretrain_y = '\n\n'.join(pretrain_y)
        file.write(pretrain_y)
    print('overwritten pretrain')

    with open('data/finetune_x.txt', 'w') as file:
        finetune_x = [' '.join(l) for l in finetune_x]
        finetune_x = [v.replace('\n ', '\n') for v in finetune_x]
        finetune_x = '\n\n'.join(finetune_x)
        file.write(finetune_x)
    with open('data/finetune_y.txt', 'w') as file:
        finetune_y = [' '.join(l) for l in finetune_y]
        finetune_y = [v.replace('\n ', '\n') for v in finetune_y]
        finetune_y = '\n\n'.join(finetune_y)
        file.write(finetune_y)
    print('overwritten finetune')
