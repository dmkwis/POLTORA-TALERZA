import dataset

REMOVE_FRACTION = 0.5


def filter_rare(data_x, data_y):
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
    new_data = new_data[int(REMOVE_FRACTION * len(new_data)):]
    new_data = list(zip(*new_data))
    new_data_x, new_data_y = list(new_data[0]), list(new_data[1])
    return new_data_x, new_data_y


if __name__ == '__main__':
    pretrain_x, pretrain_y = dataset.get_data('pretrain')
    finetune_x, finetune_y = dataset.get_data('finetune')

    word_cnt = {}
    for verse in pretrain_y + finetune_y:
        for w in verse:
            if w not in word_cnt:
                word_cnt[w] = 0
            word_cnt[w] += 1
    
    print('words counted')
    pretrain_x, pretrain_y = filter_rare(pretrain_x, pretrain_y)
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

    finetune_x, finetune_y = filter_rare(finetune_x, finetune_y)
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
