def to_onehot(value, vocabulary):
    onehot_list = [0 for _ in vocabulary]
    if value in vocabulary:
        onehot_list[vocabulary.index(value)] = 1
    return onehot_list
    