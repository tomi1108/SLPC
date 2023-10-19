def add_ids(data, label):
    if len(data) != len(label):
        raise RuntimeError(">> Error: data and label are not same length.")
        

    return data[0], label[0]