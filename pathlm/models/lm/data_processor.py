from datasets import load_dataset


def normalise_path(x):
    ans = []
    for text in x['text']:
        tmp = []
        for i, entity in enumerate(text.split(" ")):
            if i == 0:
                tmp.append("<start_previous_interacted_product>")
                tmp.append(entity.replace("_", " "))
                tmp.append("<end_previous_interacted_product>")
            elif i == len(text.split(" ")) - 1:
                tmp.append("<start_recommended_product>")
                tmp.append(entity.replace("_", " "))
                tmp.append("end_recommended_product>")
            elif i == 2:
                tmp.append("<start_external_entity>")
                tmp.append(entity.replace("_", " "))
                tmp.append("<end_external_entity>")
            else:
                tmp.append("<start_relation>")
                tmp.append(entity.replace("_", " "))
                tmp.append("<end_relation>")
        ans.append(' '.join(tmp))
    x['text'] = ans
    return x

dataset = load_dataset('csv', data_files='data/ml1m/path_with_names.csv', split='train')
dataset = dataset.map(normalise_path, batched=True, num_proc=4)
dataset.save_to_disk('data/ml1m/hf_dataset.hf')