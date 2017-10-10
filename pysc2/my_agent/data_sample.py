import numpy as np

if __name__ == "__main__":

    path = "C:/Users/chensy/Desktop/pysc2 source/data/demo1/"

    # orders = np.load(path + "order.npy")
    # label = orders[:, 1]
    #
    # # action_type: 0 : move, 1 : build_pylon, 2 : build_forge, 3: build_cannon
    # label_index = []
    # for i in range(4):
    #     label_index.append(np.where(label == i)[0])
    #
    # # each label sample some times
    # sample_num = 500
    # label_index_sample = np.array([])
    # for i in range(4):
    #     info = np.random.choice(label_index[i], sample_num)
    #     label_index_sample = np.append(label_index_sample, info)
    #
    # # shuffle ten times
    # for i in range(10):
    #     np.random.shuffle(label_index_sample)
    #
    # order_sample = orders[label_index_sample.astype('int'), :]
    # np.save('order_sample.npy', order_sample)

    order_sample = np.load("order_sample.npy")
    map_sample = np.array([])

    map_get_list = [0, 1, 5, 6]
    for frame_num in order_sample[:, 0]:
        map_temp = np.load(path + "minimap_%d.npy" % int(frame_num))[map_get_list, :].reshape(-1)
        map_sample = np.append(map_sample, map_temp)

    np.save("map_sample.npy", map_sample.reshape((order_sample.shape[0], len(map_get_list)*64*64)))

