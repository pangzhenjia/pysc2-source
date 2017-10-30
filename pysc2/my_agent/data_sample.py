import numpy as np
from pysc2.lib import point
from pysc2.lib import transform

if __name__ == "__main__":

    # path = "C:/Users/chensy/Desktop/pysc2 source/data/demo1/"

    orders = np.loadtxt("new_order.txt")
    label = orders[:, 1]

    # action_type: 0 : move, 1 : build_pylon, 2 : build_forge, 3: build_cannon, 4: nothing
    type_num = 5
    label_index = []
    for i in range(type_num):
        label_index.append(np.where(label == i)[0])

    # each label sample some times
    sample_num = 500
    label_index_sample = np.array([])
    for i in range(type_num):
        info = np.random.choice(label_index[i], sample_num)
        label_index_sample = np.append(label_index_sample, info)

    # shuffle ten times
    for i in range(10):
        np.random.shuffle(label_index_sample)

    order_sample = orders[label_index_sample.astype('int'), :]
    np.save('new_order_sample.npy', order_sample)

    # class map_size(object):
    #     x = 88
    #     y = 96
    #
    # class minimap_resolution(object):
    #     x = 64
    #     y = 64
    #
    # map_size_point = point.Point.build(map_size)
    # feature_layer_minimap_point = point.Point.build(minimap_resolution)
    #
    # world_to_minimap = transform.Linear(point.Point(1, -1), point.Point(0, map_size_point.y))
    # minimap_to_fl_minimap = transform.Linear(feature_layer_minimap_point / map_size_point)
    # world_to_fl_minimap = transform.Chain(
    #     world_to_minimap,
    #     minimap_to_fl_minimap,
    #     transform.Floor()
    # )
    #
    # class temp(object):
    #     x = 0
    #     y = 0
    #
    # name = "order.npy"
    # order_sample = np.load(name)
    # pos = order_sample[:, 2:]
    # pos_new = np.zeros((pos.shape[0], 2))
    #
    # for i in range(pos.shape[0]):
    #     temp.x = pos[i, 0]
    #     temp.y = pos[i, 1]
    #
    #     new = world_to_fl_minimap.fwd_pt(point.Point.build(temp))
    #     pos_new[i, 0] = new.x
    #     pos_new[i, 1] = new.y
    #
    # order_sample[:, 2:] = pos_new
    #
    # np.save(name, order_sample)

    # data = np.load("order.npy")
    # new_data = data.copy()
    #
    # for i in range(data.shape[0]-1):
    #     if data[i, 1] != 0:  # not move
    #         if np.sum(data[i, 1:] - data[i+1, 1:]) == 0:
    #             new_data[i+1, 1:] = [4, 0, 0]
    #
    # np.savetxt("new_order.txt", new_data)


