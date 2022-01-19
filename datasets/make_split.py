import os
import numpy as np
import json
import argparse
import os


def init_category_cache(args):
    not_used_category = args.not_used_category
    category_cache = {}
    with open(os.path.join(args.data_root_dir, 'shapenetsem_category.txt'), 'r') as shapenetsem_category:
        category_list = shapenetsem_category.readlines()
    for category in category_list:
        category = category.strip('\n')
        if not os.path.exists(os.path.join(args.data_root_dir, 'meshes', category)):
            os.mkdir(os.path.join(args.data_root_dir, 'meshes', category))
        if category in not_used_category:
            continue
        category_cache[category] = []

    return category_cache


def read_h5_file(args):
    category_cache = init_category_cache(args)
    h5_file_list = os.listdir(os.path.join(args.data_root_dir, 'grasps/'))
    num_object = 0
    for file_name in h5_file_list:
        if file_name.find('.h5') < 0:
            continue
        file_name = file_name.strip('\n')
        category = file_name.split('_')[0]
        mesh_name = file_name.split('_')[1] + '.obj'

        try:
            os.replace(os.path.join(args.data_root_dir, 'meshes', mesh_name), os.path.join(args.data_root_dir, 'meshes', category, mesh_name))
        except FileNotFoundError:
            pass

        try:
            category_cache[category].append(file_name)
            num_object += 1
        except KeyError:
            # print(f'skip category {category}')
            continue
    print(f"Random shuffle data from {num_object} object")

    return category_cache


def make_json_split(args):
    num_test_per_category = args.num_test_per_category
    num_test = 0
    category_cache = read_h5_file(args)
    for key in category_cache.keys():
        grasp_dict = {"test": [], "train": []}
        num_object = len(category_cache[key])

        # print(f'Number of data {key}: {num_object}')
        object_list = np.asarray(category_cache[key])
        random_index = np.random.choice(range(num_object), num_object, replace=False)

        if num_object < num_test_per_category*10:
            continue

        grasp_dict["test"] = list(object_list[random_index[:num_test_per_category]])
        grasp_dict["train"] = list(object_list[random_index[num_test_per_category:]])
        num_test += num_test_per_category
        jsonString = json.loads(str(grasp_dict).replace("'", '"'))
        json_file = open(os.path.join(args.data_root_dir, 'splits/') + key + '.json', "w")
        json.dump(jsonString, json_file)
        json_file.close()

    print(f'Number of test object: {num_test}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--not_used_category",
                        nargs="+",
                        default=[],
                        help="object category that not be involve in training dataset")
    parser.add_argument("--num_test_per_category",
                        help="number of test data that seperate from each category",
                        default=2)
    parser.add_argument("--data_root_dir",
                        default=os.path.join(os.path.expanduser("~"), 'pytorch_6dof-graspnet/datasets/'))

    args = parser.parse_args()

    make_json_split(args)
