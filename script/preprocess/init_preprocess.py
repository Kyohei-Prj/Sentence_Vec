import glob
import sys


def remove_header(filepath, head_index=2):

    with open(filepath) as fn:
        return fn.readlines()[head_index:]


def save_file(save_path, text):

    with open(save_path, mode='w') as fn:
        fn.writelines(text)


def init_preprocess(folder_path, save_path):

    filename_list = [fn for fn in glob.glob(folder_path)]

    for i, filename in enumerate(filename_list):
        text = remove_header(filename)
        save_to = save_path + 'text_{}.txt'.format(i)
        save_file(save_to, text)


def main():

    folder_path = sys.argv[1]
    save_path = sys.argv[2]

    print('start init_preprocess')
    init_preprocess(folder_path, save_path)
    print('end init_preprocess')


if __name__ == '__main__':
    main()
