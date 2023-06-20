import numpy as np


def generate_pos(txt_path, x_num, y_num, x_v, y_v, ovl_r):
    with open(txt_path, 'w') as f:
        for y in range(y_num):
            for x in range(x_num):
                f.write('%.2d X %.2d;;(%d, %d)\n' % (y, x, x*x_v*(1-ovl_r), y*y_v*(1-ovl_r)))
    f.close()


if __name__ == '__main__':
    generate_pos(r'J:\CML\230523-e38-I\230523_2_16-01-13\pos.txt', 11, 10, 2048, 2048, 0.40)
