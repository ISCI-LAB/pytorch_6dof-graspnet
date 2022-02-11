from functools import partial
import numpy as np
import sys,os 
import glob
sys.path.append('/home/po/TM5/UnseenObjectClustering')
import argparse
import open3d as o3d
from colorama import init, Fore, Back
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm, trange


init(autoreset=True)


def make_data(args):
    npy_path = []
    data_path = args.dataset_dir
    if(args.All_data):
        print(Fore.YELLOW +"making All .npy dataset for testing")
        for i in glob.glob(os.path.join(data_path, '*.npy')):
            npy_path.append(i)

        print(npy_path)
    else:
        print("making partial .npy dataset for testing") 
        for i in args.npy_name:
                npy_path.append(os.path.join(data_path,i+'.npy'))
            
        # print(npy_path)   
    return npy_path

def save_6dof_npy(args,partial):
    # print(args.npy_name)
    np.save('/home/po/TM5/pytorch_6dof-graspnet/demo/data/6dof87',partial)

def show_img(img, bigger=False):
    if bigger:
        plt.figure(figsize=(15,15))
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    plt.show()
def mouse_handler(event, x, y, flags, data):
    if event == cv2.EVENT_LBUTTONDOWN:
        # 標記點位置
        cv2.circle(data['img'], (x,y), 3, (0,0,255), 5, 16) 

        # 改變顯示 window 的內容
        cv2.imshow("Image", data['img'])
        
        # 顯示 (x,y) 並儲存到 list中
        print("get points: (x, y) = ({}, {})".format(x, y))
        data['points'].append((x,y))
def get_points(im):
    # 建立 data dict, img:存放圖片, points:存放點
    data = {}
    data['img'] = im.copy()
    data['points'] = []
    
    # 建立一個 window
    cv2.namedWindow("Image", 0)
    
    # 改變 window 成為適當圖片大小
    h, w, dim = im.shape
    print("Img height, width: ({}, {})".format(h, w))
    cv2.resizeWindow("Image", w, h)
        
    # 顯示圖片在 window 中
    cv2.imshow('Image',im)
    
    # 利用滑鼠回傳值，資料皆保存於 data dict中
    cv2.setMouseCallback("Image", mouse_handler, data)
    
    # 等待按下任意鍵，藉由 OpenCV 內建函數釋放資源
    cv2.waitKey()
    cv2.destroyAllWindows()
    
    # 回傳點 list
    return data['points']
def start_choose(npy_dic):
    for i,item in enumerate(tqdm(npy_dic)):
        # xyz_img = npy_dic['xyz_img']
        img_dst = item['color'].astype(np.uint8)
        label_refined = item['label_refined']
        print("Click on the screen and press any key for end process")
        points  = get_points(img_dst)
        print("points list:")
        print(points)
        if(len(points)>0):
            choose = []
            for j in points:
                choose.append(label_refined[j[1]][j[0]])
            print(choose,"\n\n")
            vis_open3d(npy=item,seg_num=choose[0])
        else:
            raise NameError("Please select at least one point")
def vis_open3d(npy,seg_num):
    ob1 = o3d.geometry.PointCloud()
    seg_npy_dic = preprocess_data(npy,seg_num)
    color = seg_npy_dic['color'][...,::-1]
    ob1.points = o3d.utility.Vector3dVector(seg_npy_dic['xyz'])
    ob1.colors = o3d.utility.Vector3dVector(color/255)
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    if(True):
        cl, ind = ob1.remove_statistical_outlier(nb_neighbors=20,std_ratio=2.0)
        display_inlier_outlier(ob1, ind)
        o3d.visualization.draw_geometries([cl]+[frame])
        partial = np.asarray(cl.points)
        save_6dof_npy(args,partial)
    else:
        o3d.visualization.draw_geometries([ob1]+[frame])
def preprocess_data(npy,seg_num):
    xyz_img = npy['xyz_img']#G
    img_dst = npy['color'].astype(np.uint8)#P
    label_refined = npy['label_refined']#H
    # xyz_img = xyz_img.reshape(-1,3)
    # P.reshape(-1,3)/255
    mask = (label_refined==seg_num)
    seg_npy_dic = {}
    seg_npy_dic['xyz'] = xyz_img[mask]
    seg_npy_dic['color'] = img_dst[mask]
    seg_npy_dic['mask'] = mask
    return seg_npy_dic
def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)
    # 选中的点为灰色，未选中点为红色
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    # 可视化
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud]) 

        

            
            


def main(args):
    npy_arry = make_data(args) 
    npy_load = []   
    for i,item in enumerate(tqdm(npy_arry)):
        tmp = np.load(item,allow_pickle=True,encoding="latin1").item()
        npy_load.append(tmp)
    print(Fore.RED +"success loading num->{} .npy".format(len(npy_arry)))
    start_choose(npy_load)
    
def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npy_name",
                        nargs="*",
                        default=[],
                        help="Import a complete single-view point cloud, using a specific format ")
    parser.add_argument("--use_filter",
                        help="Filter out point cloud noise",
                        default=True)
    parser.add_argument("--root_dir",
                        default=os.path.join(os.path.expanduser("~"), 'TM5'))
    parser.add_argument("--dataset_dir",
                        default=os.path.join(os.path.expanduser("~"),'TM5/UnseenObjectClustering/npy_dataset'))
    parser.add_argument("--label_num",
                        help="This value represents the label_num to be segmented",
                        default='1')
    parser.add_argument("--All_data",
                        help="This value represents the label_num to be segmented",
                        default=True)
    parser.add_argument("--display",
                        help="",
                        default=True)
    return parser


if __name__ == "__main__":
    
    parser = make_parser()
    args = parser.parse_args()
    main(args)
    # test_6dof(args)