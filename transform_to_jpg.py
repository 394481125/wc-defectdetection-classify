import os
import cv2

# 图片的路径
bmp_dir = r'./MyDataset/test'
jpg_dir = r'./MyDataset/test'

filelists = os.listdir(bmp_dir)

for i,file in enumerate(filelists):
    # 读图，-1为不改变图片格式，0为灰度图
    img = cv2.imread(os.path.join(bmp_dir,file),-1)
    newName = file.replace('.bmp','.jpg')
    cv2.imwrite(os.path.join(jpg_dir,newName),img)
    print('第%d张图：%s'%(i+1,newName))