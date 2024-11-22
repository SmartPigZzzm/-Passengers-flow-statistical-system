import os
import sys
import time
from datetime import datetime

import numpy as np

sys.path.insert(0, './yolov5')

from yolov5.models.experimental import attempt_load          # 这个全重文件用于加载yolov5的权重和配置文件
from yolov5.utils.datasets import LoadImages, LoadStreams    # LoadImages和LoadStreams分别用来加载图像和视频流
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords, check_imshow     # 检查图像大小、执行非极大值抑制、坐标缩放以及检查是否支持图像显示
from yolov5.utils.torch_utils import select_device, time_synchronized  # 选择合理的设备、用于测量执行过程中的时间
from deep_sort_pytorch.utils.parser import get_config       # get_config类用于解析Deep_sort的配置
from deep_sort_pytorch.deep_sort import DeepSort           # DeepSort类代表deepsort跟踪器
import argparse        # 解析命令行参数和选项，通常用于命令行接受用户输入
import cv2
import torch
import torch.backends.cudnn as cudnn  # 导入pytorch中cudnn模块，用于提高GPU在深度学习中的计算性能
# /home/nvidia/videos/9.4/f/6/f_24_09_04_10_22_13.avi
# /home/nvidia/yolov5_deep_new/f_24_08_22_081048.avi
# /home/nvidia/videos/8.31/f/14/f_24_08_31_17_37_33.avi
# /home/nvidia/videos/9.1/f/3/f_24_09_01_07_47_34.avi
source_dir = ('/home/nvidia/yolov5_deep_new/0910/f_24_09_10_18_47_30.avi')
output_dir = 'inference/output'    # 要保存到的文件夹
show_video = True   # 运行时是否显示
class_list = [0]
line_1 = [400, 100, 1400, 100]
line_4 = [400, 700, 1400, 700]
line_2 = [400, 100, 1400, 100]
line_3 = [400, 700, 1400, 700]
line_y0 = 120
line_y1 = 170
line_y2 = 220
line_y3 = 270
line_y4 = 320
line_y5 = 370
line_y6 = 420
line_y7 = 470
line_y8 = 520
line_y9 = 570

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)


def xyxy_to_xywh(x1, y1, x2, y2):
    """Calculates the relative bounding box from absolute pixel values."""
    bbox_left, bbox_top, bbox_w, bbox_h = min(x1, x2), min(y1, y2), abs(x1 - x2), abs(y1 - y2)
    x_c, y_c, w, h = bbox_left + bbox_w / 2, bbox_top + bbox_h / 2, bbox_w, bbox_h
    return x_c, y_c, w, h


def xyxy_to_tlwh(bbox_xyxy):
    return [[int(x1), int(y1), int(x2 - x1), int(y2 - y1)] for x1, y1, x2, y2 in bbox_xyxy]


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


# 可视化边界框和标签
def draw_boxes(img, bbox, identities=None, offset=(0, 0)):
    for i, box in enumerate(bbox):    # 遍历输入框的边界列表
        x1, y1, x2, y2 = [int(i) for i in box]     # 转变为整数类型
        x1, x2, y1, y2 = x1 + offset[0], x2 + offset[0], y1 + offset[1], y2 + offset[1]
        # Check if identities are provided and within range
        if identities is not None and 0 <= i < len(identities):    # 检查是否提供了标识列表
            id = int(identities[i])
        else:
            id = 0  # Use a default identity
        color = compute_color_for_labels(id)
        label = '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(img, label, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return img


def save_img(imgs, output, station_number, j):
    for i, img in enumerate(imgs):
        if not os.path.exists(output):
            os.makedirs(output)
        cv2.imwrite(filename=f'{output}/{station_number}_{j}.jpg', img=img)
        print("img has been saved!")


def crop_detection_boxes(img, boxes, point, i):
    crop_images = []
    for box in boxes:
        x1, y1, x2, y2, id = box
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        if not i:
            fix_x1 = center_x - 190
            fix_x2 = center_x + 190
            fix_y1 = center_y - 150
            fix_y2 = center_y + 230
        else:
            fix_x1 = center_x - 150
            fix_x2 = center_x + 130
            fix_y1 = center_y - 120
            fix_y2 = center_y + 130
        if id == point:
            crop_image = img[int(fix_y1):int(fix_y2), int(fix_x1):int(fix_x2)]
            crop_images.append(crop_image)
        else:
            pass
    return crop_images


def detect(opt):
    out, source, yolo_weights, deep_sort_weights, show_vid, imgsz = \
        opt.output, opt.source, opt.yolo_weights, opt.deep_sort_weights, opt.show_vid, opt.img_size
    webcam = source == '' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

#####################################################
    # 参数设置
    show_vid = show_video

#####################################################
    # 获取视频的信息
    print(f'进来的视频是{source}')
    source_str = str(source)
    source_str_end = source_str.split('/')[-1]
    source_str_first = source_str_end.split('_')[0]
    up_sign_point = {}
    down_sign_point = {}
#####################################################
    total_in = 0
    total_out = 0
    last_frame_point_out = []
    last_frame_point_in = []
    has_pase_point = []
    processed_ids = set()
    processed_ids_out = set()
#####################################################
# 这部分代码主要是用来加载配置文件，
    # initialize deepsort
    cfg = get_config()  # 获取配置信息
    cfg.merge_from_file(opt.config_deepsort)
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)
    # max_dist 表示在深度排序过程中，两个行人之间的最大欧氏距离，如果两个行人之间的距离超过此阈值，则他们将被认为是不同的身份
    # min_confidence表示用于检测行人的检测器的最小置信度阈值
    # nms_max_overlab表示在进行非极大值抑制时，重叠的阈值，如果两个边界框的重叠度超过此阈值，则较低置信度的框将被抑制
    # max_iou_distance表示在关联过程中，两个边界框之间的最大IOU（交并比）阈值。如果两个边界框的IOU超过此阈值，则他们将被认为是相同的目标
    # max_age表示一个目标的最大允许未更新帧数。如果一个目标在连续的帧中没有更新，则它被移除
    # n_init表示在新目标跟踪器中包含的帧数，用于提高新目标的信任度
    # nn_budget表示deepsort中近邻（NN）匹配算法的最大样本数量

    # Initialize
    device = select_device(opt.device)
    ##################################
    # Load model
    model = attempt_load(yolo_weights, map_location=device)  # load FP32 model#加载模型权重文件
    stride = int(model.stride.max())  # model stride 获取模型步长
    imgsz = check_img_size(imgsz, s=stride)  # check img_size   检查图像尺寸

    if show_vid:
        show_vid = check_imshow()

    if webcam:
        cudnn.benchmark = True  # 设置为True以加速常量图像尺寸的推断
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names     # 获取类别名称

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))
        # 运行一次
    # t0 = time.time()

    for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
        img = torch.from_numpy(img).to(device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        # t1 = time_synchronized() # 记录当前时间，此函数用于获取时间的时间戳
        pred = model(img, augment=opt.augment)[0]   #模型推断

        # Apply NMS #此段代码主要是进行非极大值抑制，过滤或者合并重复的检测框
        pred = non_max_suppression(
            pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        # t2 = time_synchronized() #这一行代码记录了当前时间，通常用于计算 NMS 操作的时间延迟。

        # Process detections
        for i, det in enumerate(pred):  # detections per image每张图的检测结果
            if webcam:  # batch_size >= 1 如果是批处理
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s.copy()

            s += '%gx%g ' % img.shape[2:]
            # print string  #表示当前尺寸的值，表示获取的宽和高

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class每类的检测数目
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string加入字符串

                xywh_bboxs = []
                confs = []

                # Adapt detections to deep sort input format
                for *xyxy, conf, cls in det:
                    # to deep sort format   转为deepsort格式
                    x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*xyxy)
                    xywh_obj = [x_c, y_c, bbox_w, bbox_h]
                    xywh_bboxs.append(xywh_obj)
                    confs.append([conf.item()])

                xywhs = torch.Tensor(xywh_bboxs)
                confss = torch.Tensor(confs)

                # pass detections to deepsort

                outputs = deepsort.update(xywhs, confss, im0)

                # 初始化轨迹字典
                if 'trajectories' not in locals():
                    trajectories = {}
                if 'up_ids' not in locals():
                    up_ids = set()  # 用于存储上车已处理的轨迹 ID
                if 'down_ids' not in locals():
                    down_ids = set()  # 用于下车存储已处理的轨迹 ID
                if 'down_img_ids' not in locals():
                    down_img_ids = set()     # 用于存储下车已截图的轨迹 ID
                if 'up_img_ids' not in locals():
                    up_img_ids = set()     # 用于存储上车已截图的轨迹 ID

                valid_outputs = []

                # 绘制轨迹的起始点和终止点
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, : 4]
                    identities = outputs[:, -1]
                    right_bottom_x = bbox_xyxy[:, 2]
                    right_bottom_y = bbox_xyxy[:, 3]
                    left_top_y = bbox_xyxy[:, 1]
                    left_top_x = bbox_xyxy[:, 0]
                    center_point_x = left_top_x + (right_bottom_x - left_top_x) / 2
                    center_point_y = left_top_y + (right_bottom_y - left_top_y) / 2
                    if source_str_first == 'f':
                        valid_indices = (400 < center_point_x) & (center_point_x < 1400) & (100 < center_point_y) & (center_point_y < 700)
                        filtered_bbox_xyxy = bbox_xyxy[valid_indices]
                        filtered_identities = identities[valid_indices]
                        draw_boxes(im0, filtered_bbox_xyxy, filtered_identities)
                        outputs = outputs[valid_indices]
                    elif source_str_first == 'r':
                        valid_indices = (400 < center_point_x) & (center_point_x < 1400) & (100 < center_point_y) & (center_point_y < 700)
                        filtered_bbox_xyxy = bbox_xyxy[valid_indices]
                        filtered_identities = identities[valid_indices]
                        draw_boxes(im0, filtered_bbox_xyxy, filtered_identities)
                        outputs = outputs[valid_indices]
                        # 更新轨迹
                    for output in outputs:
                        track_id = int(output[-1])  # 检测框的身份
                        bbox = output[:4]  # 检测框的坐标
                        # 计算中心点
                        center_point = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)  # 计算中心点

                        if track_id not in trajectories:
                            # 初始化轨迹
                            trajectories[track_id] = [center_point]  # 存储起始点的中心点
                        else:
                            # 更新轨迹，添加当前检测框的中心点
                            trajectories[track_id].append(center_point)  # 存储中心点
                        # 绘制轨迹
                        trajectory_points = trajectories[track_id]
                        if len(trajectory_points) > 1:
                            for i in range(len(trajectory_points) - 1):
                                cv2.line(im0, tuple(map(int, trajectory_points[i])),
                                             tuple(map(int, trajectory_points[i + 1])), (0, 255, 0), 2)
                                start_point = trajectory_points[0]
                                if source_str_first == 'f':
                                    passed_middle_line = any((start_point[1] < line_y <= center_point[1]) or (
                                                start_point[1] > line_y >= center_point[1]) for line_y in
                                                                 [line_y0, line_y1, line_y2, line_y3, line_y4, line_y5, line_y6, line_y7, line_y8, line_y9])
                                    if passed_middle_line:
                                        # 判断方向
                                        if center_point[1] > start_point[1]:
                                            if track_id not in up_ids:
                                                direction = "up"
                                                total_in += 1
                                                if track_id not in up_img_ids:
                                                    up_imgs = crop_detection_boxes(im0s, outputs, track_id, i=True)
                                                    save_img(up_imgs, 'up_file', 3, track_id)
                                                    up_img_ids.add(track_id)
                                                up_ids.add(track_id)
                                                up_sign_point[track_id] = center_point
                                            else:
                                                if track_id in up_img_ids:
                                                    up_imgs = crop_detection_boxes(im0s, outputs, track_id, i=True)
                                                    save_img(up_imgs, 'up_file', 3, track_id)
                                                    up_img_ids.add(track_id)
                                            # print(f'jishu dian {center_point}')
                                        elif center_point[1] < start_point[1] and track_id not in down_ids:
                                            direction = "down"
                                            total_out += 1
                                            if track_id not in down_img_ids:
                                                down_imgs = crop_detection_boxes(im0s, outputs, track_id, i=False)
                                                save_img(down_imgs, 'down_file', 5, track_id)
                                                down_img_ids.add(track_id)
                                            down_ids.add(track_id)
                                            down_sign_point[track_id] = center_point
                                            # print(f'-----------------------{down_sign_point[track_id]}')
                                        
                                elif source_str_first == 'r':
                                    passed_middle_line = any((start_point[1] < line_y <= center_point[1]) or (
                                                start_point[1] > line_y >= center_point[1]) for line_y in
                                                                 [line_y0, line_y1, line_y2, line_y3, line_y4, line_y5, line_y6, line_y7, line_y8, line_y9])
                                    if passed_middle_line:
                                        # 判断方向
                                        if center_point[1] > start_point[1]:
                                            if track_id not in up_ids:
                                                direction = "up"
                                                total_in += 1
                                                if track_id not in up_img_ids:
                                                    up_imgs = crop_detection_boxes(im0s, outputs, track_id, i=True)
                                                    save_img(up_imgs, 'up_file', 3, track_id)
                                                    up_img_ids.add(track_id)
                                                up_ids.add(track_id)
                                                up_sign_point[track_id] = center_point
                                            else:
                                                if track_id in up_img_ids:
                                                    up_imgs = crop_detection_boxes(im0s, outputs, track_id, i=True)
                                                    save_img(up_imgs, 'up_file', 3, track_id)
                                                    up_img_ids.add(track_id)
                                        elif center_point[1] < start_point[1] and track_id not in down_ids:
                                            direction = "down"
                                            total_out += 1
                                            if track_id not in down_img_ids:
                                                down_imgs = crop_detection_boxes(im0s, outputs, track_id, i=False)
                                                save_img(down_imgs, 'down_file', 5, track_id)
                                                down_img_ids.add(track_id)
                                            down_ids.add(track_id)
                                            down_sign_point[track_id] = center_point
                            # del trajectories[track_id]
                    for track_id in list(trajectories.keys()):
                        trajectory_points = trajectories[track_id]
                        # 判断轨迹是否消失
                        x = [int(output[-1]) for output in outputs]
                        # print(f'dangqian id : {x}')
                        if track_id not in x:  # 如果该 ID 不在当前输出中
                            if len(trajectory_points) > 1:
                                end_point = trajectory_points[-1]
                                end_point_y = end_point[1]
                                # print(track_id, end_point)
                                try:
                                    if track_id in up_sign_point.keys() or down_sign_point.keys():
                                        up_tf = up_sign_point.get(track_id)
                                        down_tf = down_sign_point.get(track_id)
                                        if up_tf and not down_tf and end_point_y < up_sign_point[track_id][1] and track_id in up_ids:
                                            total_in -= 1
                                            up_ids.remove(track_id)
                                            print('f -----------------')
                                            up_path = f'up_file/3_{track_id}.jpg'
                                            if track_id in up_img_ids and os.path.exists(up_path):
                                                os.remove(up_path)
                                                print('img is deleted!')
                                        elif down_tf and not up_tf and end_point_y > down_sign_point[track_id][1] and track_id in down_ids:
                                            total_out -= 1
                                            down_ids.remove(track_id)
                                            print('r -----------------')
                                            down_path = f'down_file/5_{track_id}.jpg'
                                            if track_id in down_img_ids and os.path.exists(down_path):
                                                os.remove(down_path)
                                                print('img is deleted!')
                                        elif up_tf and down_tf:
                                            if end_point_y < up_sign_point[track_id][1] and track_id in up_ids:
                                                total_in -= 1
                                                up_ids.remove(track_id)
                                                up_path = f'up_file/3_{track_id}.jpg'
                                                if track_id in up_img_ids and os.path.exists(up_path):
                                                    os.remove(up_path)
                                                    print('img is deleted!')
                                            elif end_point_y > down_sign_point[track_id][1] and track_id in down_ids:
                                                total_out -= 1
                                                down_ids.remove(track_id)
                                                print('++++++++++++++++++++++')
                                                down_path = f'down_file/5_{track_id}.jpg'
                                                if track_id in down_img_ids and os.path.exists(down_path):
                                                    os.remove(down_path)
                                                    print('img is deleted!')
                                except Exception as e:
                                    print("Error during communication:", {str(e)})
                            # del trajectories[track_id]


            cv2.putText(im0, f'up: {total_in}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(im0, f'down: {total_out}', (50, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            if source_str_first == 'f':
                # print('f')
                cv2.rectangle(im0, (line_1[0], line_1[1]), (line_4[2], line_4[3]), (255, 165, 0), 3)
            elif source_str_first == 'r':
                # print('r')
                cv2.rectangle(im0, (line_2[0], line_2[1]), (line_3[2], line_3[3]), (255, 165, 0), 3)

            if show_vid:
                cv2.namedWindow(p, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(p, 640, 640)
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):
                    break
    print (total_in, total_out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_weights', type=str, default='/home/nvidia/yolov5_deep_new/yolov5/weights/8mbus_4.0.pt', help='model.pt path')
    parser.add_argument('--deep_sort_weights', type=str, default='/home/nvidia/yolov5_deep_new/deep_sort_pytorch/deep_sort/deep/checkpoint/deepsort5.t7', help='ckpt.t7 path')
    # file/folder, 0 for webcam
    parser.add_argument('--source', type=str, default=source_dir, help='source')
    parser.add_argument('--output', type=str, default=output_dir, help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.3, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
    parser.add_argument('--classes', nargs='+', default=class_list, type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort_pytorch/configs/deep_sort.yaml")
    args = parser.parse_args()
    args.img_size = check_img_size(args.img_size)

    with torch.no_grad():
        detect(args)
