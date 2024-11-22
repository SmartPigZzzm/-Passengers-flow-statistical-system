# 保存录制视频
import json
import sys
import os
import numpy as np
import shutil
import multiprocessing
import serial as ser
import can
import ctypes
import threading
import time
import logging
import argparse
import cv2
import torch
import torch.backends.cudnn as cudnn
from multiprocessing import Process, Value, Array, sharedctypes
sys.path.insert(0, './yolov5')
from yolov5.models.experimental import attempt_load
from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords, check_imshow
from yolov5.utils.torch_utils import select_device, time_synchronized
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort

class DetectionSystem:
    def __init__(self):
        self.global_direction = 0
        self.global_station_number = 0
        self.stop_event = multiprocessing.Event()
        self.front_arr = sharedctypes.RawArray(ctypes.c_int8, 1920*1080*3)
        self.rear_arr = sharedctypes.RawArray(ctypes.c_int8, 1920*1080*3)
        self.isGrab = Value('i', True)
        self.show_video = True
        self.class_list = [0]
        self.line_1 = [500, 150, 1250, 150]
        self.line_4 = [500, 420, 1250, 420]
        self.line_2 = [500, 120, 1250, 120]
        self.line_3 = [500, 400, 1250, 400]
        self.year = self.month = self.day = self.hour = self.minute = self.second = 0
        self.ret0 = self.frame0 = self.ret2 = self.frame2 = None
        self.direction = self.station_number = 0
        self.front_videos = "/home/nvidia/front_videos"
        self.rear_videos = "/home/nvidia/rear_videos"
        self.undetected_front_videos = "/home/nvidia/undetected_front_videos"
        self.undetected_rear_videos = "/home/nvidia/undetected_rear_videos"
        self.root_path = "/home/nvidia/yolov5_deep"
        self.line_y0 = 120
        self.line_y1 = 170
        self.line_y2 = 220
        self.line_y3 = 270
        self.line_y4 = 320
        self.line_y5 = 370
        self.line_y6 = 420
        self.line_y7 = 470
        self.line_y8 = 520
        self.line_y9 = 570
        self.process = {}

    def record_and_rename_video(self, destination_folder, file):
        ori_path = os.path.join(self.root_path, file)
        new_path = os.path.join(destination_folder, file)
        if not os.path.exists(new_path):
            shutil.move(ori_path, new_path)

    def start_cam(self, i, j, front_arr, rear_arr, isGrab):
        cap0 = cv2.VideoCapture(i)
        cap0.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        cap0.set(3, 1920)
        cap0.set(4, 1080)
        cap2 = cv2.VideoCapture(j)
        cap2.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        cap2.set(3, 1920)
        cap2.set(4, 1080)
        while isGrab.value:
            ret0, frame0 = cap0.read()
            ret2, frame2 = cap2.read()
            if ret0 and ret2:
                frame0_f = frame0.flatten(order='C')
                frame2_f = frame2.flatten(order='C')
                front_frame = np.frombuffer(self.front_arr, dtype=np.uint8)
                rear_frame = np.frombuffer(self.rear_arr, dtype=np.uint8)
                front_frame[:] = frame0_f
                rear_frame[:] = frame2_f
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        cap0.release()
        cap2.release()
        cv2.destroyAllWindows()

    def record_videos(self, which_door, shared_frame, output_path):
        fps = 25.0
        recording_duration = 60
        start_time = time.time()
        speed = 2
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MJPG'), fps, (1920, 1080))
        while time.time() - start_time < recording_duration:
            frame_show = np.frombuffer(shared_frame, dtype=np.uint8).reshape(1080, 1920, 3)
            out.write(frame_show)
            cv2.namedWindow(f'{which_door}', cv2.WINDOW_NORMAL)
            cv2.resizeWindow(f'{which_door}', 640, 640)
            cv2.imshow(f'{which_door}', frame_show)
            delay = int((1000 / fps) / speed)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.isGrab.value = False
                break
        out.release()
        cv2.destroyAllWindows()

    def xyxy_to_xywh(self, x1, y1, x2, y2):
        bbox_left, bbox_top, bbox_w, bbox_h = min(x1, x2), min(y1, y2), abs(x1 - x2), abs(y1 - y2)
        x_c, y_c, w, h = bbox_left + bbox_w / 2, bbox_top + bbox_h / 2, bbox_w, bbox_h
        return x_c, y_c, w, h

    def xyxy_to_tlwh(self, bbox_xyxy):
        return [[int(x1), int(y1), int(x2 - x1), int(y2 - y1)] for x1, y1, x2, y2 in bbox_xyxy]

    def compute_color_for_labels(self, label):
        palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
        color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
        return tuple(color)

    def draw_boxes(self, img, bbox, identities=None, offset=(0, 0)):
        fixed_color = (0, 255, 0)
        for i, box in enumerate(bbox):
            x1, y1, x2, y2 = [int(i) for i in box]
            x1 += offset[0]
            x2 += offset[0]
            y1 += offset[1]
            y2 += offset[1]
            if identities is not None and 0 <= i < len(identities):
                id = int(identities[i])
            else:
                id = 0
            color = self.compute_color_for_labels(id)
            label = f'{id}'
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
            cv2.rectangle(img, (x1, y1), (x2, y2), fixed_color, 3)
            cv2.rectangle(img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
            cv2.putText(img, label, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
        return img

    def save_img(self, imgs, output, station_number, j):
        for i, img in enumerate(imgs):
            if not os.path.exists(output):
                os.makedirs(output)
            cv2.imwrite(filename=f'{output}/{station_number}_{j}.jpg', img=img)
            print("img has been saved!")

    def crop_detection_boxes(self, img, boxes, point, i):
        crop_images = []
        for box in boxes:
            x1, y1, x2, y2, id = box
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            if not i:
                fix_x1 = center_x - 190
                fix_x2 = center_x + 190
                fix_y1 = center_y - 150
                fix_y2 = center_y + 200
            else:
                fix_x1 = center_x - 150
                fix_x2 = center_x + 130
                fix_y1 = center_y - 120
                fix_y2 = center_y + 300
            if id == point:
                crop_image = img[int(fix_y1):int(fix_y2), int(fix_x1):int(fix_x2)]
                crop_images.append(crop_image)
            else:
                pass
        return crop_images

    def start_detection(self, source_dir):
        parser = argparse.ArgumentParser()
        parser.add_argument('--yolo_weights', type=str, default='/home/nvidia/yolov5_deep/yolov5/weights/8mbus_4.0.pt', help='model.pt path')
        parser.add_argument('--deep_sort_weights', type=str, default='/home/nvidia/yolov5_deep/deep_sort_pytorch/deep_sort/deep/checkpoint/deepsort5.t7', help='ckpt.t7 path')
        parser.add_argument('--source', type=str, default=source_dir, help='source')
        parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
        parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.3, help='IOU threshold for NMS')
        parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
        parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
        parser.add_argument('--classes', nargs='+', default=self.class_list, type=int, help='filter by class')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument("--config_deepsort", type=str, default="deep_sort_pytorch/configs/deep_sort.yaml")
        args = parser.parse_args()
        args.img_size = check_img_size(args.img_size)

        with torch.no_grad():
            total_in, total_out = self.detect(args)
        return total_in, total_out

    def detect(self, opt):
        source, yolo_weights, deep_sort_weights, show_vid, imgsz = opt.source, opt.yolo_weights, opt.deep_sort_weights, opt.show_vid, opt.img_size
        webcam = source == '' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')
        show_vid = self.show_video
        source_str = str(source)
        source_str_first = source_str.split('_')[0]
        total_in = total_out = 0
        last_frame_point_out = []
        last_frame_point_in = []
        has_pase_point = []
        processed_ids = set()
        processed_ids_out = set()
        up_sign_point = {}
        down_sign_point = {}
        cfg = get_config()
        cfg.merge_from_file(opt.config_deepsort)
        deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT, max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE, nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE, max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET, use_cuda=True)
        device = select_device(opt.device)
        model = attempt_load(yolo_weights, map_location=device)
        stride = int(model.stride.max())
        imgsz = check_img_size(imgsz, s=stride)

        if show_vid:
            show_vid = check_imshow()

        if webcam:
            cudnn.benchmark = True
            dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        else:
            dataset = LoadImages(source, img_size=imgsz)

        names = model.module.names if hasattr(model, 'module') else model.names

        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

        for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
            img = torch.from_numpy(img).to(device)
            img = img.float()
            img /= 255.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            pred = model(img, augment=opt.augment)[0]
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
            t2 = time_synchronized()

            for i, det in enumerate(pred):
                if webcam:
                    p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
                else:
                    p, s, im0 = path, '', im0s

                s += '%gx%g ' % img.shape[2:]

                if det is not None and len(det):
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()
                        s += '%g %ss, ' % (n, names[int(c)])

                    xywh_bboxs = []
                    confs = []

                    for *xyxy, conf, cls in det:
                        x_c, y_c, bbox_w, bbox_h = self.xyxy_to_xywh(*xyxy)
                        xywh_obj = [x_c, y_c, bbox_w, bbox_h]
                        xywh_bboxs.append(xywh_obj)
                        confs.append([conf.item()])

                    xywhs = torch.Tensor(xywh_bboxs)
                    confss = torch.Tensor(confs)

                    outputs = deepsort.update(xywhs, confss, im0)
                    if 'trajectories' not in locals():
                        trajectories = {}
                    if 'up_ids' not in locals():
                        up_ids = set()  # 用于存储上车已处理的轨迹 ID
                    if 'down_ids' not in locals():
                        down_ids = set()  # 用于下车存储已处理的轨迹 ID
                    if 'down_img_ids' not in locals():
                        down_img_ids = set()  # 用于存储下车已截图的轨迹 ID
                    if 'up_img_ids' not in locals():
                        up_img_ids = set()  # 用于存储上车已截图的轨迹 ID

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
                            valid_indices = (400 < center_point_x) & (center_point_x < 1400) & (
                                        100 < center_point_y) & (center_point_y < 450)
                            filtered_bbox_xyxy = bbox_xyxy[valid_indices]
                            filtered_identities = identities[valid_indices]
                            self.draw_boxes(im0, filtered_bbox_xyxy, filtered_identities)
                            outputs = outputs[valid_indices]
                        elif source_str_first == 'r':
                            valid_indices = (400 < center_point_x) & (center_point_x < 1400) & (
                                        100 < center_point_y) & (center_point_y < 450)
                            filtered_bbox_xyxy = bbox_xyxy[valid_indices]
                            filtered_identities = identities[valid_indices]
                            self.draw_boxes(im0, filtered_bbox_xyxy, filtered_identities)
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
                                                                 [self.line_y0, self.line_y1, self.line_y2, self.line_y3, self.line_y4, self.line_y5, self.line_y6, self.line_y7, self.line_y8, self.line_y9])
                                        if passed_middle_line:
                                            # 判断方向
                                            if center_point[1] > start_point[1] and track_id not in up_ids:
                                                direction = "up"
                                                total_in += 1
                                                up_ids.add(track_id)
                                                up_sign_point[track_id] = center_point
                                                # print(f'jishu dian {center_point}')
                                            elif center_point[1] < start_point[1] and track_id not in down_ids:
                                                direction = "down"
                                                total_out += 1
                                                down_ids.add(track_id)
                                                down_sign_point[track_id] = center_point

                                    elif source_str_first == 'r':
                                        passed_middle_line = any((start_point[1] < line_y <= center_point[1]) or (
                                                start_point[1] > line_y >= center_point[1]) for line_y in
                                                                 [self.line_y0, self.line_y1, self.line_y2, self.line_y3, self.line_y4, self.line_y5, self.line_y6, self.line_y7, self.line_y8, self.line_y9])
                                        if passed_middle_line:
                                            # 判断方向
                                            if center_point[1] > start_point[1] and track_id not in up_ids:
                                                direction = "up"
                                                total_in += 1
                                                up_ids.add(track_id)
                                                up_sign_point[track_id] = center_point
                                            elif center_point[1] < start_point[1] and track_id not in down_ids:
                                                direction = "down"
                                                total_out += 1
                                                down_ids.add(track_id)
                                                down_sign_point[track_id] = center_point

                        for track_id in list(trajectories.keys()):
                            trajectory_points = trajectories[track_id]
                            # 判断轨迹是否消失
                            x = [int(output[-1]) for output in outputs]
                            # print(f'dangqian id : {x}')
                            if track_id not in x:  # 如果该 ID 不在当前输出中
                                if len(trajectory_points) > 1:
                                    end_point = trajectory_points[-1]
                                    end_point_y = end_point[1]
                                    try:
                                        if track_id in up_sign_point.keys() or down_sign_point.keys():
                                            up_tf = up_sign_point.get(track_id)
                                            down_tf = down_sign_point.get(track_id)
                                            if up_tf and not down_tf and end_point_y <= up_sign_point[track_id][1] and track_id in up_ids:
                                                total_in -= 1
                                                up_ids.remove(track_id)
                                            elif down_tf and not up_tf and end_point_y >= down_sign_point[track_id][1] and track_id in down_ids:
                                                total_out -= 1
                                                down_ids.remove(track_id)
                                            elif up_tf and down_tf:
                                                if end_point_y <= up_sign_point[track_id][1] and track_id in up_ids:
                                                    total_in -= 1
                                                    up_ids.remove(track_id)
                                                elif end_point_y >= down_sign_point[track_id][1] and track_id in down_ids:
                                                    total_out -= 1
                                                    down_ids.remove(track_id)
                                    except Exception as e:
                                        print("Error during communication:", {str(e)})
                                # del trajectories[track_id]

                cv2.putText(im0, f'up: {total_in}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(im0, f'down: {total_out}', (50, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                if source_str_first == 'f':
                    cv2.rectangle(im0, (self.line_1[0], self.line_1[1]), (self.line_4[2], self.line_4[3]), (255, 165, 0), 3)
                elif source_str_first == 'r':
                    cv2.rectangle(im0, (self.line_2[0], self.line_2[1]), (self.line_3[2], self.line_3[3]), (255, 165, 0), 3)

                if show_vid:
                    cv2.namedWindow(p, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(p, 640, 640)
                    cv2.imshow(p, im0)
                    if cv2.waitKey(1) == ord('q'):
                        break
        print('\n')
        print(total_in, total_out)
        return total_in, total_out

    def parse_frame(self, frame):
        print("--------------get_message!--------------")
        if frame[6] == 2 and frame[4] == 6:
            direction = int(frame[11])
            station_number = int(frame[12])
            print("--------------get success!--------------")
            return direction, station_number
        else:
            return 0, 0

    def read_data_from_485(self, port, baudrate, timeout):
        try:
            with ser.Serial(port, baudrate, bytesize=ser.EIGHTBITS, stopbits=ser.STOPBITS_ONE, timeout=timeout) as se:
                while True:
                    received_frame = se.read(300)
                    print("--------------wait frame!--------------")
                    if received_frame:
                        value_list = [int.from_bytes(received_frame, byteorder='little')]
                        get_datas = []
                        indices = [index for index, value in enumerate(value_list) if value == 165]
                        print("_____________received!______________")
                        for start_index in indices:
                            frame_length = len(value_list)
                            if start_index + 4 <= frame_length:
                                second_value = value_list[start_index + 1]
                                first_len = value_list[start_index + 2]
                                sec_len = value_list[start_index + 3]
                                len_list = [first_len, sec_len]
                                combined_int = len_list[0] + (len_list[1] << 8)
                                hex_len = hex(combined_int)
                                length_int = int(hex_len, 16)
                                print("--------------get_length!--------------")
                                if second_value == 90 and start_index + length_int - 1 <= frame_length:
                                    get_datas.append(value_list[start_index: start_index + length_int])
                                    print("--------------append success!--------------")

                        for frame in get_datas:
                            self.direction, self.station_number = self.parse_frame(frame)

        except Exception as e:
            print("Error during communication:", str(e))

    def send_data(self, total_in, total_out, total_num, o_h, o_m, o_s, c_h, c_m, c_s):
        data0 = self.station_number + ((self.direction & 0b00000001) << 7)
        data1 = o_h + ((o_m & 0b00000111) << 5)
        data2 = ((o_m & 0b00111000) >> 3) + ((o_s & 0b00011111) << 3)
        data3 = ((total_in & 0b01111111) << 1) + ((o_s & 0b00100000) >> 5)
        data4 = c_h + ((c_m & 0b00000111) << 5)
        data5 = ((c_m & 0b00111000) >> 3) + ((c_s & 0b00011111) << 3)
        data6 = ((total_out & 0b01111111) << 1) + ((c_s & 0b00100000) >> 5)
        data7 = abs(total_num)
        return data0, data1, data2, data3, data4, data5, data6, data7

    def send_can_messages(self, bus, total_0_in, total_0_out, total_num_new, fo_h, fo_m, fo_s, fc_h, fc_m, fc_s,
                          total_1_in, total_1_out, ro_h, ro_m, ro_s, rc_h, rc_m, rc_s):
        data0, data1, data2, data3, data4, data5, data6, data7 = self.send_data(total_0_in, total_0_out, total_num_new,
                                                                                fo_h, fo_m, fo_s, fc_h, fc_m, fc_s)
        data8, data9, data10, data11, data12, data13, data14, data15 = self.send_data(total_1_in, total_1_out, 0,
                                                                                      ro_h, ro_m, ro_s, rc_h, rc_m, rc_s)
        message1 = can.Message(arbitration_id=0x18F72187, data=[data0, data1, data2, data3, data4, data5, data6, data7], is_extended_id=True)
        message2 = can.Message(arbitration_id=0x18F72287, data=[data8, data9, data10, data11, data12, data13, data14, 0], is_extended_id=True)

        for n in range(6):
            try:
                bus.send(message1)
                bus.send(message2)
                print("Sent message!")
                # print(f"----direction station_number:{self.direction, self.station_number}----")
                time.sleep(0.1)  # 添加延时防止总线拥堵
            except can.CanError as e:
                print(f"Failed to send message: {e}")

    def receive_can_messages(self, bus, f, r, z, m, stop_event):
        front_record = rear_record = None
        previous_result_0 = previous_result_1 = 0
        ro_h = ro_m = ro_s = rc_h = rc_m = rc_s = fo_h = fo_m = fo_s = fc_h = fc_m = fc_s = 0
        fo_y = fo_mon = fo_d = ro_y = ro_mon = ro_d = 0
        size_filter = 3000000
        rear_filename = front_filename = None
        detection = None
        while not stop_event.is_set():
            try:
                message = bus.recv()
                if message.arbitration_id == 0x18FF0F0F:
                    self.year = int(message.data[0])
                    self.month = int(message.data[1])
                    self.day = int(message.data[2])
                    self.hour = int(message.data[3])
                    self.minute = int(message.data[4])
                    self.second = int(message.data[5])
                if message.arbitration_id == 0x180AEF28:
                    result_1 = (message.data[0] & 0b00110000) >> 4
                    result_0 = (message.data[0] & 0b00001100) >> 2
                    # print(f"front_door_sign:{result_0}")
                    # print(f"rear_door_sign:{result_1}")
                    if result_1 == 1 and result_1 != previous_result_1:
                        ro_y, ro_mon, ro_d, ro_h, ro_m, ro_s = self.year, self.month, self.day, self.hour, self.minute, self.second
                        if ro_y != 0:
                            print(ro_y, ro_mon, ro_d, ro_h, ro_m, ro_s)
                            rear_filename = f'r_{ro_y:02}_{ro_mon:02}_{ro_d:02}_{ro_h:02}_{ro_m:02}_{ro_s:02}.avi'
                            rear_record = multiprocessing.Process(target=self.record_videos, args=('rear', self.rear_arr, rear_filename))
                            rear_record.start()
                            previous_result_1 = result_1
                            z = True
                    elif result_1 == 0 and result_1 != previous_result_1:
                        try:
                            rear_record.terminate()
                            rear_record.join()
                        except Exception as e:
                            logging.error(f"Error in handle_detection: {e}")
                        finally:
                            rc_h, rc_m, rc_s = self.hour, self.minute, self.second
                            print(rc_h, rc_m, rc_s)
                            r = True
                            previous_result_1 = result_1
                    else:
                        pass

                    if result_0 == 1 and result_0 != previous_result_0:
                        fo_y, fo_mon, fo_d, fo_h, fo_m, fo_s = self.year, self.month, self.day, self.hour, self.minute, self.second
                        if fo_y != 0:
                            print(fo_y, fo_mon, fo_d, fo_h, fo_m, fo_s)
                            front_filename = f'f_{fo_y:02}_{fo_mon:02}_{fo_d:02}_{fo_h:02}_{fo_m:02}_{fo_s:02}.avi'
                            front_record = multiprocessing.Process(target=self.record_videos, args=('front', self.front_arr, front_filename))
                            front_record.start()
                            previous_result_0 = result_0
                            m = True
                    elif result_0 == 0 and result_0 != previous_result_0:
                        try:
                            front_record.terminate()
                            front_record.join()
                        except Exception as e:
                            logging.error(f"Error in handle_detection: {e}")
                        finally:
                            fc_h, fc_m, fc_s = self.hour, self.minute, self.second
                            print(fc_h, fc_m, fc_s)
                            previous_result_0 = result_0
                            f = True
                    else:
                        pass

                    # 处理进程启动
                    if f and r and z and m:
                        if detection is None or not detection.is_alive():
                            detection = multiprocessing.Process(target=self.handle_detection, args=(bus, True, True, front_filename, rear_filename, ro_y, ro_mon, ro_d, ro_h, ro_m, ro_s, rc_h, rc_m, rc_s, fo_y, fo_mon, fo_d, fo_h, fo_m, fo_s, fc_h, fc_m, fc_s))
                            detection.start()
                            f = r = z = m = False
                    elif f and not z:   # only f
                        if os.path.exists(front_filename) and os.path.getsize(front_filename) > size_filter:
                            if detection is None or not detection.is_alive():
                                detection = multiprocessing.Process(target=self.handle_detection, args=(bus, True, False, front_filename, None, fo_y, fo_mon, fo_d, fo_h, fo_m, fo_s, fc_h, fc_m, fc_s, fo_y, fo_mon, fo_d, fo_h, fo_m, fo_s, fc_h, fc_m, fc_s))
                                detection.start()
                                f = r = z = m = False
                    elif r and not m:   # only r
                        if detection is None or not detection.is_alive():
                            detection = multiprocessing.Process(target=self.handle_detection, args=(bus, False, True, None, rear_filename, ro_y, ro_mon, ro_d, ro_h, ro_m, ro_s, rc_h, rc_m, rc_s, ro_y, ro_mon, ro_d, ro_h, ro_m, ro_s, rc_h, rc_m, rc_s))
                            detection.start()
                            f = r = z = m = False
                    else:
                        pass
                else:
                    pass

            except Exception as e:
                logging.error(f"Error in receive_can_messages: {e}")

    def save_result(self, i, line):
        if i == 0:  # 前
            with open('inference/output/front.txt', 'a') as f:
                f.write(line)
        elif i == 1:    # 后
            with open('inference/output/rear.txt', 'a') as f:
                f.write(line)
        elif i == 2:
            with open('inference/output/total_num.txt', 'w') as f:
                f.write(line)

    def read_num(self):
        with open('inference/output/total_num.txt', 'r') as f:
            total_num = f.read().strip()
            return total_num

    def handle_detection(self, bus, l, j, f_video, r_video, ro_y, ro_mon, ro_d, ro_h, ro_m, ro_s, rc_h, rc_m, rc_s, fo_y, fo_mon, fo_d, fo_h, fo_m, fo_s, fc_h, fc_m, fc_s):
        try:
            time.sleep(0.2)
            if l:
                total_0_in, total_0_out = self.start_detection(f_video)
                front_result = f'f_{fo_y}_{fo_mon:02}_{fo_d:02}_{fo_h:02}_{fo_m:02}_{fo_s:02}\t front_up:{total_0_in}\t front_down:{total_0_out}\t\n'
                self.save_result(i=0, line=front_result)
                f_files = os.listdir(self.front_videos)
                if len(f_files) <= 1500:
                    self.record_and_rename_video(self.front_videos, f_video)

            else:
                total_0_in, total_0_out = 0, 0
            if j:
                total_1_in, total_1_out = self.start_detection(r_video)
                real_result = f'r_{ro_y}_{ro_mon:02}_{ro_d:02}_{ro_h:02}_{ro_m:02}_{ro_s:02}\t real_up:{total_1_in}\t real_down:{total_1_out}\t\n'
                self.save_result(i=1, line=real_result)
                r_files = os.listdir(self.rear_videos)
                if len(r_files) <= 1500:
                    self.record_and_rename_video(self.rear_videos, r_video)
            else:
                total_1_in, total_1_out = 0, 0
            total_num = self.read_num()
            total_num = int(total_num)
            nums = total_0_in + total_1_in - (total_0_out + total_1_out)
            total_num_new = total_num + nums
            if total_num_new <= 0:
                total_num_new = 0
            total_num = total_num_new
            self.save_result(2, str(total_num))
            send_thread = threading.Thread(target=self.send_can_messages, args=(bus, total_0_in, total_0_out, abs(total_num_new), fo_h, fo_m, fo_s, fc_h, fc_m, fc_s,
                                                                                total_1_in, total_1_out, ro_h, ro_m, ro_s, rc_h, rc_m, rc_s), daemon=True)
            send_thread.start()
            send_thread.join()
        except Exception as e:
            logging.error(f"Error in handle_detection: {e}")

    def main(self):
        start_fr = multiprocessing.Process(target=self.start_cam, args=(0, 2, self.front_arr, self.rear_arr, self.isGrab))
        start_fr.start()
        can_interface = "socketcan"
        channel = "can0"
        recv_id = 0x180AEF28
        special_id = 0x18FF0F0F
        self.save_result(2, '0')
        bus = can.interface.Bus(channel=channel, bustype=can_interface)
        can_filter = [{"can_id": recv_id, "can_mask": 0x1FFFFFFF, "extended": True},
                      {"can_id": special_id, "can_mask": 0x1FFFFFFF, "extended": True}]
        bus.set_filters(can_filter)
        if not os.path.exists(self.front_videos):
            os.makedirs(self.front_videos)
        if not os.path.exists(self.rear_videos):
            os.makedirs(self.rear_videos)
        if not os.path.exists(self.undetected_front_videos):
            os.makedirs(self.undetected_front_videos)
        if not os.path.exists(self.undetected_rear_videos):
            os.makedirs(self.undetected_rear_videos)
        for filename in os.listdir(self.root_path):
            video_extensions = {'.mp4', '.avi'}
            file_extension = os.path.splitext(filename)[1].lower()
            if file_extension in video_extensions:
                file_path = os.path.join(self.root_path, filename)
                if filename.lower().startswith('f') and os.path.getsize(file_path) > 3000000:
                    self.record_and_rename_video(self.undetected_front_videos, filename)
                elif filename.lower().startswith('r') and os.path.getsize(file_path) > 3000000:
                    self.record_and_rename_video(self.undetected_rear_videos, filename)
                else:
                    if os.path.exists(file_path):
                        os.remove(file_path)

        f = False
        r = False
        z = False
        m = False
        receive_thread = threading.Thread(target=self.receive_can_messages, args=(bus, f, r, z, m, self.stop_event), daemon=True)
        # rs485_thread = threading.Thread(target = self.read_data_from_485, args=('/dev/ttyTHS0', 9600, 1))
        receive_thread.start()
        # rs485_thread.start()
        try:
            receive_thread.join()
            # rs485_thread.join()
        except KeyboardInterrupt:
            self.stop_event.set()
        except Exception as e:
            logging.error(f"Error in main: {e}")
        finally:
            bus.shutdown()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    detection_system = DetectionSystem()
    detection_system.main()
