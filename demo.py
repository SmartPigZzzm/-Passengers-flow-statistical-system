# 正常检测跟踪
import sys
import os
import shutil
import multiprocessing
import serial as ser
import can
import threading
import time
import subprocess
import psutil
import logging
import argparse
import cv2
import torch
import numpy as np
import torch.backends.cudnn as cudnn

from PIL import Image, ImageDraw, ImageFont
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
        self.total_num_old = 0
        self.stop_event = multiprocessing.Event()
        self.my_queue = multiprocessing.Queue()
        self.source_dir0 = '0.avi'
        self.source_dir2 = '2.avi'
        self.show_video = True
        self.class_list = [0]
        self.point_idx_0 = 2
        self.point_idx_1 = 0
        self.line_1 = [0, 290, 1920, 290]
        self.line_2 = [0, 440, 1920, 440]
        self.line_3 = [0, 340, 1920, 340]
        self.line_4 = [0, 490, 1920, 490]
        self.x_i, self.y_i = self.get_point_indices(self.point_idx_0)
        self.x_l, self.y_l = self.get_point_indices(self.point_idx_1)
        self.total_num_old = 0

    def get_point_indices(self, point_idx):
        if point_idx == 0:
            return 0, 1
        elif point_idx == 1:
            return 2, 1
        elif point_idx == 2:
            return 2, 3
        elif point_idx == 3:
            return 0, 3
        else:
            raise ValueError("Invalid point index")

    def record_videos(self, video_source, output_path):
        cap = cv2.VideoCapture(video_source)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        cap.set(3, 1920)
        cap.set(4, 1080)
        fps = 25.0
        recording_duration = 180
        start_time = time.time()
        speed = 2
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MJPG'), fps, (1920, 1080))
        while time.time() - start_time < recording_duration:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            cv2.namedWindow(f'{video_source}', cv2.WINDOW_NORMAL)
            cv2.resizeWindow(f'{video_source}', 640, 640)
            cv2.imshow(f'{video_source}', frame)
            delay = int((1000 / fps) / speed)
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    def point_position(self, line, x, y, condition):
        x1, y1, x2, y2 = line
        if y1 == y2:
            return condition(y, y1)
        elif x1 == x2:
            return condition(x, x1)
        else:
            return condition((x - x1) / (x2 - x1), (y - y1) / (y2 - y1))

    def judge_size(self, direction, line, x, y):
        return self.point_position(line, x, y, lambda a, b: a < b) if direction == 0 else self.point_position(line, x, y, lambda a, b: a > b)

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

    def start_detection(self, source_dir):
        parser = argparse.ArgumentParser()
        parser.add_argument('--yolo_weights', type=str, default='/home/nvidia/yolov5_deep/yolov5/weights/latest.pt', help='model.pt path')
        parser.add_argument('--deep_sort_weights', type=str, default='/home/nvidia/yolov5_deep/deep_sort_pytorch/deep_sort/deep/checkpoint/deepsort3.t7', help='ckpt.t7 path')
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

        cam = cv2.VideoCapture(source)
        cam.release()

        total_in = total_out = 0
        last_frame_point_out = []
        last_frame_point_in = []
        has_pase_point = []
        processed_ids = set()
        processed_ids_out = set()

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
                    if len(outputs) > 0:
                        bbox_xyxy = outputs[:, :4]
                        identities = outputs[:, -1]
                        self.draw_boxes(im0, bbox_xyxy, identities)
                        tlwh_bboxs = self.xyxy_to_tlwh(bbox_xyxy)
                        if not last_frame_point_in:
                            for point in outputs:
                                if self.judge_size(0, self.line_2, point[self.x_i], point[self.y_i]) and self.judge_size(0, self.line_4, point[self.x_i], point[self.y_i]):
                                    last_frame_point_in.append(point[-1])
                        else:
                            for point in outputs:
                                if point[-1] in last_frame_point_in and not self.judge_size(0, self.line_2, point[self.x_i], point[self.y_i]) and not self.judge_size(0, self.line_4, point[self.x_i], point[self.y_i]):
                                    if point[-1] not in processed_ids:
                                        last_frame_point_in.remove(point[-1])
                                        has_pase_point.append(point[-1])
                                        total_in += 1
                                        processed_ids.add(point[-1])
                                elif point[-1] not in last_frame_point_in and self.judge_size(0, self.line_2, point[self.x_i], point[self.y_i]) and self.judge_size(0, self.line_4, point[self.x_i], point[self.y_i]):
                                    last_frame_point_in.append(point[-1])
                            for point_idx in last_frame_point_in:
                                if point_idx not in outputs[:, -1]:
                                    last_frame_point_in.remove(point_idx)
                        if not last_frame_point_out:
                            for point in outputs:
                                if self.judge_size(1, self.line_3, point[self.x_l], point[self.y_l]) and self.judge_size(1, self.line_1, point[self.x_l], point[self.y_l]):
                                    last_frame_point_out.append(point[-1])
                        else:
                            for point in outputs:
                                if point[-1] in last_frame_point_out and not self.judge_size(1, self.line_3, point[self.x_l], point[self.y_l]) and not self.judge_size(1, self.line_1, point[self.x_l], point[self.y_l]):
                                    if point[-1] not in processed_ids_out:
                                        last_frame_point_out.remove(point[-1])
                                        has_pase_point.append(point[-1])
                                        total_out += 1
                                        processed_ids_out.add(point[-1])
                                elif point[-1] not in last_frame_point_out and self.judge_size(1, self.line_3, point[self.x_l], point[self.y_l]) and self.judge_size(1, self.line_1, point[self.x_l], point[self.y_l]):
                                    last_frame_point_out.append(point[-1])
                            for point_idx in last_frame_point_out:
                                if point_idx not in outputs[:, -1]:
                                    last_frame_point_out.remove(point_idx)
                else:
                    deepsort.increment_ages()

                cv2.line(im0, (self.line_1[0], self.line_1[1]), (self.line_1[2], self.line_1[3]), (0, 0, 255), 2)
                cv2.line(im0, (self.line_3[0], self.line_3[1]), (self.line_3[2], self.line_3[3]), (0, 0, 255), 2)
                cv2.line(im0, (self.line_2[0], self.line_2[1]), (self.line_2[2], self.line_2[3]), (255, 0, 0), 2)
                cv2.line(im0, (self.line_4[0], self.line_4[1]), (self.line_4[2], self.line_4[3]), (255, 0, 0), 2)
                cv2.putText(im0, f'up: {total_in}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(im0, f'down: {total_out}', (50, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

                if show_vid:
                    cv2.namedWindow(p, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(p, 640, 640)
                    cv2.imshow(p, im0)
                    if cv2.waitKey(1) == ord('q'):
                        break
        return total_in, total_out

    def parse_frame(self, frame):
        if frame[6] == 2 and frame[4] == 6:
            direction = int(frame[11])
            station_number = int(frame[12])
            return direction, station_number
        else:
            return 0, 0

    def read_data_from_485(self, port, baudrate, timeout):
        try:
            with ser.Serial(port, baudrate, timeout=timeout) as se:
                while True:
                    received_frame = se.read(300)
                    if received_frame:
                        value_list = [int(byte) for byte in received_frame]
                        get_datas = []
                        indices = [index for index, value in enumerate(value_list) if value == 165]
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
                                if second_value == 90 and start_index + length_int - 1 <= frame_length:
                                    get_datas.append(value_list[start_index: start_index + length_int])

                        for frame in get_datas:
                            direction, station_number = self.parse_frame(frame)
                            if direction is not None and station_number is not None:
                                return direction, station_number

                    else:
                        return 0, 0
        except Exception as e:
            print("Error during communication:", str(e))
            return 0, 0

    def send_data(self, total_in, total_out, total_num, o_h, o_m, o_s, c_h, c_m, c_s):
        direction, station_number = self.read_data_from_485('/dev/ttyTHS0', 9600, 1)
        data0 = station_number + ((direction & 0b00000001) << 7)
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
                time.sleep(0.1)  # 添加延时防止总线拥堵
            except can.CanError as e:
                print(f"Failed to send message: {e}")

    def get_can_time(self,bus, my_queue, stop_event):
        while not stop_event.is_set():
            message = bus.recv()
            if message.arbitration_id == 0x18FF0F0F:
                hour = int(message.data[3])
                minute = int(message.data[4])
                second = int(message.data[5])
                # print(hour, minute, second)
                while not my_queue.empty():
                    my_queue.get_nowait()
                my_queue.put((hour, minute, second))
                print('get time !')

    def receive_can_messages(self, bus, f, r, my_queue, stop_event):
        front_record = rear_record = None
        previous_result_0 = previous_result_1 = 0
        ro_h = ro_m = ro_s = rc_h = rc_m = rc_s = fo_h = fo_m = fo_s = fc_h = fc_m = fc_s = 0
        while not stop_event.is_set():
            try:
                message = bus.recv()
                if message.arbitration_id == 0x180AEF28:
                    result_1 = (message.data[0] & 0b00110000) >> 4
                    result_0 = (message.data[0] & 0b00001100) >> 2
                    print(f"front_door_sign:{result_0}")
                    print(f"rear_door_sign:{result_1}")
                    if result_1 == 1 and result_1 != previous_result_1:
                        ro_h, ro_m, ro_s = my_queue.get()
                        rear_record = multiprocessing.Process(target=self.record_videos, args=(2, '2.avi'))
                        rear_record.start()
                        previous_result_1 = result_1
                    elif result_1 == 0 and result_1 != previous_result_1:
                        try:
                            rear_record.terminate()
                        except Exception as e:
                            logging.error(f"Error in terminate: {e}")
                        r = False
                        previous_result_1 = result_1
                        rc_h, rc_m, rc_s = my_queue.get()
                    else:
                        pass

                    if result_0 == 1 and result_0 != previous_result_0:
                        fo_h, fo_m, fo_s = my_queue.get()
                        front_record = multiprocessing.Process(target=self.record_videos, args=(0, '0.avi'))
                        front_record.start()
                        previous_result_0 = result_0
                    elif result_0 == 0 and result_0 != previous_result_0:
                        try:
                            front_record.terminate()
                        except Exception as e:
                            logging.error(f"Error in terminate: {e}")
                        previous_result_0 = result_0
                        f = False
                        fc_h, fc_m, fc_s = my_queue.get()

                    if not f and not r:
                        f = r = True
                        detection = multiprocessing.Process(target=self.handle_detection, args=(bus, self.source_dir0, self.source_dir2, ro_h, ro_m, ro_s, rc_h, rc_m, rc_s, fo_h, fo_m, fo_s, fc_h, fc_m, fc_s))
                        detection.start()
                    else:
                        pass
            except Exception as e:
                logging.error(f"Error in receive_can_messages: {e}")

    def handle_detection(self, bus, source_dir0, source_dir2, ro_h, ro_m, ro_s, rc_h, rc_m, rc_s, fo_h, fo_m, fo_s, fc_h, fc_m, fc_s):
        try:
            time.sleep(0.3)
            total_0_in, total_0_out = self.start_detection(source_dir0)
            total_1_in, total_1_out = self.start_detection(source_dir2)
            nums = total_0_in + total_1_in - (total_0_out + total_1_out)
            print(f"front: {total_0_in, total_0_out}")
            print(f"rear:  {total_1_in, total_1_out}")
            total_num_new = self.total_num_old + nums
            if total_num_new <= 0:
                total_num_new = 0
            send_thread = threading.Thread(target=self.send_can_messages, args=(bus, total_0_in, total_0_out, abs(total_num_new), fo_h, fo_m, fo_s, fc_h, fc_m, fc_s,
                                                                                total_1_in, total_1_out, ro_h, ro_m, ro_s, rc_h, rc_m, rc_s), daemon=True)
            send_thread.start()
            send_thread.join()
            self.total_num_old = total_num_new
        except Exception as e:
            logging.error(f"Error in handle_detection: {e}")

    def main(self):
        can_interface = "socketcan"
        channel = "can0"
        recv_id = 0x180AEF28
        special_id = 0x18FF0F0F
        bus = can.interface.Bus(channel=channel, bustype=can_interface)
        can_filter = [{"can_id": recv_id, "can_mask": 0x1FFFFFFF, "extended": True}, {"can_id": special_id, "can_mask": 0x1FFFFFFF, "extended": True}]
        bus.set_filters(can_filter)
        f = True
        r = True
        receive_thread = threading.Thread(target=self.receive_can_messages, args=(bus, f, r, self.my_queue, self.stop_event), daemon=True)
        time_thread = threading.Thread(target=self.get_can_time, args=(bus, self.my_queue, self.stop_event), daemon=True)
        receive_thread.start()
        time_thread.start()
        try:
            receive_thread.join()
            time_thread.join()
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
