import argparse
import time
import pandas as pd
from pypylon import pylon
from pathlib import Path
from datetime import datetime

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np

# from yolov5_1.models.experimental import attempt_load
# from yolov5_1.utils.datasets import LoadStreams, LoadImages
# from yolov5_1.utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
#     strip_optimizer, set_logging, increment_path
# from yolov5_1.utils.plots import plot_one_box
# from yolov5_1.utils.torch_utils import select_device, load_classifier, time_synchronized
try:
    from yolov5_1.models.experimental import attempt_load
    from yolov5_1.utils.datasets import LoadStreams, LoadImages
    from yolov5_1.utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
        strip_optimizer, set_logging, increment_path
    from yolov5_1.utils.plots import plot_one_box
    from yolov5_1.utils.torch_utils import select_device, load_classifier, time_synchronized

except:
    from models.experimental import attempt_load
    from utils.datasets import LoadStreams, LoadImages
    from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
        strip_optimizer, set_logging, increment_path
    from utils.plots import plot_one_box
    from utils.torch_utils import select_device, load_classifier, time_synchronized






class DetectYolo:
    def __init__(self):
        parser = argparse.ArgumentParser()
        # parser.add_argument('--weights', nargs='+', type=str, default=r'F:\Ph.D\contactlens\contact_lens_project\weights\weights/best.pt',
        #                     help='model.pt path(s)')  # yolov5s.pt
        # parser.add_argument('--weights', nargs='+', type=str, default=r'./yolov5l.pt',
        #                     help='model.pt path(s)')  # yolov5s.pt
        parser.add_argument('--weights', nargs='+', type=str, default=r'F:\Ph.D\contactlens\contact_lens_project\weights\exp33_s_mirror_0630-20220630T115612Z-001\exp33_s_mirror_0630\weights/best.pt',
                            help='model.pt path(s)')  # yolov5s.pt
        # parser.add_argument('--weights', nargs='+', type=str,
        #                     default=r'yolov5s.pt',
        #                     help='model.pt path(s)')  # yolov5s.ptA
        parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam  ::own_camera
        parser.add_argument('--img-size', type=int, default=1800, help='inference size (pixels)')
        parser.add_argument('--conf-thres', type=float, default=0.05, help='object confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
        parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--view-img', action='store_true', help='display results')
        parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
        parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
        parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument('--update', action='store_true', help='update all models')
        parser.add_argument('--project', default='runs/detect', help='save results to project/name')
        parser.add_argument('--name', default='exp', help='save results to project/name')
        parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
        parser.add_argument('--collect', default=False, help='to collect data during running program')
        parser.add_argument('--command', default=True, help='to collect data during running program')
        parser.add_argument('--print-result', default=True, help='show result on image ')
        self.opt = parser.parse_args()

        # init model หรือ
        self.source, self.weights, self.view_img, self.save_txt, self.imgsz = self.opt.source, self.opt.weights, self.opt.view_img, self.opt.save_txt, self.opt.img_size
        self.webcam = self.source.isnumeric() or self.source.endswith('.txt') or self.source.lower().startswith(
            ('rtsp://', 'rtmp://', 'http://'))

        # Directories
        self.save_dir = Path(
            increment_path(Path(self.opt.project) / self.opt.name, exist_ok=self.opt.exist_ok))  # increment run
        (self.save_dir / 'labels' if self.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Initialize
        set_logging()
        self.device = select_device(self.opt.device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        self.model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
        self.imgsz = check_img_size(self.imgsz, s=self.model.stride.max())  # check img_size
        if self.half:
            self.model.half()  # to FP16

        # Second-stage classifier
        self.classify = False
        if self.classify:
            self.modelc = load_classifier(name='resnet101', n=2)  # initialize
            self.modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=self.device)['model']).to(self.device).eval()

        # Set Dataloader
        vid_path, vid_writer = None, None
        if self.webcam:
            self.view_img = True
        #     cudnn.benchmark = True  # set True to speed up constant image size inference
        #     self.dataset = LoadStreams(self.source, img_size=self.imgsz)

        elif self.source == 'own_camera':
            pass

        else:
            self.save_img = True
            self.dataset = LoadImages(self.source, img_size=self.imgsz)
        self.save_img = True
        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]

        # Run inference
        self.t0 = time.time()
        img = torch.zeros((1, 3, self.imgsz, self.imgsz), device=self.device)  # init img
        _ = self.model(img.half() if self.half else img) if self.device.type != 'cpu' else None  # run once

    @staticmethod
    def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
        # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return img, ratio, (dw, dh)

    def set_param(self):
        self.img_size = round(self.opt.img_size/32+1)*32

    def detect(self,frame):
        box_info = []
        # work space
        ### check RGB or MONO
        if len(frame.shape) ==2:
            frame = cv2.cvtColor(frame,cv2.COLOR_GRAY2BGR)

        im0 = []
        img0 = frame
        # cv2.imshow("aaaaa",frame)
        im0s = img0

        img, ratio, (dw, dh) = self.letterbox(img0, new_shape= self.img_size)
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        if self.opt.command :
            # print(img.size())
            # print(img.size())
            pred = self.model(img, augment=self.opt.augment)[0]

            # Apply NMS
            pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes,
                                       agnostic=self.opt.agnostic_nms)

            # Apply Classifier
            # if self.classify:
            #     pred = apply_classifier(pred, self.modelc, img, im0s)

            if self.opt.print_result:
                # Process detections
                for i, det in enumerate(pred):  # detections per image
                    p, s, im0 = Path('0'), '', im0s

                    # save_path = str(self.save_dir / p.name)
                    # txt_path = str(save_dir / 'labels' / p.stem) + (
                    #     '_%g' % dataset.frame if dataset.mode == 'video' else '')
                    s += '%gx%g ' % img.shape[2:]  # print string
                    gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                    if det is not None and len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                        # Print results
                        classes = []
                        for c in det[:, -1].unique():
                            n = (det[:, -1] == c).sum()  # detections per class
                            s += '%g %ss, ' % (n, self.names[int(c)])  # add to string
                            s1 = '%g' % (n)
                            # classes.append([self.names[int(c)], s1])

                        # Write results
                        box_info = []
                        for *xyxy, conf, cls in reversed(det):
                            # if self.save_txt:  # Write to file
                                # xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(
                                #     -1).tolist()  # normalized xywh
                                # line = (cls, *xywh, conf) if self.opt.save_conf else (cls, *xywh)  # label format
                                # with open(txt_path + '.txt', 'a') as f:
                                #     f.write(('%g ' * len(line)).rstrip() % line + '\n')

                            if self.save_img or self.view_img:  # Add bbox to image
                                label = '%s %.2f' % (self.names[int(cls)], conf)
                                # plot_one_box(xyxy, img0, label=label, color=self.colors[int(cls)], line_thickness=3)

                                c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                                confident = '%g' % (conf)
                                box_info.append([label, c1, c2, confident])


        elif not self.opt.command:
            pred = self.model(img, augment=self.opt.augment)[0]

            # Apply NMS
            pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes,
                                       agnostic=self.opt.agnostic_nms)
            t2 = time_synchronized()

            # Apply Classifier
            if self.classify:
                pred = apply_classifier(pred, self.modelc, img, im0s)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if self.webcam:  # batch_size >= 1
                    p, s, im0 = Path('0'), '%g: ' % i, im0s[i].copy()
                else:
                    p, s, im0 = Path('0'), '', im0s

                save_path = str(self.save_dir / p.name)
                # txt_path = str(save_dir / 'labels' / p.stem) + (
                #     '_%g' % dataset.frame if dataset.mode == 'video' else '')
                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    classes = []
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += '%g %ss, ' % (n, self.names[int(c)])  # add to string
                        s1 = '%g' % (n)
                        classes.append([self.names[int(c)], s1])

                    # Write results

                    for *xyxy, conf, cls in reversed(det):
                        if self.save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(
                                -1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if self.opt.save_conf else (cls, *xywh)  # label format
                            # with open(txt_path + '.txt', 'a') as f:
                            #     f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if self.save_img or self.view_img:  # Add bbox to image
                            label = '%s %.2f' % (self.names[int(cls)], conf)
                            plot_one_box(xyxy, img0, label=label, color=self.colors[int(cls)], line_thickness=3)

                            c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                            confident = '%g' % (conf)
                            box_info.append([label, c1, c2, confident])

            # Print time (inference + NMS)
            # print('%sDone. (%.3fs)' % (s, t2 - t1))
        # if im0 == []:
        #     im0 = frame.copy()
        return img0, box_info

if __name__ == '__main__':

    detect = DetectYolo()
    img = cv2.imread("../data/new company/burr/FOV1/2.png")
    print(img.shape)
    cv2.namedWindow("asd",cv2.WINDOW_NORMAL)
    cv2.imshow("asd",img)
    cv2.waitKey(0)
    detect.set_param()
    print("detect")
    while True:
        img = cv2.imread("../data/new company/burr/FOV1/2.png")
        cv2.imshow("asd", img)
        cv2.waitKey(0)
        t0 = time.time()
        img, boxes = detect.detect(img)
        print("time per image {}s".format(round(time.time() - t0,3)))
        print(boxes)
        cv2.imshow("asd",img)
        cv2.waitKey(0)

    # while True:
    #     cap = cv2.VideoCapture(0)
    #     ret, img = cap.read()
    #     cv2.namedWindow("asd", cv2.WINDOW_NORMAL)
    #     cv2.imshow("asd", img)
    #     cv2.waitKey(1)
    #     t0 = time.time()
    #     print("detect")
    #     img = detect.detect(img)
    #     print("time per image {}s".format(round(time.time() - t0, 3)))
    #     cv2.imshow("asd", img)
    #     cv2.waitKey(1)




