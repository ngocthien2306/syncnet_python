import time
import numpy as np
import cv2
import torch
from torchvision import transforms
from .nets import S3FDNet
from .box_utils import nms_

PATH_WEIGHT = './detectors/s3fd/weights/sfd_face.pth'
img_mean = np.array([104., 117., 123.])[:, np.newaxis, np.newaxis].astype('float32')


class S3FD():

    def __init__(self, device='cuda'):

        tstamp = time.time()
        self.device = device

        print('[S3FD] loading with', self.device)
        self.net = S3FDNet(device=self.device).to(self.device)
        state_dict = torch.load(PATH_WEIGHT, map_location=self.device)
        self.net.load_state_dict(state_dict)
        self.net.eval()
        print('[S3FD] finished loading (%.4f sec)' % (time.time() - tstamp))
    
    def detect_faces(self, image, conf_th=0.8, scales=[1]):

        w, h = image.shape[1], image.shape[0]

        bbox_list = []

        with torch.no_grad():
            for s in scales:
                scaled_img = cv2.resize(image, dsize=(0, 0), fx=s, fy=s, interpolation=cv2.INTER_LINEAR)

                scaled_img = np.swapaxes(scaled_img, 1, 2)
                scaled_img = np.swapaxes(scaled_img, 1, 0)
                scaled_img = scaled_img[[2, 1, 0], :, :]
                scaled_img = scaled_img.astype('float32')
                scaled_img -= img_mean
                scaled_img = scaled_img[[2, 1, 0], :, :]
                x = torch.from_numpy(scaled_img).unsqueeze(0).to(self.device)
                y = self.net(x)

                detections = y.data
                scale = torch.Tensor([w, h, w, h])

                for i in range(detections.size(1)):
                    j = 0
                    while detections[0, i, j, 0] > conf_th:
                        score = detections[0, i, j, 0]
                        pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
                        bbox_list.append((pt[0], pt[1], pt[2], pt[3], float(score)))
                        j += 1

        bboxes = np.array(bbox_list, dtype=np.float32) if bbox_list else np.empty(shape=(0, 5))
        keep = nms_(bboxes, 0.1)
        bboxes = bboxes[keep]

        return bboxes

    def detect_faces_batch(self, images, conf_th=0.8, scales=[1]):
        """Detect faces in a batch of same-sized images.
        Returns a list of bbox arrays (one per image), each shape (N, 5)."""
        if not images:
            return []

        w, h = images[0].shape[1], images[0].shape[0]
        scale_tensor = torch.Tensor([w, h, w, h])
        all_results = [[] for _ in range(len(images))]

        with torch.no_grad():
            for s in scales:
                tensors = []
                for image in images:
                    t = cv2.resize(image, dsize=(0, 0), fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
                    t = np.swapaxes(t, 1, 2)
                    t = np.swapaxes(t, 1, 0)
                    t = t[[2, 1, 0], :, :].astype('float32')
                    t -= img_mean
                    t = t[[2, 1, 0], :, :]
                    tensors.append(torch.from_numpy(t))

                x = torch.stack(tensors, 0).to(self.device)  # (B, C, H, W)
                y = self.net(x)                               # (B, num_cls, top_k, 5)

                detections = y.data
                for b in range(len(images)):
                    for i in range(detections.size(1)):
                        j = 0
                        while detections[b, i, j, 0] > conf_th:
                            score = detections[b, i, j, 0]
                            pt = (detections[b, i, j, 1:] * scale_tensor).cpu().numpy()
                            all_results[b].append((pt[0], pt[1], pt[2], pt[3], float(score)))
                            j += 1

        results = []
        for bbox_list in all_results:
            bboxes = np.array(bbox_list, dtype=np.float32) if bbox_list else np.empty(shape=(0, 5))
            keep = nms_(bboxes, 0.1)
            results.append(bboxes[keep])
        return results
