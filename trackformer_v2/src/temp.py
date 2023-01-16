import numpy as np
import cv2
import random
import argparse

def main(args):
    results = np.load(args.r, allow_pickle=True)
    track_info = dict(results["results"].item())

    global ignore_id
    ignore_id = set()
    color_dict = {}

    def click_event(event, x, y, flags, params):
        global ignore_id
        if event == cv2.EVENT_LBUTTONDOWN:
            for id, bbox in params.items():
                if bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]:
                    if id in ignore_id:
                        ignore_id.remove(id)
                    else:
                        ignore_id.add(id)


    cap = cv2.VideoCapture(args.v)
    frame_id = -1
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Video ending, exiting...")
            break

        frame_id += 1
        current_bboxs = {}

        for i, track in track_info.items():
            if i not in color_dict:
                color_dict[i] = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))
            
            if frame_id in track:
                bbox = track[frame_id]["bbox"]
                current_bboxs[i] = bbox

                if i not in ignore_id:
                    left_top, right_bottom = (bbox[0], bbox[1]), (bbox[2], bbox[3])
                    cv2.rectangle(frame, left_top, right_bottom, color_dict[i], 2)

        cv2.imshow("offline", frame)
        cv2.setMouseCallback("offline", click_event, current_bboxs)
        if cv2.waitKey(300) == ord('q'): break

    
    cap.release()
    cv2.destroyAllWindows()
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser("video tracking")
    parser.add_argument("-v", required=True, type=str, help="video path")
    parser.add_argument("-r", required=True, type=str, help="trackformer result npz")
    parser.add_argument("-b", "--with-bbox", type=bool, default=False, help="show bbox")

    args = parser.parse_args()
    main(args)