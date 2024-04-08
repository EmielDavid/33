# 33 (as of December 2023)
import numpy as np
import cv2
import depthai
import blobconverter
import time


# function for the entire pipeline
# shadow_color is BGR
# delay should be 1
# record is fps if necessary otherwise no recording
# record_time is the amount of seconds that the recording should be
# filename in .avi

def create_depthai_pipeline(shadow_color=(255, 255, 255), delay=1, record=0, record_time=None):
    # ask for the name of the file
    if record:
        filename = input("Enter the filename for recording with .avi (e.g., 'output.avi'): ")
    else:
        filename = None
    # pipeline definition
    pipeline = depthai.Pipeline()
    cam_rgb = pipeline.create(depthai.node.ColorCamera)
    cam_rgb.setPreviewSize(300, 300)
    cam_rgb.setInterleaved(False)
    detection_nn = pipeline.create(depthai.node.MobileNetDetectionNetwork)
    detection_nn.setBlobPath(blobconverter.from_zoo(name='mobilenet-ssd', shaves=6))
    detection_nn.setConfidenceThreshold(0.6)
    cam_rgb.preview.link(detection_nn.input)
    xout_rgb = pipeline.create(depthai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")
    cam_rgb.preview.link(xout_rgb.input)
    xout_nn = pipeline.create(depthai.node.XLinkOut)
    xout_nn.setStreamName("nn")
    detection_nn.out.link(xout_nn.input)
    # optional to play with
    lower_color = np.array([0, 0, 0])
    upper_color = np.array([255, 255, 255])

    # method to delete redundant edges

    def aspect_ratio(contour):
        x, y, w, h = cv2.boundingRect(contour)
        return float(w) / h

    # For displaying the output (name of window) (dimensions of window) (feed)
    def output_frame(name, feed, dimensions=[1000, 1000]):
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, dimensions[0], dimensions[1])
        cv2.imshow(name, feed)

    # initialize all the variables if it's recording
    def record_settings(record_time, dimensions=(300, 300)):
        frame_rate = cam_rgb.getFps()
        expected_frames = record_time * frame_rate
        start_time = time.time()
        frame_count = 0
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(filename, fourcc, frame_rate, dimensions)
        return out, frame_rate, frame_count, expected_frames, start_time

    with depthai.Device(pipeline) as device:
        q_rgb = device.getOutputQueue("rgb")
        q_nn = device.getOutputQueue("nn")
        frame = None
        detections = []

        def frame_norm(frame, box):
            norm_vals = np.full(len(box), frame.shape[0])
            norm_vals[::2] = frame.shape[1]
            return (np.clip(np.array(box), 0, 1) * norm_vals).astype(int)

        if record:
            out, frame_rate, frame_count, expected_frames, start_time = record_settings(record_time)
        while True:
            in_rgb = q_rgb.tryGet()
            in_nn = q_nn.tryGet()
            if in_rgb is not None:
                frame = in_rgb.getCvFrame()
            if in_nn is not None:
                detections = in_nn.detections
            if frame is not None:
                frame_silhouette = frame.copy()
                frame_shadow = np.zeros_like(frame)  # Create an empty frame for the shadow
                shadow_mask = np.zeros_like(frame)  # Create a single shadow mask for all contours
                for detection in detections:
                    # bounding box for human
                    bbox = frame_norm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                    x_start, y_start, x_end, y_end = bbox[0], bbox[1], bbox[2], bbox[3]
                    # cropped frame blurred and gray scaled
                    person_crop = frame[y_start:y_end, x_start:x_end]
                    person_crop_gray = cv2.cvtColor(person_crop, cv2.COLOR_BGR2GRAY)
                    person_crop_gray_blur = cv2.GaussianBlur(person_crop_gray, (3, 3), 0)
                    # kernel can be (5,5) for more blur
                    kernel = np.ones((3, 3), np.uint8)
                    eroded = cv2.erode(person_crop_gray_blur, kernel, iterations=1)
                    dilated = cv2.dilate(eroded, kernel, iterations=1)
                    edges = cv2.Canny(image=dilated, threshold1=100, threshold2=300, L2gradient=True)
                    mask = cv2.inRange(person_crop, lower_color, upper_color)
                    silhouette_mask = cv2.bitwise_and(edges, mask)
                    contours, _ = cv2.findContours(silhouette_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        for contour in contours:
                            contour[:, :, 0] += x_start
                            contour[:, :, 1] += y_start
                            # misschien goed om te proberen bij lower half of ROI en dan alleen horizontale lijnen
                            ar = aspect_ratio(contour)
                            if 0.1 <= ar <= 10:  # Adjust the aspect ratio range based on your needs
                                cv2.drawContours(frame_silhouette, [contour], -1, (0, 255, 0), 2)
                            cv2.fillPoly(shadow_mask, [contour], shadow_color)
                shadow_mask = cv2.addWeighted(frame_shadow, 0.5, shadow_mask, 50, 0)
                frame_shadow = cv2.add(frame_shadow, shadow_mask)
                if record:
                    # record the shadow frame and keep notion of the frames
                    out.write(frame_shadow)
                    frame_count += 1
                    if frame_count >= expected_frames:
                        break
                output_frame("silhouette", frame_silhouette)
                output_frame("shadow", frame_shadow)
            if cv2.waitKey(delay) == ord('q'):
                break
            if record_time is not None and time.time() - start_time >= record_time:
                break
        if record:
            out.release()  # Release the video writer
        cv2.destroyAllWindows()  # Close all OpenCV windows


if __name__ == "__main__":
    # create_depthai_pipeline(shadow_color=(255, 255, 255), delay=1, record=True, record_time=10)
    create_depthai_pipeline(shadow_color=(255, 255, 255), delay=1)
    # create_depthai_pipeline(shadow_color=(255, 255, 255), delay=60) #HAWK
    # create_depthai_pipeline(shadow_color=(255, 255, 255), delay=1, record=True, record_time=60)
