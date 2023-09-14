import cv2

def crop_region_of_interest(
        image, 
        x=0,
        y=0,
        w=150,
        h=150,
        extra_size=50,
        save_path=None,
        show_image=False
) -> any:
    roi = image[y-extra_size:y+h+extra_size, x-extra_size:x+w+extra_size]

    if roi.shape[0] > 0 and roi.shape[1] > 0:
        if show_image:
            cv2.imshow('face', roi)
        if save_path:
            cv2.imwrite(save_path, roi)
        return roi
    else:
        roi = image[y:y+h, x:x+w]
        if roi.shape[0] > 0 and roi.shape[1] > 0:
            if show_image:
                cv2.imshow('face', roi)
            if save_path:
                cv2.imwrite(save_path, roi)
            return roi
        else:
            return None
