import numpy as np

# Crop brain
def crop_image_from_gray(img, gray_img, tol=5):
    mask = gray_img > tol

    
    check_shape = img[:,:,0][np.ix_(mask.any(1), mask.any(0))].shape[0]
    if (check_shape == 0):
        return img
    else:
        img1 = img[:,:,0][np.ix_(mask.any(1), mask.any(0))]
        img2 = img[:,:,1][np.ix_(mask.any(1), mask.any(0))]
        img3 = img[:,:,2][np.ix_(mask.any(1), mask.any(0))]
        img = np.stack([img1, img2, img3], axis=-1)
    return img
	
def cropimg(img, bony, sz=384):
    cropped_img = np.zeros(shape=(sz, sz, 3), dtype=np.uint8)
    img = crop_image_from_gray(img, bony)
    h, w, c = img.shape
    if not h < sz:
        img = img[:-(h-sz+2),:,:]
    if not w < sz:
        img = img[:,:-(w-sz+2),:]
    h, w, c = img.shape
    if h % 2 != 0:
        img = img[1:,:,:]
    if w % 2 != 0:
        img = img[:,1:,:]
    h, w, c = img.shape
    h_diff = (sz - h) // 2
    w_diff = (sz - w) // 2
    cropped_img[h_diff:-h_diff, w_diff:-w_diff ,:] = img
    return cropped_img