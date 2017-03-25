import pickle
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split
import cv2
import glob
import numpy as np
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip
import math


# HOG, color space, parameters, why
def get_hog_features(img, orient, pix_per_cell, cell_per_block, as_vector):
    features = hog(img,orientations=orient,pixels_per_cell=(pix_per_cell,pix_per_cell),cells_per_block=(cell_per_block,cell_per_block),transform_sqrt=True,visualise=False,feature_vector=as_vector)
    # print('hog:'+str(features.shape))
    return features


def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel()
    # print('spatial:'+str(features.shape))
    # Return the feature vector
    return features


def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    # print('color:'+str(hist_features.shape))
    return hist_features


def extract_features(imgs,orient,pix_per_cell,cell_per_block,hog_channel,color_channel=None,spatial_size=(32, 32),color_bins=32,get_hog=True):
    features=[]
    for f in imgs:
        this_img_features = []
        # read as BGR
        image = cv2.imread(f)
        features.append(extract_img_feature(image,
            orient,pix_per_cell,cell_per_block,hog_channel,color_channel,spatial_size,color_bins,get_hog))
    return features


def extract_img_feature(image,orient,pix_per_cell,cell_per_block,hog_channel,color_channel=None,spatial_size=(32, 32),color_bins=32,get_hog=True,given_hog_features=None):
    this_img_features = []
    # read as BGR
    if color_channel == 'HSV':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    elif color_channel == 'LUV':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
    elif color_channel == 'HLS':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    elif color_channel == 'YUV':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    elif color_channel == 'YCrCb':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    # hog features
    if get_hog:
        if hog_channel == 'ALL':
            for n in range(3):
                hog_features = get_hog_features(image[:,:,n], orient, pix_per_cell, cell_per_block,True)        
                this_img_features.append(hog_features)
        else:
            hog_features = get_hog_features(image[:,:,hog_channel], orient, pix_per_cell, cell_per_block,True)
            this_img_features.append(hog_features)
    else:
        this_img_features = given_hog_features

    # spatial bins features
    patial_features = bin_spatial(image, size=spatial_size)
    this_img_features.append(patial_features)

    # color histogram features
    hist_features = color_hist(image, nbins=color_bins)
    this_img_features.append(hist_features)
    return np.concatenate(this_img_features)


# sliding window search
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    if x_start_stop == [None,None]:
        x_start_stop = [0,img.shape[1]]
    if y_start_stop == [None,None]:
        y_start_stop = [0,img.shape[0]]
    x_span = x_start_stop[1]-x_start_stop[0]
    y_span = y_start_stop[1]-y_start_stop[0]
    x_step = int(xy_window[0] * xy_overlap[0])
    y_step = int(xy_window[1] * xy_overlap[1])
    x_windows = int((x_span-x_step) / x_step)
    y_windows = int((y_span-y_step) / y_step)
    window_list = []
    for x in range(x_windows):
        for y in range(y_windows):
            window_list.append(((x_start_stop[0]+x*x_step,y_start_stop[0]+y*y_step),(x_start_stop[0]+x*x_step+xy_window[0],y_start_stop[0]+xy_window[1]+y*y_step)))
    return window_list


def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy


def search_windows(img, windows, clf, scaler, color_channel='BGR', 
                    spatial_size=(32, 32), color_bins=32, 
                    orient=9, pix_per_cell=8, cell_per_block=2, 
                    hog_channel=0,get_hog=True):
    on_windows = []
    window_sizes = set([abs(i[1]-j[1]) for (i,j) in windows])
    hogs_size = {}
    for size in sorted(list(window_sizes)):
        hogs_size[size] = {}
        img_for_hog = np.copy(img)
        img_for_hog = cv2.resize(img_for_hog[400:660,200:,:],(size,size))
        hogs_size[size][0] = get_hog_features(img_for_hog[:,:,0], orient, pix_per_cell, cell_per_block, as_vector=False)
        hogs_size[size][1] = get_hog_features(img_for_hog[:,:,1], orient, pix_per_cell, cell_per_block, as_vector=False)
        hogs_size[size][2] = get_hog_features(img_for_hog[:,:,2], orient, pix_per_cell, cell_per_block, as_vector=False)

    for window in windows:
        print('-'*20)
        size = abs(window[0][0]-window[1][0])
        y1 = window[0][1]
        y2 = window[1][1]
        x1 = window[0][0]
        x2 = window[1][0]
        print(y1//64-1,y2//64-1,x1//64-1,x2//64-1)
        hog0 = hogs_size[size][0][:].ravel()
        hog1 = hogs_size[size][1][:].ravel()
        hog2 = hogs_size[size][2][:].ravel()
        all_hogs= [hog0,hog1,hog2]
        test_img = cv2.resize(img[y1:y2, x1:x2], (64, 64))
        print(hog0.shape)
        print(hogs_size[size][0].shape)
        print(size)
        features = extract_img_feature(test_img, orient,pix_per_cell,
                        cell_per_block,hog_channel,color_channel,
                        spatial_size,color_bins,get_hog=False,given_hog_features=all_hogs)
        print(features.shape)
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        prediction = clf.predict(test_features)
        if prediction == 1:
            on_windows.append(window)
    return on_windows


def find_cars(img, ystart, ystop, xstart, xstop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
    draw_img = np.copy(img)
    ctrans_tosearch = cv2.cvtColor(img[ystart:ystop,:,:],cv2.COLOR_BGR2HSV)
    
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell)-1
    nyblocks = (ch1.shape[0] // pix_per_cell)-1 
    nfeat_per_block = orient*cell_per_block**2
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell)-1 
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, as_vector=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, as_vector=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, as_vector=False)
    
    on_windows = []
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((hog_features, spatial_features, hist_features)).reshape(1, -1))    
            test_prediction = svc.predict(test_features)
            
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                # cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6) 
                if xbox_left > xstart:
                    on_windows.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))
    # cv2.imshow('asdf',draw_img)
    # cv2.waitKey(0)
                
    return on_windows


def train_model(orient=9,pix_per_cell=8,cell_per_block=2,hog_channel='ALL',color_channel='HSV',spatial_size=(32, 32),color_bins=32):
    # training
    car_training_images = glob.glob('vehicles//vehicles//*//*.png')
    cars = []
    for i in car_training_images:
        cars.append(i)

    non_car_training_images = glob.glob('non-vehicles//non-vehicles//*//*.png')
    non_cars = []
    for i in non_car_training_images:
        non_cars.append(i)

    np.random.shuffle(cars)
    np.random.shuffle(non_cars)

    sample_size=5000
    cars = cars[:sample_size]
    non_cars = non_cars[:sample_size]

    orient=9
    pix_per_cell=8
    cell_per_block=2
    hog_channel='ALL'
    color_channel='HSV'
    spatial_size=(32, 32)
    color_bins=32

    car_features = extract_features(cars,orient,pix_per_cell, cell_per_block,hog_channel,color_channel,spatial_size,color_bins)
    non_car_features = extract_features(non_cars,orient,pix_per_cell, cell_per_block,hog_channel,color_channel,spatial_size,color_bins)

    X = np.vstack((car_features,non_car_features)).astype(np.float64)
    X_scaler = StandardScaler().fit(X)
    scaled_X = X_scaler.transform(X)

    y = np.hstack((np.ones(len(car_features)), np.zeros(len(non_car_features))))

    X_train, X_test, y_train, y_test = train_test_split(scaled_X,y,test_size=0.2, random_state=33)

    svc = LinearSVC()
    svc.fit(X_train,y_train)

    print(svc.score(X_test,y_test))

    pickle.dump(svc,open("svc_trained.p","wb"))
    s = pickle.dumps(X_scaler)
    pickle.dump(X_scaler,open("scaler_trained.p","wb"))
    return svc, X_scaler


def load_trained():
    # save a ton of time by not training every run
    svc = pickle.load(open("svc_trained.p","rb"))
    X_scaler = pickle.load(open("scaler_trained.p","rb"))
    # print(svc.get_params())
    # print(X_scaler.get_params())
    return svc, X_scaler


def add_heat(heatmap, bbox_list):
    for box in bbox_list:
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    return heatmap

def apply_threshold(heatmap, threshold):
    heatmap[heatmap <= threshold] = 0
    return heatmap


def draw_labeled_bboxes(img, labels):
    for car_number in range(1, labels[1]+1):
        nonzero = (labels[0] == car_number).nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    return img

# video pipeline
def process_video_frames(image,trained_model,trained_normalizer,frame_class):
    orient=9
    pix_per_cell=8
    cell_per_block=2
    hog_channel='ALL'
    color_channel='HSV'
    spatial_size=(32, 32)
    color_bins=32
    svc = trained_model
    X_scaler = trained_normalizer
    image_bgr = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    windows = []
    # windows += slide_window(image, x_start_stop=[200, image.shape[1]], y_start_stop=[400, 660], 
    #                 xy_window=(64, 64), xy_overlap=(0.5, 0.5))
    # windows += slide_window(image, x_start_stop=[200, image.shape[1]], y_start_stop=[400, 660], 
    #                 xy_window=(96, 96), xy_overlap=(0.5, 0.5))
    # windows += slide_window(image, x_start_stop=[200, image.shape[1]], y_start_stop=[400, 660], 
    #                 xy_window=(128, 128), xy_overlap=(0.5, 0.5))
    # hot_windows = search_windows(image,windows,svc,X_scaler,color_channel, 
    #                     spatial_size, color_bins, orient, pix_per_cell, cell_per_block,hog_channel)
    # img, ystart, ystop, xstart, xstop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins
    hot_windows = []
    max_scale = 3.0
    for s in range(int(max_scale)):
        scale = s/(max_scale-1) +1
        hot_windows.extend(find_cars(img=image, ystart=int(400*math.sqrt(scale)), ystop=660, 
        xstart=600, xstop=image.shape[1], 
        scale=scale, svc=svc, X_scaler=X_scaler, orient=orient, pix_per_cell=pix_per_cell, 
        cell_per_block=cell_per_block, spatial_size=spatial_size, hist_bins=color_bins))
    heat = np.zeros_like(image[:,:,0]).astype(np.float)
    heatmap = add_heat(heat,hot_windows)
    # heat = apply_threshold(heat,7)
    # heatmap = np.clip(heat, 0, 255)
    frame_class.update_heatmap(heatmap)
    labels = label(frame_class.heatmap)
    detected_cars_frame = draw_labeled_bboxes(image, labels)
    return detected_cars_frame

def process_video(v_in,v_out):
    svc, X_scaler = load_trained()
    clip1 = VideoFileClip(v_in, audio=False)
    c = Car_Frame()
    detected_clip = clip1.fl_image(lambda x: process_video_frames(x,svc,X_scaler,c))
    detected_clip.write_videofile(v_out,audio=False)
    return

class Car_Frame:
    def __init__(self,shape=(720,1280)):
        self.shape = shape
        self.heatmap = np.zeros(self.shape).astype(np.float64)
        self.heatmap_history = []
    def update_heatmap(self,new_heatmap):
        self.heatmap_history.append(new_heatmap)
        new_sum = np.zeros(self.shape).astype(np.float64)
        for h in self.heatmap_history[-10:]:
            new_sum += h
        new_sum = np.clip(new_sum, 0, 255)
        self.heatmap = apply_threshold(new_sum,20)
        return


if __name__=='__main__':
    if 0:
        orient=9
        pix_per_cell=8
        cell_per_block=2
        hog_channel='ALL'
        color_channel='HSV'
        spatial_size=(32, 32)
        color_bins=32
        # train:
        if 0:
            svc, X_scaler = train_model(orient,pix_per_cell,cell_per_block,hog_channel,color_channel,spatial_size,color_bins)
        # load trained model:
        svc, X_scaler = load_trained()
        test_images = glob.glob('test_images//*.jpg')
        for test_image in test_images:
            image = cv2.imread(test_image)
            # windows = []
            # for size in range(90,110,5):
            #     windows += slide_window(image, x_start_stop=[200, image.shape[1]], y_start_stop=[400, 660], 
            #                     xy_window=(size, size), xy_overlap=(0.5, 0.5))
            # hot_windows = search_windows(image,windows,svc,X_scaler,color_channel, 
            #                     spatial_size, color_bins, orient, pix_per_cell, cell_per_block,hog_channel)
            hot_windows = []
            for s in range(8):
                scale = s/7.0+1
                hot_windows.extend(find_cars(img=image, ystart=int(400), ystop=660, 
                    xstart=200, xstop=image.shape[1], 
                    scale=scale, svc=svc, X_scaler=X_scaler, orient=orient, pix_per_cell=pix_per_cell, 
                    cell_per_block=cell_per_block, spatial_size=spatial_size, hist_bins=color_bins))
            heat = np.zeros_like(image[:,:,0]).astype(np.float)
            heat = add_heat(heat,hot_windows)
            heat = apply_threshold(heat,8)
            heatmap = np.clip(heat, 0, 255)
            labels = label(heatmap)
            cv2.imshow('cars',draw_boxes(image, hot_windows, color=(0, 0, 255), thick=6))
            cv2.waitKey(0)
            cv2.imshow('cars',draw_labeled_bboxes(image, labels))
            cv2.waitKey(0)
    if 1:
        process_video('test_video.mp4','test_video_output.mp4')
    if 1:
        process_video('project_video.mp4','project_video_output.mp4')