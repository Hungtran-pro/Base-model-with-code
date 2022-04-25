from http.client import ImproperConnectionState
import numpy as np
from sklearn import linear_model #for logistic regression
from sklearn.metrics import accuracy_score # for evaluation
from scipy import misc # for loading image
import imageio
np.random.seed(1) #for fixing random values

path = '/home/trhun/Documents/ML_sample/Face/randomfaces4ar/'
train_ids = np.arange(1, 26)
test_ids = np.arange(26, 50)
view_ids = np.hstack((np.arange(1, 8), np.arange(14, 21))) #Stack arrays in sequence horizontally (column wise).

D = 165*120 # original dimension
d = 500 # new dimension 

# generate the projection matrix 
ProjectionMatrix = np.random.randn(D, d) 

#Xây dựng danh sách các tên file
def build_list_fn(pre, img_ids, view_ids):
    """
    INPUT:
        pre = 'M-' or 'W-'
        img_ids: indexes of images
        view_ids: indexes of views
    OUTPUT:
        a list of filenames 
    """
    list_fn = []
    for im_id in img_ids:
        for v_id in view_ids:
            fn = path + pre + str(im_id).zfill(3) + '-' + str(v_id).zfill(2) + '.bmp'
            list_fn.append(fn)
    return list_fn 

#Feature Extraction: Xây dựng dữ liệu cho training set và test set.
def rgb2gray(rgb): #đưa về  dạng ảnh xám
#     Y' = 0.299 R + 0.587 G + 0.114 B 
    return rgb[:,:,0]*.299 + rgb[:, :, 1]*.587 + rgb[:, :, 2]*.114

# feature extraction 
def vectorize_img(filename):    
    # load image 
    rgb = imageio.imread(filename)
    # convert to gray scale 
    gray = rgb2gray(rgb)
    # vectorization each row is a data point
    print(gray.shape())
    im_vec = gray.reshape(1, D)
    return im_vec 

def build_data_matrix(img_ids, view_ids):
    total_imgs = img_ids.shape[0]*view_ids.shape[0]*2 # for W + M
    X_full = np.zeros((total_imgs, D))
    y = np.hstack((np.zeros((int(total_imgs/2), )), np.ones((int(total_imgs/2), ))))
    
    list_fn_m = build_list_fn('M-', img_ids, view_ids)
    list_fn_w = build_list_fn('W-', img_ids, view_ids)
    list_fn = list_fn_m + list_fn_w 
    
    for i in range(len(list_fn)):
        X_full[i, :] = vectorize_img(list_fn[i])

    X = np.dot(X_full, ProjectionMatrix)
    return (X, y)

(X_train_full, y_train) = build_data_matrix(train_ids, view_ids)
x_mean = X_train_full.mean(axis = 0)
x_var  = X_train_full.var(axis = 0)

def feature_extraction(X):
    return (X - x_mean)/x_var     

X_train = feature_extraction(X_train_full)
X_train_full = None ## free this variable 

(X_test_full, y_test) = build_data_matrix(test_ids, view_ids)
X_test = feature_extraction(X_test_full)
X_test_full = None 