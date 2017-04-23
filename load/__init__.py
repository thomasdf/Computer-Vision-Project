import os

base_dir = os.path.dirname(os.path.dirname(__file__))

car_path = base_dir + '/datasets/object-detection-crowdai/labels.csv'
sign_path = base_dir + '/signs/csv/signs.csv'

car_img_path = base_dir + '/datasets/object-detection-crowdai/'
sign_img_path = base_dir + '/datasets/traffic-signs/GTSRB/Final_Training/Images/'
labels = ['signs', 'Pedestrian', 'Car', 'Truck']
