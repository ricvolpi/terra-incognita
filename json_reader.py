import sys
import os

import json
import numpy as np

def extract_metadata(path_to_json):

	with open(path_to_json) as json_file:
		data = json.load(json_file)

	file_names = [str(item['file_name']) for item in data['images']]
	locations = [int(item['location']) for item in data['images']]
	labels = [int(item['category_id']) for item in data['annotations']]
	category_names = [str(item['name']) for item in data['categories']]
	category_labels = [int(item['id']) for item in data['categories']]

	return file_names, locations, labels, category_names, category_labels

data_dir = '/data/rvolpi/CaltechCameraTraps/ECCV2018'

all_file_names, all_locations, all_labels, all_category_names, all_category_labels = extract_metadata(os.path.join(data_dir,'annotations/CaltechCameraTrapsECCV18.json'))
cis_test_file_names, cis_test_locations, cis_test_labels, cis_test_category_names, cis_test_category_labels = extract_metadata(os.path.join(data_dir,'annotations/cis_test_annotations.json'))
cis_val_file_names, cis_val_locations, cis_val_labels, cis_val_category_names, cis_val_category_labels = extract_metadata(os.path.join(data_dir,'annotations/cis_val_annotations.json'))
trans_test_file_names, trans_test_locations, trans_test_labels, trans_test_category_names, trans_test_category_labels = extract_metadata(os.path.join(data_dir,'annotations/trans_test_annotations.json'))
trans_val_file_names, trans_val_locations, trans_val_labels, trans_val_category_names, trans_val_category_labels = extract_metadata(os.path.join(data_dir,'annotations/trans_val_annotations.json'))
train_file_names, train_locations, train_labels, train_category_names, train_category_labels = extract_metadata(os.path.join(data_dir,'annotations/train_annotations.json'))

pass


