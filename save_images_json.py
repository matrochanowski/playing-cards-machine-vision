from collections import defaultdict
import json

with open('jsons/_annotations_trainset.json', 'r') as file:
    data = json.load(file)

all_objects = dict()

# Grupowanie adnotacji według image_id dla szybszego dostępu
annotations_by_image = defaultdict(list)
for annotation in data['annotations']:
    annotations_by_image[annotation['image_id']].append({
        'bbox': annotation['bbox'],
        'class_id': annotation['category_id']
    })

# Iteracja przez obrazy i dodawanie adnotacji
for image in data['images']:
    image_id = image['id']
    all_objects[image_id] = {
        'path': image['file_name'],
        'annotations': annotations_by_image[image_id]  # Lista adnotacji dla danego obrazu
    }

with open('jsons/images.json', 'w') as outfile:
    json.dump(all_objects, outfile)
