from datasets import dataset

# Missing values between labels and bounding boxes¶
missing_values_bounding_boxes = train_dataset.find_missing_values_bounding_boxes()
missing_values_labels = train_dataset.find_missing_values_labels()

print("Number of missing values for bounding boxes:", missing_values_bounding_boxes)
print("Number of missing values for labels:", missing_values_labels)

 # Size of images
image_size_summary = train_dataset.summarize_image_sizes() #TODO(this is not good, this is the status after resize)

print("Image Size Summary:")
print("Minimum Width:", image_size_summary['min_width'])
print("Maximum Width:", image_size_summary['max_width'])
print("Mean Width:", image_size_summary['mean_width'])
print("Minimum Height:", image_size_summary['min_height'])
print("Maximum Height:", image_size_summary['max_height'])
print("Mean Height:", image_size_summary['mean_height'])

# Number of images
train_image_count = len(train_dataset)
test_image_count = len(test_dataset)

print("Number of images in the training dataset:", train_image_count)
print("Number of images in the test dataset:", test_image_count)

# Distribution of the number of labels belonging to an image
train_dataset.plot_label_per_image_distribution()

# Distribution of the labels
train_dataset.plot_label_distribution()

# Distribution of the size of the bounding boxes (with original image size)
train_dataset.plot_bounding_box_size_distribution()
