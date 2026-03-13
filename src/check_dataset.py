import os

dataset_path = "data/shanghaitech"

print("Dataset path exists:", os.path.exists(dataset_path))

training_path = os.path.join(dataset_path, "training")
testing_path = os.path.join(dataset_path, "testing")

print("Training path:", training_path)
print("Testing path:", testing_path)

print("Training exists:", os.path.exists(training_path))
print("Testing exists:", os.path.exists(testing_path))

if os.path.exists(training_path):
    print("Training folders:", len(os.listdir(training_path)))

if os.path.exists(testing_path):
    print("Testing folders:", len(os.listdir(testing_path)))