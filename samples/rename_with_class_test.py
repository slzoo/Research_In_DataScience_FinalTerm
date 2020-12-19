import csv, os


def main(input_dir, width):
    for root, directories, files in os.walk(input_dir):
        for filename in files:
            filepath = os.path.join(root, filename)
            print(filepath)


with open('answer.csv', mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file, fieldnames=['Id', 'Class'])
    input_dir = 'dataset/test'
    class_dict = {}
    line_count = 0

    for row in csv_reader:
        if line_count == 0:
            # print(f'Column names are {", ".join(row)}')
            line_count += 1
        class_dict[str(row['Id'])[:-4]] = row['Class']
        # print(row['Id'], class_dict.get(str(row['Id'])[:-4]))
        # print(f'{row["Id"]}, {row["Class"]}')
        line_count += 1
    # print(f'Processed {line_count} lines.')

    for root, directories, files in os.walk(input_dir):
        for filename in files:
            # print(filename[:-8])
            if filename[:-8] in class_dict:
                # print(filename, class_dict[filename[:-8]])
                filepath = os.path.join(root, filename)
                # print(filepath, class_dict[filename[:-8]])
                # print("mv "+filepath+" "+input_dir+"/"+class_dict[filename[:-8]])
                os.system("mv "+filepath+" "+input_dir+"/"+class_dict[filename[:-8]])
"""
            filepath = os.path.join(root, filename)
            print(filepath)
"""
