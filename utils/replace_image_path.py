import json
import argparse

def process_eval_data(input_path, output_path, old_prefix, new_prefix):
    """Process evaluation data in JSONL format"""
    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        for line in infile:
            item = json.loads(line)
            if item["image"].startswith(old_prefix):
                item["image"] = item["image"].replace(old_prefix, new_prefix, 1)
            outfile.write(json.dumps(item) + '\n')

def process_train_data(input_path, output_path, old_prefix, new_prefix):
    """Process training data in JSON format"""
    with open(input_path, 'r') as f:
        data = json.load(f)

    for item in data:
        if "image_path" in item and item["image_path"].startswith(old_prefix):
            item["image_path"] = item["image_path"].replace(old_prefix, new_prefix, 1)

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['eval', 'train'], required=True,
                      help='Specify whether to process evaluation or training data')
    parser.add_argument('--input_path', type=str, required=True,
                      help='Path to input file (JSONL for eval, JSON for train)')
    parser.add_argument('--output_path', type=str, required=True,
                      help='Path to output file')
    parser.add_argument('--old_prefix', type=str, default='/projectnb/ivc-ml/yuwentan/dataset',
                      help='Old path prefix to replace')
    parser.add_argument('--new_prefix', type=str, default='/your/custom/path',
                      help='New path prefix')

    args = parser.parse_args()

    if args.mode == 'eval':
        process_eval_data(args.input_path, args.output_path, args.old_prefix, args.new_prefix)
    else:
        process_train_data(args.input_path, args.output_path, args.old_prefix, args.new_prefix)

if __name__ == "__main__":
    main()

